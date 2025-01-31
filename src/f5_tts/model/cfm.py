"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
    apply_vtln,
)




class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        pad_with_filler = False,
        pretraining = True
    ):
        super().__init__()
        
        print("initializing cfm2")

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.pad_with_filler = pad_with_filler

        self.clean_pad_embed = nn.Embedding(1, num_channels)
        # self.clean_pad_embed = nn.Parameter(torch.randn(1, num_channels))
        self.pretraining = pretraining
            
        

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        noisy_audio,
        duration,
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave
        
        print("\n\nshapes : ")
        print(cond.shape, noisy_audio.shape)
        # cond = cond[:, :24000]
        print(cond.shape, noisy_audio.shape)


        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

            
        if noisy_audio.ndim==2:
            noisy_audio = self.mel_spec(noisy_audio)
            noisy_audio = noisy_audio.permute(0, 2, 1)
            assert noisy_audio.shape[-1] == self.num_channels
            
        cond = cond.to(next(self.parameters()).dtype)
        noisy_audio = noisy_audio.to(next(self.parameters()).dtype)
        
        batch, seq_len, dtype, device = *cond.shape[:2], cond.dtype, cond.device
        assert batch == 1, f"batch size 1 is only supported but found {batch}"
        
        print(f"infering pretrained model : {self.pretraining}")
        print("shapes : ", cond.shape, noisy_audio.shape)

        noisy_max_len = noisy_audio.shape[1]
        cond_max_len = cond.shape[1]
        padding = noisy_max_len - cond_max_len
        cond = F.pad(cond, (0, 0, 0, padding), value=0)
        cond_mask = torch.ones_like(cond, dtype=torch.bool)
        cond_mask[:, cond_max_len:, :] = False
        if self.pretraining:
            assert cond.shape == noisy_audio.shape, f"cond and noisy_audio shape should match, {cond.shape}, {noisy_audio.shape}"
        
        else:
            assert duration is not None
            cond = cond[:, cond_max_len:duration] = self.clean_pad_embed
            
        mask = None
        
        def fn(t, x):
            pred = self.transformer(
                x=x, cond=cond, noisy_mel=noisy_audio, time=t, mask=mask, drop_audio_cond=False, drop_noisy_mel=False
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=cond, noisy_mel=noisy_audio, time=t, mask=mask, drop_audio_cond=True, drop_noisy_mel=True
            )
            return pred + (pred - null_pred) * cfg_strength

        y0 = []
        if exists(seed):
            torch.manual_seed(seed)
        y0.append(torch.randn(noisy_max_len, self.num_channels, device=self.device, dtype=cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0


        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # print("dtypes :", y0.dtype, t.dtype)
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)

        sampled = trajectory[-1]
        out = sampled
        out = torch.where(cond_mask, cond, out)
        if not self.pretraining:
            out = out[:, :duration, :]
        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)
        return out, trajectory  


    def forward(
        self,
        clean_mel, # shape : list((d,n))
        clean_mel_lengths, # shape : (b)
        noisy_mel, # shape : (b, n, d)
        noisy_mel_lengths, # shape : (b)
        noise_scheduler: str | None = None,
    ):


        batch, dtype, device, _σ1 = len(clean_mel), noisy_mel.dtype, self.device, self.sigma        
        max_noisy_mel = torch.max(noisy_mel_lengths)
            
        if self.pad_with_filler:
            padded_clean_mel = []        
            for mel in clean_mel:
                mel = mel.transpose(0,1).to(dtype) # d,n -> n,d
                if mel.shape[0] < max_noisy_mel:
                    pad_size = max_noisy_mel - mel.shape[0]
                    padding = self.clean_pad_embed[0].repeat(pad_size, 1)
                    padded_mel = torch.cat([mel, padding], dim=0)
                else:
                    padded_mel = mel[:max_noisy_mel]
                padded_clean_mel.append(padded_mel)
            padded_clean_mel = torch.stack(padded_clean_mel)
        else:
            padded_clean_mel = clean_mel
            assert padded_clean_mel.shape == noisy_mel.shape

        seq_len = padded_clean_mel.shape[1]
        mask = lens_to_mask(clean_mel_lengths, length=seq_len)

        frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        rand_span_mask = mask_from_frac_lengths(clean_mel_lengths, frac_lengths, max_len = seq_len)

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = padded_clean_mel

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        number_of_zeros = (cond == 0).sum().item()

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_noisy_mel = True
        else:
            drop_noisy_mel = False

        # if want rigourously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, noisy_mel=noisy_mel, time=time, drop_audio_cond=drop_audio_cond, drop_noisy_mel=drop_noisy_mel
        )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]
        return loss.mean(), cond, pred
