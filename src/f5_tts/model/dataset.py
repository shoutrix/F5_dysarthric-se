import json
import random
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
# from datasets import Dataset as Dataset_
# from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler, DataLoader
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default
import pandas as pd
import numpy as np

import pyarrow.dataset as ds
import pyarrow.feather as feather


class CustomDataset(Dataset):
    def __init__(
        self,
        metadata_path,
        target_sample_rate=16000,
        hop_length=160,
        n_mel_channels=80,
        n_fft=512,
        win_length=512,
        mel_spec_type="vocos",
    ):
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.n_mel_channels = n_mel_channels

        self.table = feather.read_table(metadata_path, memory_map=True)

        self.ids = self.table.column("id_").to_pylist()
        self.clean_wavs = self.table.column("clean_wav")
        self.noisy_wavs = self.table.column("noisy_wav")

        # self.mel_spectrogram = MelSpec(
        #     n_fft=n_fft,
        #     hop_length=hop_length,
        #     win_length=win_length,
        #     n_mel_channels=n_mel_channels,
        #     target_sample_rate=target_sample_rate,
        #     mel_spec_type=mel_spec_type,
        # )

    def __len__(self):
        return len(self.ids)
    
    
    def get_frame_len(self, index):
        sample = self.noisy_wavs[index].as_py()
        duration = np.frombuffer(sample, dtype=np.float32).reshape(-1, self.n_mel_channels)
        return duration.shape[0]
        

    def __getitem__(self, index):
        
        clean_bytes = self.clean_wavs[index].as_py()
        noisy_bytes = self.noisy_wavs[index].as_py()

        clean_mel_spec = np.frombuffer(clean_bytes, dtype=np.float32).reshape(-1, self.n_mel_channels)
        noisy_mel_spec = np.frombuffer(noisy_bytes, dtype=np.float32).reshape(-1, self.n_mel_channels)
        
        # print(clean_mel_spec.shape, noisy_mel_spec.shape)
        # print(crash)

        return {
            "mel_spec": torch.tensor(clean_mel_spec), 
            "noisy_mel_spec": torch.tensor(noisy_mel_spec),
        }


class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)

        del indices

        # if want to have different batches between epochs, may just set a seed and log it in ckpt
        # cuz during multi-gpu training, although the batch on per gpu not change between epochs, the formed general minibatch is different
        # e.g. for epoch n, use (random_seed + n)
        random.seed(random_seed)
        random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# collation
def collate_fn(batch):
    noisy_mel_specs = [item["noisy_mel_spec"] for item in batch]
    noisy_mel_lengths = torch.LongTensor([spec.shape[0] for spec in noisy_mel_specs])
    clean_mel_specs = [item["mel_spec"] for item in batch]
    clean_mel_lengths = torch.LongTensor([spec.shape[0] for spec in clean_mel_specs])
    
    # print("noisy : ", [a.shape for a in noisy_mel_specs])
    # print("clean : ", [a.shape for a in clean_mel_specs])
    
    max_noisy_mel_length = noisy_mel_lengths.amax()
    max_clean_mel_length = clean_mel_lengths.amax()
    
    # speaker_id = torch.LongTensor([item["speaker_id"] for item in batch])

    padded_noisy_mel_specs = []
    for spec in noisy_mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, 0, 0, max_noisy_mel_length - spec.size(0))
        padded_spec = F.pad(spec, padding, value=0)
        padded_noisy_mel_specs.append(padded_spec)

    noisy_mel_specs = torch.stack(padded_noisy_mel_specs)
    
    # print(noisy_mel_specs.shape, clean_mel_specs.shape)

    # print(noisy_mel_specs.shape)
    
    return dict(
        noisy_mel=noisy_mel_specs,
        clean_mel=clean_mel_specs,
        clean_mel_lengths=clean_mel_lengths,
        noisy_mel_lengths=noisy_mel_lengths
    )
