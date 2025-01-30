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



class CustomDataset(Dataset):
    def __init__(
        self,
        metadata,
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
        
        self.data = ds.dataset(metadata, format="parquet")
        self.data = self.data.to_table()


        self.mel_spectrogram = MelSpec(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=mel_spec_type,
            )
        

    def get_frame_len(self, index):
        duration = self.data["duration"][index].as_py()
        return duration * self.target_sample_rate / self.hop_length

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, index):
        while True:
            row = {col:self.data[col][index].as_py() for col in self.data.column_names}
            clean_path = row["clean_wav_path"]
            noisy_path = row["noisy_wav_path"]
            duration = row["duration"]

            if 0.3 <= duration <= 30:
                break  # valid

            index = (index + 1) % len(self.data)

        # process noisy speech
        if noisy_path.split(".")[1] in ["wav", "mp3", "flac"]:  
            noisy_audio, source_sample_rate = torchaudio.load(noisy_path)
            if noisy_audio.shape[0] > 1:
                noisy_audio = torch.mean(noisy_audio, dim=0, keepdim=True)
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                noisy_audio = resampler(noisy_audio)
            noisy_mel_spec = self.mel_spectrogram(noisy_audio)
            noisy_mel_spec = noisy_mel_spec.squeeze(0)  # '1 d t -> d t'
        elif noisy_path.endswith(".npy"):
            noisy_mel_spec = np.load(noisy_path)
            
            
        # process clean speech
        if clean_path.split(".")[1] in ["wav", "mp3", "flac"]:  
            clean_audio, source_sample_rate = torchaudio.load(clean_path)
            if clean_audio.shape[0] > 1:
                clean_audio = torch.mean(clean_audio, dim=0, keepdim=True)
            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                clean_audio = resampler(clean_audio)
            clean_mel_spec = self.mel_spectrogram(clean_audio)
            clean_mel_spec = clean_mel_spec.squeeze(0)
        elif clean_path.endswith(".npy"):
            clean_mel_spec = np.load(clean_path)
            clean_mel_spec = torch.from_numpy(clean_mel_spec)

        return {
            "mel_spec": clean_mel_spec, 
            "noisy_mel_spec": noisy_mel_spec,
        }


# Dynamic Batch Sampler
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
            frame_len = data_source.get_frame_len(idx)
            if frame_len < 2813: # 30 sec at 24kHz and 256 hop length
                indices.append((idx, frame_len))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        max_frame_len = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames and max batch size of {max_samples} per gpu"
        ):
            max_frame_len = max(max_frame_len, frame_len)
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples) and (max_frame_len*len(batch)<=frames_threshold):
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
        # random.seed(random_seed)
        # random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# collation
def collate_fn(batch):
    noisy_mel_specs = [item["noisy_mel_spec"] for item in batch]
    noisy_mel_lengths = torch.LongTensor([spec.shape[-1] for spec in noisy_mel_specs])
    clean_mel_specs = [item["mel_spec"] for item in batch]
    clean_mel_lengths = torch.LongTensor([spec.shape[-1] for spec in clean_mel_specs])
    
    # print("noisy : ", [a.shape for a in noisy_mel_specs])
    # print("clean : ", [a.shape for a in clean_mel_specs])
    
    max_noisy_mel_length = noisy_mel_lengths.amax()
    max_clean_mel_length = clean_mel_lengths.amax()
    
    # speaker_id = torch.LongTensor([item["speaker_id"] for item in batch])

    padded_noisy_mel_specs = []
    for spec in noisy_mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_noisy_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_noisy_mel_specs.append(padded_spec)

    noisy_mel_specs = torch.stack(padded_noisy_mel_specs)
    
    padded_clean_mel_specs = []
    for spec in clean_mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_noisy_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_clean_mel_specs.append(padded_spec)

    clean_mel_specs = torch.stack(padded_clean_mel_specs)
    
    # print(noisy_mel_specs.shape, clean_mel_specs.shape)


    return dict(
        noisy_mel=noisy_mel_specs,
        clean_mel=clean_mel_specs,
        clean_mel_lengths=clean_mel_lengths,
        noisy_mel_lengths=noisy_mel_lengths
        # speaker_id=speaker_id,
    )
    
    
# # testing components --->
# metadata = "/speech/shoutrik/torch_exp/F5_tts/F5-TTS/data/SAPC_data/test_data/metadata.csv"
# dataset = CustomDataset(metadata)
# from torch.utils.data import SequentialSampler
# sampler = SequentialSampler(dataset)

# batch_sampler = DynamicBatchSampler(sampler, 38400, max_samples=64, random_seed=42, drop_last=False)

# train_dataloader = DataLoader(
#     dataset,
#     collate_fn=collate_fn,
#     num_workers=4,
#     pin_memory=True,
#     persistent_workers=True,
#     batch_sampler=batch_sampler,
# )


# for x in train_dataloader:
    
#     print(x["noisy_mel"].shape)
#     print(x["noisy_mel_lengths"])
#     print(len(x["clean_mel"]))
#     print(x["clean_mel"][0].shape)
#     print(x["clean_mel_lengths"])
#     break