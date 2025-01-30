import tempfile
from pydub import AudioSegment, silence
import soundfile as sf
import torch
import torchaudio


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio


def remove_silence(ref_audio_orig, clip_short=True, show_info=print):

    aseg = AudioSegment.from_file(ref_audio_orig)

    if clip_short:
        # 1. try to find long silence for clipping
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        
    
    aseg = AudioSegment.silent(duration=400) + remove_silence_edges(aseg) + AudioSegment.silent(duration=400)
    sample = torch.tensor(aseg.get_array_of_samples())
    return sample

ref_audio_orig = "/speech/shoutrik/gpu17_18_backup/SAPCchallenge_16k/stroke_3e01c059-ccd9-40a2-5242-08dc17032f59_29479_4553.wav"
audio, sr = torchaudio.load(ref_audio_orig)
audio_no_silence = torchaudio.transforms.Vad(sample_rate=sr)(audio)

print(audio_no_silence.shape)

torchaudio.save("no_silence.wav", audio_no_silence, sr)

# remove_silence(ref_audio_orig)
