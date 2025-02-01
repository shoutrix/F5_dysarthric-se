#!/bin/bash

python infer_pretrained.py \
    --config /speech/shoutrik/dysarthric_exp/F5_dysarthric-se/src/f5_tts/infer/examples/basic/basic.toml \
    --model F5-TTS \
    --model_cfg /speech/shoutrik/dysarthric_exp/F5_dysarthric-se/src/f5_tts/configs/F5TTS_enhance_Small_train.yaml \
    --ckpt_file /speech/shoutrik/dysarthric_exp/F5_dysarthric-se/ckpts/F5TTS_small_LibriDys_noisy_clean_02/model_last.pt \
    --pretrain True \
    --ref_clean_audio /speech/Database/LibriTTS/LibriTTS/test-clean/121/127105/121_127105_000022_000001.wav \
    --ref_noisy_audio /speech/shoutrik/speech_enhancement/libri_960_noise_data/noisy_dataset_test/15dB/121_127105_000022_000001.wav \
    --gen_file /speech/shoutrik/speech_enhancement/libri_960_noise_data/noisy_dataset_test/wav.scp \
    --output_dir /speech/shoutrik/dysarthric_exp/F5_dysarthric-se/src/f5_tts/infer/sample_test \
    # --fix_duration 12 \