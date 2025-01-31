# training script.

import os
from importlib.resources import files
import wandb
import hydra
import torch
from f5_tts.model import CFM, DiT, Trainer, UNetT
from f5_tts.model.dataset import CustomDataset

os.chdir("/speech/shoutrik/dysarthric_exp/F5_dysarthric-se")  # change working directory to root of project (local editable)

os.environ["OMP_NUM_THREADS"] = str(4)
os.environ["MKL_NUM_THREADS"] = str(4)
torch.set_num_threads(4)

@hydra.main(version_base="1.3", config_path="/speech/shoutrik/dysarthric_exp/F5_dysarthric-se/src/f5_tts/configs/", config_name="F5TTS_enhance_Small_train")
def main(cfg):
    
    print("config --->")
    print(cfg)
    
    # tokenizer = cfg.model.tokenizer
    mel_spec_type = cfg.model.mel_spec.mel_spec_type
    exp_name = "F5TTS_small_LibriDys_noisy_clean_02" # change this for a new experiment
    metadata = "/speech/shoutrik/dysarthric_exp/F5_dysarthric-se/data/libri_noisy_dataset/train.parquet" # make sure all wav files of dysarthric dataset are within 0.3 to 30 sec duration
    ckpt_save_dir = f"/speech/shoutrik/dysarthric_exp/F5_dysarthric-se/ckpts/{exp_name}"
    wandb_resume_id = "zo4vzara" # replace with a new id for a new experiment. Generate with wandb.util.generate_id() after wandb.login(key=wandb_api_key)
    wandb_project="speech_enhancement" # don't change

    # set model
    if "F5TTS" in cfg.model.name:
        model_cls = DiT
    elif "E2TTS" in cfg.model.name:
        model_cls = UNetT

    model = CFM(
        transformer=model_cls(**cfg.model.arch, mel_dim=cfg.model.mel_spec.n_mel_channels),
        mel_spec_kwargs=cfg.model.mel_spec,
        pad_with_filler=False
    )

    # init trainer
    trainer = Trainer(
        model,
        epochs=cfg.optim.epochs,
        learning_rate=cfg.optim.learning_rate,
        num_warmup_updates=cfg.optim.num_warmup_updates,
        save_per_updates=cfg.ckpts.save_per_updates,
        checkpoint_path=ckpt_save_dir,
        batch_size=cfg.datasets.batch_size_per_gpu,
        batch_size_type=cfg.datasets.batch_size_type,
        max_samples=cfg.datasets.max_samples,
        grad_accumulation_steps=cfg.optim.grad_accumulation_steps,
        max_grad_norm=cfg.optim.max_grad_norm,
        logger=cfg.ckpts.logger,
        wandb_project=wandb_project,
        wandb_run_name=exp_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=cfg.ckpts.last_per_steps,
        log_samples=True,
        bnb_optimizer=cfg.optim.bnb_optimizer,
        mel_spec_type=mel_spec_type,
        is_local_vocoder=cfg.model.vocoder.is_local,
        local_vocoder_path=cfg.model.vocoder.local_path,
    )

    train_dataset = CustomDataset(
            metadata,
            **cfg.model.mel_spec,
        )
    
    trainer.train(
        train_dataset,
        num_workers=cfg.datasets.num_workers,
        resumable_with_seed=666, # seed for shuffling dataset
    )


if __name__ == "__main__":
    main()
