"""Train real, synthetic-only, fused or real-data-finetuned image reconstruction."""
import argparse
import copy
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from TVSD.config import load_config, save_config, set_seed
from TVSD.data import reconstruction_datasets
from TVSD.subjects import SUBJECT_NEURON_MAP
from TVSD.downstream_task.models.unet_cond_base import Unet
from TVSD.downstream_task.models.vqvae import VQVAE
from TVSD.downstream_task.scheduler.linear_noise_scheduler import LinearNoiseScheduler
from TVSD.downstream_task.core import train_one_step, validation_loss


def subject_config(config, subject):
    config = copy.deepcopy(config)
    sl = None
    if subject:
        info = SUBJECT_NEURON_MAP[subject]
        sl = slice(info['start'], info['end'])
        config['ldm_params']['condition_config']['neural_condition_config']['neural_embed_dim'] = info['count']
    return config, sl


def build_models(config, vae_checkpoint, device):
    # Preserve the reference RNG consumption: initialize U-Net before VQ-VAE.
    model = Unet(config['autoencoder_params']['z_channels'], config['ldm_params']).to(device)
    vae = VQVAE(config['dataset_params']['im_channels'], config['autoencoder_params']).to(device)
    vae.load_state_dict(torch.load(vae_checkpoint, map_location=device, weights_only=True))
    vae.eval()
    vae.requires_grad_(False)
    return model, vae


def build_scheduler(config):
    cfg = config['diffusion_params']
    return LinearNoiseScheduler(cfg['num_timesteps'], cfg['beta_start'], cfg['beta_end'])


def prepare_training_config(config, subject=None, finetune=False):
    config, subject_slice = subject_config(config, subject)
    if finetune:
        # train_finetune_from_synthetic.sh explicitly passes --no_timeseries.
        config['dataset_params'].pop('timeseries_h5_path', None)
    return config, subject_slice


def train(config, output, vae_checkpoint, regime='real', synthetic_path=None,
          subject=None, device='cuda', resume=None, finetune=False):
    config, subject_slice = prepare_training_config(config, subject, finetune)
    cfg = config['train_params']
    # The reference downstream trainer calls set_seed() with its default 42.
    set_seed(cfg.get('reconstruction_seed', 42))
    torch.backends.cudnn.deterministic = True
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output/'config.yaml')
    train_data, val_data, _ = reconstruction_datasets(config, regime, synthetic_path)
    kwargs = dict(batch_size=cfg['ldm_batch_size'], num_workers=cfg.get('num_workers', 1), pin_memory=True)
    train_loader = DataLoader(train_data, shuffle=True, **kwargs)
    val_loader = DataLoader(val_data, shuffle=False, **kwargs)
    model, vae = build_models(config, vae_checkpoint, device)
    optimizer = Adam(model.parameters(), lr=cfg['ldm_lr'])
    scheduler = build_scheduler(config)
    start, best = 0, float('inf')
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if not finetune:
            start, best = ckpt['epoch']+1, ckpt['best_val_loss']
    elif finetune:
        raise ValueError('finetune requires --resume with a synthetic-pretrained checkpoint')
    if finetune and regime != 'real':
        raise ValueError('finetune uses the real training split')
    drop = config['ldm_params']['condition_config']['neural_condition_config']['cond_drop_prob']
    for epoch in range(start, cfg['ldm_epochs']):
        losses = [train_one_step(model, vae, optimizer, scheduler, im, response,
                                 drop, subject_slice, device) for im, response in train_loader]
        loss = validation_loss(model, vae, val_loader, scheduler, subject_slice, device)
        improved = loss < best
        best = min(best, loss)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                       best_val_loss=best, val_loss=loss, config=config, subject=subject,
                       regime=regime, vae_checkpoint=str(vae_checkpoint))
        torch.save(payload, output/'latest.pth')
        if improved:
            torch.save(payload, output/'best.pth')
        if (epoch+1) % cfg.get('save_every', 5) == 0 or epoch+1 == cfg['ldm_epochs']:
            torch.save(payload, output/f'epoch_{epoch:04d}.pth')
        print(f'epoch={epoch} train={sum(losses)/len(losses):.6f} val={loss:.6f}', flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--vae-checkpoint', required=True)
    parser.add_argument('--regime', choices=['real', 'synthetic', 'fused'], default='real')
    parser.add_argument('--synthetic-path')
    parser.add_argument('--subject', choices=list(SUBJECT_NEURON_MAP))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume')
    parser.add_argument('--finetune', action='store_true')
    args = vars(parser.parse_args())
    config = load_config(args.pop('config'))
    train(config, **args)


if __name__ == '__main__':
    main()
