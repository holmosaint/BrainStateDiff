"""Train the state-conditioned V4 diffusion transformer: python -m TVSD.diffusion.train."""
import argparse
from pathlib import Path

import torch
from diffusers import DDIMScheduler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from TVSD.config import load_config, save_config, set_seed
from TVSD.data import timeseries_datasets
from TVSD.diffusion.batching import build_dataloader
from TVSD.diffusion.core import EMA, compute_neural_stats, train_one_step, evaluate
from TVSD.diffusion.model import NeuralDiTTS


def build_model(config):
    return NeuralDiTTS(**config['model_params'],
                      cond_drop_prob=config['train_params']['cond_drop_prob'])


def build_scheduler(config):
    cfg = config['diffusion_params']
    return DDIMScheduler(num_train_timesteps=cfg['num_timesteps'],
                         beta_start=cfg['beta_start'], beta_end=cfg['beta_end'],
                         beta_schedule='scaled_linear', clip_sample=False,
                         prediction_type=cfg['prediction_type'])


def build_optimizer(model, config, steps_per_epoch):
    cfg = config['train_params']
    lr, warmup = cfg['lr'], cfg['warmup_steps']
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad],
                      lr=lr, weight_decay=cfg['weight_decay'])
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6/max(lr, 1e-8),
                               end_factor=1.0, total_iters=warmup)
    cosine = CosineAnnealingLR(optimizer,
                              T_max=max(1, cfg['num_epochs']*steps_per_epoch-warmup), eta_min=1e-6)
    return optimizer, SequentialLR(optimizer, [warmup_scheduler, cosine], milestones=[warmup])


def train(config, output, device='cuda', resume=None):
    device = torch.device(device)
    cfg = config['train_params']
    set_seed(cfg['seed'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_config(config, output/'config.yaml')
    train_data, val_data, _ = timeseries_datasets(config)
    temporal = config['temporal_params']
    stats = compute_neural_stats(train_data, temporal['baseline_bins'], temporal['target_bins'])
    torch.save(stats, output/'neural_stats.pt')
    loader_args = dict(batch_size=cfg['batch_size'], baseline_bins=temporal['baseline_bins'],
                       target_bins=temporal['target_bins'], seed=cfg['seed'], num_workers=cfg['num_workers'])
    train_loader = build_dataloader(train_data, **loader_args)
    val_loader = build_dataloader(val_data, **loader_args, shuffle=False, drop_last=False)
    if len(train_loader) == 0:
        raise ValueError('Training data must contain at least one complete batch per monkey')
    model = build_model(config).to(device)
    scheduler = build_scheduler(config)
    optimizer, lr_scheduler = build_optimizer(model, config, len(train_loader))
    ema = EMA(model, cfg['ema_decay'])
    start, best = 0, float('inf')
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
        ema.load_state_dict(ckpt['ema'])
        start, best = ckpt['epoch']+1, ckpt['best_val_loss']
        # Restore new-format RNG states when present; original checkpoints omit these.
        if 'rng' in ckpt:
            torch.set_rng_state(ckpt['rng'].cpu())
    for epoch in range(start, cfg['num_epochs']):
        train_loader.batch_sampler.set_epoch(epoch)
        losses = []
        for batch in train_loader:
            losses.append(train_one_step(model, scheduler, optimizer, *batch, stats, device))
            ema.update(model)
            lr_scheduler.step()
        val_loss, _ = evaluate(model, scheduler, val_loader, stats, device)
        improved = val_loss < best
        best = min(best, val_loss)
        payload = dict(epoch=epoch, model=model.state_dict(), ema=ema.state_dict(),
                       optimizer=optimizer.state_dict(), lr_scheduler=lr_scheduler.state_dict(),
                       best_val_loss=best, val_loss=val_loss, config=config,
                       neural_stats=stats, rng=torch.get_rng_state())
        torch.save(payload, output/'latest.pt')
        if improved:
            torch.save(payload, output/'best.pt')
        if (epoch+1) % cfg['save_every_epochs'] == 0 or epoch+1 == cfg['num_epochs']:
            torch.save(payload, output/f'epoch_{epoch:04d}.pt')
        print(f'epoch={epoch} train={sum(losses)/len(losses):.6f} val={val_loss:.6f}', flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume')
    args = parser.parse_args()
    train(load_config(args.config), args.output, args.device, args.resume)


if __name__ == '__main__':
    main()
