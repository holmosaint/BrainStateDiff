"""Generate matched-state or cross-state neural responses with fixed real evaluation splits."""
import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from TVSD.config import load_config, set_seed
from TVSD.data import timeseries_datasets, dataset_paths, read_pickle
from TVSD.splits import load_split_indices
from TVSD.subjects import SUBJECT_NEURON_MAP
from TVSD.diffusion.batching import build_dataloader
from TVSD.diffusion.core import compute_neural_stats
from TVSD.diffusion.train import build_model, build_scheduler
from TVSD.diffusion.sampling import sample_batch, CrossConditionDataset, make_cross_indices


def generate_dataset(model, scheduler, dataset, stats, config, device,
                     steps=50, guidance_scale=1.0, batch_size=None):
    cfg, temporal = config['train_params'], config['temporal_params']
    loader = build_dataloader(dataset, batch_size=batch_size or cfg['batch_size'],
        baseline_bins=temporal['baseline_bins'], target_bins=temporal['target_bins'],
        shuffle=False, seed=cfg['seed'], num_workers=cfg['num_workers'], drop_last=False)
    collected = {0: [], 1: []}
    for images, baseline, _, subjects, mask in loader:
        result = sample_batch(model, scheduler, images, baseline, subjects, mask,
                              stats, steps, guidance_scale, torch.device(device))
        for sid, value in result.items():
            collected[sid].append(value.cpu().numpy())
    result = np.empty((len(dataset), 315), dtype=np.float32)
    for info in SUBJECT_NEURON_MAP.values():
        result[:, info['start']:info['end']] = np.concatenate(collected[info['subject_id']])
    return result


def sample(config, checkpoint, output, mode='matched', factor=1, seed=42,
           cross_seed=0, device='cuda', steps=50, guidance_scale=1.0,
           use_ema=True, stats_path=None, batch_size=None):
    if mode not in ('matched', 'cross') or factor < 1:
        raise ValueError('mode must be matched/cross and factor must be positive')
    set_seed(seed)
    train, _, _ = timeseries_datasets(config)
    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt['ema' if use_ema else 'model'])
    model.eval()
    if stats_path:
        stats = torch.load(stats_path, map_location='cpu', weights_only=False)
    elif 'neural_stats' in ckpt:
        stats = ckpt['neural_stats']
    else:
        temporal = config['temporal_params']
        stats = compute_neural_stats(train, temporal['baseline_bins'], temporal['target_bins'])
    scheduler = build_scheduler(config)
    real_path, _ = dataset_paths(config)
    real = read_pickle(real_path)
    split = load_split_indices(config['dataset_params'].get('split_file'))
    stimuli = [real['train_stimuli'][i] for i in split['train_indices']]
    activities, labels, image_indices, baseline_indices = [], [], [], []
    for r in range(factor):
        dataset = train
        order = list(range(len(train)))
        if mode == 'cross':
            # Both permutations and seed offsets match the reference sampler.
            order = make_cross_indices(len(train), cross_seed+r)
            donor = make_cross_indices(len(train), cross_seed+factor+r)
            dataset = CrossConditionDataset(train, order, donor)
            image_indices.append(order)
            baseline_indices.append(donor)
        activities.append(generate_dataset(model, scheduler, dataset, stats, config,
                                            device, steps, guidance_scale, batch_size))
        labels.extend(stimuli[i] for i in order)
    payload = dict(train_stimuli=labels, train_activity=np.concatenate(activities),
                   val_stimuli=[real['train_stimuli'][i] for i in split['val_indices']],
                   val_activity=real['train_activity'][split['val_indices']],
                   test_stimuli=real['test_stimuli'], test_activity=real['test_activity'])
    if mode == 'cross':
        payload.update(cross_image_indices=image_indices, cross_baseline_indices=baseline_indices)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'wb') as handle:
        pickle.dump(payload, handle)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['matched', 'cross'], default='matched')
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cross-seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stats', dest='stats_path')
    parser.add_argument('--no-ema', action='store_true')
    args = vars(parser.parse_args())
    config = load_config(args.pop('config'))
    args['use_ema'] = not args.pop('no_ema')
    sample(config, **args)


if __name__ == '__main__':
    main()
