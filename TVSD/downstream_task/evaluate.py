"""Evaluate a reconstruction checkpoint on real validation or test responses."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from TVSD.config import load_config, set_seed
from TVSD.data import reconstruction_datasets
from TVSD.subjects import SUBJECT_NEURON_MAP
from TVSD.downstream_task.train import subject_config, build_models, build_scheduler
from TVSD.downstream_task.core import reconstruct, compute_pair_metrics
from TVSD.downstream_task.models.lpips import LPIPS


def evaluate(config, checkpoint, vae_checkpoint, output, split='test', subject=None,
             device='cuda', batch_size=32, synthetic_path=None):
    config, sl = subject_config(config, subject)
    set_seed(config['train_params'].get('reconstruction_seed', 42))
    datasets = reconstruction_datasets(config, 'synthetic' if synthetic_path else 'real', synthetic_path)
    dataset = datasets[1 if split == 'val' else 2]
    model, vae = build_models(config, vae_checkpoint, device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    # Both bare upstream state dicts and training checkpoint payloads are supported.
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    scheduler = build_scheduler(config)
    perceptual = LPIPS(net='vgg').to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=config['train_params'].get('num_workers', 1))
    rows = []
    for images, responses in loader:
        if sl is not None:
            responses = responses[:, sl]
        predictions = reconstruct(model, vae, scheduler, responses,
                                   config['dataset_params']['im_size'], config['autoencoder_params'], device)
        for prediction, truth in zip(predictions, (images+1)/2):
            psnr, lpips = compute_pair_metrics(prediction, truth, perceptual)
            rows.append(dict(image_id=len(rows), PSNR=psnr, LPIPS=lpips))
    if not rows:
        raise ValueError('Evaluation split is empty')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'per_image.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['image_id', 'PSNR', 'LPIPS'])
        writer.writeheader()
        writer.writerows(rows)
    summary = dict(split=split, subject=subject, n=len(rows), checkpoint=str(checkpoint),
                    PSNR=float(np.mean([r['PSNR'] for r in rows])),
                    LPIPS=float(np.mean([r['LPIPS'] for r in rows])))
    (output/'metrics.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--vae-checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--subject', choices=list(SUBJECT_NEURON_MAP))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--synthetic-path', help='Use the unchanged real val/test arrays retained in this synthetic pickle')
    args = vars(parser.parse_args())
    config = load_config(args.pop('config'))
    print(json.dumps(evaluate(config, **args), indent=2))


if __name__ == '__main__':
    main()
