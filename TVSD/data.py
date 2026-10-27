"""TVSD images and neural labels, with the original fixed train/validation split.

Images are the first RGB frame of each stimulus MP4, resized and mapped to [-1,1].
The pickle and HDF5 are trusted local scientific artifacts.
"""
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms

from TVSD.splits import apply_train_val_split, load_split_indices


class ImageResponseDataset(Dataset):
    def __init__(self, stimuli, activity, image_dir, image_size=256):
        self.image_paths = list(stimuli)
        self.labels = torch.tensor(activity, dtype=torch.float32)
        self.labels[torch.isnan(self.labels)] = 0
        self.image_dir = Path(image_dir)
        self.transform = transforms.Compose([
            transforms.Resize(image_size), transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        if len(self.image_paths) != len(self.labels):
            raise ValueError('Stimulus and response counts differ')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_dir / (self.image_paths[index] + '.mp4')
        capture = cv2.VideoCapture(str(path))
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise OSError(f'Cannot read stimulus frame: {path}')
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return self.transform(image), self.labels[index]


def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def dataset_paths(config):
    data = config['dataset_params']
    root = Path(data['im_path'])
    return (root / data.get('pickle_name', 'ventral_stream_dataset.pickle'),
            Path(data.get('stimulus_path', root)) / 'ventral_stream')


def timeseries_datasets(config):
    """Return train/validation/test datasets of 30-bin, 315-channel responses."""
    data = config['dataset_params']
    pickle_path, image_dir = dataset_paths(config)
    labels = read_pickle(pickle_path)
    with h5py.File(data['timeseries_h5_path'], 'r') as handle:
        train = handle['train_timeseries'][()]
        test = handle['test_timeseries'][()]
    train = ImageResponseDataset(labels['train_stimuli'], train, image_dir, data['im_size'])
    train, val = apply_train_val_split(train, data.get('split_file'))
    test = ImageResponseDataset(labels['test_stimuli'], test, image_dir, data['im_size'])
    return train, val, test


class AverageResponseDataset(Dataset):
    def __init__(self, base, bins):
        self.base, self.bins = base, bins

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, activity = self.base[index]
        return image, activity[self.bins[0]:self.bins[1]].mean(dim=0)


def reconstruction_datasets(config, regime='real', synthetic_path=None):
    """Reference protocols: real timeseries target; synthetic pickle; concatenated mix.

    Synthetic pickles retain the reference sampler's real validation/test arrays.
    The fused real training portion uses the original real pickle, as in the source.
    """
    if regime not in ('real', 'synthetic', 'fused'):
        raise ValueError(f'Unknown regime: {regime}')
    data = config['dataset_params']
    real_path, image_dir = dataset_paths(config)
    real = read_pickle(real_path)
    size = data['im_size']
    def build(payload, split):
        return ImageResponseDataset(payload[split+'_stimuli'], payload[split+'_activity'], image_dir, size)
    if regime == 'real':
        if data.get('timeseries_h5_path'):
            datasets = timeseries_datasets(config)
            temporal = config['temporal_params']
            # Derive bins from actual sequence length, matching the reference.
            bins_total = datasets[0].dataset.labels.shape[1]
            width = temporal['total_duration_ms'] / bins_total
            bins = [int(ms/width) for ms in temporal['response_bins_ms']]
            return tuple(AverageResponseDataset(ds, bins) for ds in datasets)
        train, val = apply_train_val_split(build(real, 'train'), data.get('split_file'))
        return train, val, build(real, 'test')
    if synthetic_path is None:
        raise ValueError('synthetic_path is required for synthetic/fused training')
    synthetic = read_pickle(synthetic_path)
    train, val, test = (build(synthetic, split) for split in ('train', 'val', 'test'))
    if regime == 'fused':
        real_train, _ = apply_train_val_split(build(real, 'train'), data.get('split_file'))
        train = ConcatDataset([train, real_train])
    return train, val, test
