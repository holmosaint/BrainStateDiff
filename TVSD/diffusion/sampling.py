from __future__ import annotations
import random
import numpy as np
import torch
from diffusers import DDIMScheduler
from TVSD.subjects import MAX_NEURONS, NUM_SUBJECTS, TOTAL_NEURONS, SUBJECT_NEURON_MAP, get_subject_neuron_counts
from TVSD.diffusion.model import NeuralDiTTS
from TVSD.diffusion.core import normalize_component, denormalize_component


@torch.no_grad()
def sample_batch(
    model: NeuralDiTTS,
    scheduler: DDIMScheduler,
    images: torch.Tensor,
    baseline: torch.Tensor,
    subject_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    neural_stats: dict,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    device: torch.device = torch.device("cpu"),
):
    """DDIM sampling for one batch. Returns dict: sid -> (B_s, n_neurons)."""
    model.eval()
    B_all = images.shape[0]

    images = images.to(device)
    baseline = baseline.to(device)
    subject_ids = subject_ids.to(device)
    pad_mask = pad_mask.to(device)

    baseline = normalize_component(baseline, subject_ids, neural_stats, "baseline")

    scheduler.set_timesteps(num_inference_steps, device=device)

    x = torch.zeros(B_all, MAX_NEURONS, device=device)
    neuron_counts = get_subject_neuron_counts()
    for sid in range(NUM_SUBJECTS):
        mask = subject_ids == sid
        if mask.any():
            n = neuron_counts[sid]
            x[mask, :n] = torch.randn(mask.sum(), n, device=device)

    for t_val in scheduler.timesteps:
        ts = torch.full((B_all,), int(t_val), device=device, dtype=torch.long)

        if guidance_scale > 1.0:
            pred_dict = model.forward_cfg(
                x, ts, images, baseline, subject_ids, pad_mask,
                guidance_scale=guidance_scale,
            )
        else:
            pred_dict = model(x, ts, images, baseline, subject_ids, pad_mask)

        noise_pred = torch.zeros_like(x)
        for sid, pred in pred_dict.items():
            mask = subject_ids == sid
            n = neuron_counts[sid]
            noise_pred[mask, :n] = pred

        step_out = scheduler.step(noise_pred, t_val, x)
        x = step_out.prev_sample

    x = denormalize_component(x, subject_ids, neural_stats, "target")
    results = {}
    for sid in range(NUM_SUBJECTS):
        mask = subject_ids == sid
        if mask.any():
            n = neuron_counts[sid]
            results[sid] = x[mask, :n]
    return results


class CrossConditionDataset(torch.utils.data.Dataset):
    """
    Wraps a base timeseries dataset to return the image from one shuffled
    index and the timeseries (baseline) from another shuffled index.

    __getitem__ returns:
        image:        from image_indices[idx]
        timeseries:   from baseline_indices[idx]
    """

    def __init__(self, base_dataset, image_indices, baseline_indices):
        self.base_dataset = base_dataset
        self.image_indices = image_indices
        self.baseline_indices = baseline_indices

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_cross, _ = self.base_dataset[self.image_indices[idx]]
        _, ts_cross = self.base_dataset[self.baseline_indices[idx]]
        return img_cross, ts_cross


def make_cross_indices(n: int, seed: int) -> list:
    """
    Generate a permutation of [0, n) guaranteed to have no fixed points
    (derangement), so every sample gets a *different* image.
    """
    rng = random.Random(seed)
    indices = list(range(n))
    for _ in range(1000):
        rng.shuffle(indices)
        if all(indices[i] != i for i in range(n)):
            return indices
    # Fallback: simple cyclic shift
    return [(i + 1) % n for i in range(n)]

