from __future__ import annotations
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler
from TVSD.subjects import MAX_NEURONS, NUM_SUBJECTS, SUBJECT_NEURON_MAP, get_subject_neuron_counts
from TVSD.diffusion.model import NeuralDiTTS

def _get_component_stats(stats, sid, component):
    """The released model uses a scalar mean/std pooled over both components."""
    return stats[sid]["mean"], stats[sid]["std"], get_subject_neuron_counts()[sid]


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for sp, mp in zip(self.shadow.parameters(), model.parameters()):
            sp.data.mul_(self.decay).add_(mp.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


def compute_neural_stats(base_dataset, baseline_bins, target_bins, device="cpu"):
    """
    Compute subject-level mean/std by pooling across:
      - all neurons of a subject
      - both baseline and target components
      - all samples in the training split

    Returns dict keyed by subject_id:
        {sid: {"mean": scalar, "std": scalar}}
    """
    slices = {
        info["subject_id"]: slice(info["start"], info["end"])
        for info in SUBJECT_NEURON_MAP.values()
    }

    all_baselines = []
    all_targets = []
    for i in range(len(base_dataset)):
        _, ts = base_dataset[i]  # ts: (T, N)
        bl = ts[baseline_bins[0]:baseline_bins[1]].mean(dim=0)
        tg = ts[target_bins[0]:target_bins[1]].mean(dim=0)
        all_baselines.append(bl)
        all_targets.append(tg)

    all_baselines = torch.stack(all_baselines)  # (N_samples, 315)
    all_targets = torch.stack(all_targets)

    stats = {}
    for sid, sl in slices.items():
        bl_data = all_baselines[:, sl].float()
        tg_data = all_targets[:, sl].float()
        pooled = torch.cat([bl_data.reshape(-1), tg_data.reshape(-1)], dim=0)
        stats[sid] = {
            "mean": pooled.mean(),
            "std": pooled.std().clamp(min=1e-6),
        }
    return stats


def normalize_component(
    activity: torch.Tensor,
    subject_ids: torch.Tensor,
    stats: dict,
    component: str,
) -> torch.Tensor:
    out = activity.clone()
    for sid in range(NUM_SUBJECTS):
        mask = subject_ids == sid
        if mask.any():
            mean, std, n = _get_component_stats(stats, sid, component)
            mean = mean.to(activity.device)
            std = std.to(activity.device)
            out[mask, :n] = (out[mask, :n] - mean) / std
    return out


def denormalize_component(
    activity: torch.Tensor,
    subject_ids: torch.Tensor,
    stats: dict,
    component: str,
) -> torch.Tensor:
    out = activity.clone()
    for sid in range(NUM_SUBJECTS):
        mask = subject_ids == sid
        if mask.any():
            mean, std, n = _get_component_stats(stats, sid, component)
            mean = mean.to(activity.device)
            std = std.to(activity.device)
            out[mask, :n] = out[mask, :n] * std + mean
    return out


def train_one_step(
    model: NeuralDiTTS,
    scheduler: DDIMScheduler,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    subject_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    neural_stats: dict,
    device: torch.device,
):
    model.train()
    B_all = target.shape[0]

    images = images.to(device)
    baseline = baseline.to(device)
    target = target.to(device)
    subject_ids = subject_ids.to(device)
    pad_mask = pad_mask.to(device)

    # Normalize both components
    baseline = normalize_component(baseline, subject_ids, neural_stats, "baseline")
    target = normalize_component(target, subject_ids, neural_stats, "target")

    # Sample timestep
    t = torch.randint(0, scheduler.config.num_train_timesteps, (B_all,), device=device).long()

    # Add noise to target (only real neuron positions)
    noise = torch.zeros_like(target)
    neuron_counts = get_subject_neuron_counts()
    for sid in range(NUM_SUBJECTS):
        mask = subject_ids == sid
        if mask.any():
            n = neuron_counts[sid]
            noise[mask, :n] = torch.randn(mask.sum(), n, device=device)

    noisy_target = scheduler.add_noise(target, noise, t)

    # Forward: baseline is a condition, noisy_target is the diffusion input
    pred_dict = model(noisy_target, t, images, baseline, subject_ids, pad_mask)

    # Loss
    prediction_type = scheduler.config.prediction_type
    total_loss = 0.0
    for sid, pred in pred_dict.items():
        n = neuron_counts[sid]
        mask = subject_ids == sid
        if prediction_type == "epsilon":
            loss_target = noise[mask, :n]
        elif prediction_type == "sample":
            loss_target = target[mask, :n]
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type}")
        total_loss = total_loss + F.mse_loss(pred, loss_target)
    total_loss = total_loss / len(pred_dict)

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return total_loss.item()


@torch.no_grad()
def evaluate(
    model: NeuralDiTTS,
    scheduler: DDIMScheduler,
    dataloader,
    neural_stats: dict,
    device: torch.device,
):
    model.eval()
    neuron_counts = get_subject_neuron_counts()
    prediction_type = scheduler.config.prediction_type

    total_loss = 0.0
    per_subject_loss = {sid: 0.0 for sid in range(NUM_SUBJECTS)}
    per_subject_count = {sid: 0 for sid in range(NUM_SUBJECTS)}
    num_batches = 0

    for images, baseline, target, subject_ids, pad_mask in dataloader:
        B_all = target.shape[0]
        images = images.to(device)
        baseline = baseline.to(device)
        target = target.to(device)
        subject_ids = subject_ids.to(device)
        pad_mask = pad_mask.to(device)

        baseline = normalize_component(baseline, subject_ids, neural_stats, "baseline")
        target = normalize_component(target, subject_ids, neural_stats, "target")

        t = torch.randint(0, scheduler.config.num_train_timesteps, (B_all,), device=device).long()

        noise = torch.zeros_like(target)
        for sid in range(NUM_SUBJECTS):
            mask = subject_ids == sid
            if mask.any():
                n = neuron_counts[sid]
                noise[mask, :n] = torch.randn(mask.sum(), n, device=device)

        noisy_target = scheduler.add_noise(target, noise, t)
        pred_dict = model(noisy_target, t, images, baseline, subject_ids, pad_mask)

        batch_loss = 0.0
        for sid, pred in pred_dict.items():
            n = neuron_counts[sid]
            mask = subject_ids == sid
            if prediction_type == "epsilon":
                loss_target = noise[mask, :n]
            elif prediction_type == "sample":
                loss_target = target[mask, :n]
            else:
                raise ValueError(f"Unsupported prediction_type: {prediction_type}")
            loss_s = F.mse_loss(pred, loss_target).item()
            per_subject_loss[sid] += loss_s
            per_subject_count[sid] += 1
            batch_loss += loss_s
        batch_loss /= max(len(pred_dict), 1)

        total_loss += batch_loss
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_per_subject = {
        sid: per_subject_loss[sid] / max(per_subject_count[sid], 1)
        for sid in range(NUM_SUBJECTS)
    }
    return avg_loss, avg_per_subject

