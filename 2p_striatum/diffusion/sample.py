"""Generate matched-state or cross-state trials and real decoder reference splits."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .chunk_dataset import ChunkDataset, ChunkNormStats, build_session_to_subject_map
from .cursor_rollout import load_cursor_model, build_cursor_coef_tensors
from .manifests import (build_manifest_standard, build_manifest_cross,
                        collect_session_histories, precompute_cursor_distribution)
from .sampling import autoregressive_rollout
from .train import read_sessions, load_model
from .vae import (load_frozen_stitched_vae, SessionBoundVAE,
                  encode_frames_independent, decode_frames_independent)


def build_session_manifest(loader, session_index, mode, samples_per_trial,
                           cross_n, cursor_dist, seed):
    if mode == "matched":
        return build_manifest_standard(loader, session_index, samples_per_trial)
    matched = build_manifest_standard(loader, session_index, 1)
    crossed = build_manifest_cross(loader, session_index, cross_n,
        samples_per_trial, cursor_dist, seed * 1000 + session_index)
    return matched + crossed


@torch.no_grad()
def generate_from_manifest(manifest, histories, model, scheduler, ae, device,
                           norm_stats, cursor_coefs, cursor_intercepts,
                           train_dir, ddim_steps=50, max_rollout_chunks=18, max_retries=5):
    """Generate every manifest entry; discard only repeated wrong-direction failures."""
    train_dir = Path(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    stop_counts = {}
    for index, entry in enumerate(manifest):
        h = histories[entry["history_idx"]]
        c_hist = h["c_hist"].clone()
        if entry["cursor_raw"] is not None:
            c_hist[0, -1, 0] = (entry["cursor_raw"] - norm_stats.c_mean.item()) / norm_stats.c_std.item()
        result = None
        for _ in range(max_retries):
            r = autoregressive_rollout(model, scheduler,
                h["z_hist"], h["v_hist"], c_hist, entry["trial_type"],
                h["session_id"], h["subject_id"], 0, norm_stats,
                cursor_coefs, cursor_intercepts, max_rollout_chunks, ddim_steps)
            if r.stop_reason != "wrong_direction":
                result = r
                break
        if result is None:
            trial = dict(discarded=True, trial_type=entry["trial_type"],
                         session_id=h["session_id"], length=0)
            reason = "discarded"
        else:
            x = decode_frames_independent(ae, result.z_generated_raw.unsqueeze(0).to(device))[0].cpu()
            trial = dict(x_recon=x, v_raw=result.v_generated_raw.clone(),
                         c_raw=result.c_generated_raw.clone(), trial_type=result.trial_type,
                         session_id=result.session_id, length=len(x), stop_reason=result.stop_reason)
            reason = result.stop_reason
        stop_counts[reason] = stop_counts.get(reason, 0) + 1
        torch.save(trial, train_dir / f"trial_{index:04d}.pt")
    return {"n_total": len(manifest), "stop_counts": stop_counts}


@torch.no_grad()
def save_real_splits(latent_dir, output, session_idx, ae, device):
    """Roundtrip real neural frames 46:T through the same frozen VAE."""
    for source, target in (("train", "real_train"), ("valid", "valid"), ("test", "test")):
        dst = Path(output) / f"session_{session_idx}" / target
        dst.mkdir(parents=True, exist_ok=True)
        for path in sorted((Path(latent_dir) / f"session_{session_idx}" / source).glob("trial_*.pt")):
            trial = torch.load(path, map_location="cpu", weights_only=False)
            T = trial["length"]
            if T <= 46:
                continue
            neural = trial["neural_seq"][46:T].unsqueeze(0).to(device)
            z = encode_frames_independent(ae, neural, sample_posterior=False)
            x = decode_frames_independent(ae, z)[0].cpu()
            torch.save(dict(x_recon=x, v_raw=trial["v_seq"][46:T].clone(),
                c_raw=trial["c_seq"][46:T].clone(), trial_type=trial["trial_type"],
                session_id=trial["session_id"], length=len(x), stop_reason="real"), dst / path.name)


def sample(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    specs = read_sessions(args.sessions)
    names = [s["name"] for s in specs]
    mapping, _ = build_session_to_subject_map(names)
    model, scheduler, _ = load_model(args.checkpoint, names, args.device, args.ema)
    vae, latent_dim, _ = load_frozen_stitched_vae(args.vae_checkpoint, args.device)
    if latent_dim != model.latent_dim:
        raise ValueError("Diffusion and VAE latent dimensions differ")
    stats = ChunkNormStats.load(Path(args.norm_stats))
    coefs, intercepts = build_cursor_coef_tensors(names, load_cursor_model(Path(args.cursor_model_csv)))
    ds = ChunkDataset(Path(args.latent_dir), "train", list(range(len(names))), mapping,
                      mode="fixed", history_len=model.history_len, chunk_len=model.chunk_len,
                      norm_stats=stats)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    cursor_dist = (precompute_cursor_distribution(Path(args.latent_dir), len(names))
                   if args.mode == "cross" else None)
    summary = {}
    for index, spec in enumerate(specs):
        histories = collect_session_histories(loader, index, torch.device(args.device))
        manifest = build_session_manifest(loader, index, args.mode,
            args.samples_per_trial, args.cross_n, cursor_dist, args.seed)
        directory = Path(args.output) / f"session_{index}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
        ae = SessionBoundVAE(vae, spec["session_id"])
        summary[spec["name"]] = generate_from_manifest(manifest, histories, model,
            scheduler, ae, args.device, stats, coefs, intercepts, directory / "train",
            args.ddim_steps, args.max_rollout_chunks, args.max_retries)
        save_real_splits(args.latent_dir, args.output, index, ae, args.device)
    (Path(args.output) / "sampling.json").write_text(json.dumps(dict(args=vars(args), results=summary), indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("latent-dir", "sessions", "checkpoint", "norm-stats", "vae-checkpoint", "cursor-model-csv", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--mode", choices=["matched", "cross"], default="matched")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--device", default="cpu")
    for name, default in dict(samples_per_trial=1, cross_n=2, ddim_steps=50,
                             max_rollout_chunks=18, max_retries=5, seed=42).items():
        p.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    a = p.parse_args()
    if min(a.samples_per_trial, a.cross_n, a.ddim_steps, a.max_rollout_chunks, a.max_retries) < 1:
        p.error("sample counts, step counts, and retry count must be positive")
    print(json.dumps(sample(a), indent=2))


if __name__ == "__main__":
    main()
