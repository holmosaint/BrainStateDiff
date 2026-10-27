"""Train autoregressive neural/velocity chunk diffusion from extracted trials."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from diffusers import DDIMScheduler
from .chunk_denoiser import ChunkDenoiser
from .chunk_dataset import build_chunk_dataloaders, build_session_to_subject_map
from .cursor_rollout import load_cursor_model, build_cursor_coef_tensors
from .training import EMA, train_one_epoch, evaluate_loss

MODEL_KEYS = ("latent_dim", "chunk_len", "history_len", "model_dim", "cond_dim",
              "denoise_depth", "n_heads", "mlp_ratio", "dropout")


def read_sessions(path):
    """Session list order defines the embedding index; preserve it across stages."""
    return json.loads(Path(path).read_text())["sessions"]


def make_model(args, session_names, device):
    mapping, subjects = build_session_to_subject_map(session_names)
    return ChunkDenoiser(**{k: args[k] for k in MODEL_KEYS},
                         num_sessions=len(session_names), num_subjects=len(subjects),
                         session_to_subject=mapping).to(device)


def make_scheduler(args):
    return DDIMScheduler(num_train_timesteps=args["diffusion_steps"],
                         beta_schedule=args["beta_schedule"],
                         prediction_type=args["prediction_type"], clip_sample=False)


def load_model(checkpoint, session_names, device="cpu", use_ema=False):
    """Load this package's or the original chunk-diffusion checkpoint format."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "session_names" in ckpt and ckpt["session_names"] != list(session_names):
        raise ValueError("Session names and order must match the diffusion checkpoint")
    model = make_model(ckpt["args"], session_names, device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if use_ema:
        ema = EMA(model)
        ema.load_state_dict(ckpt["ema_state_dict"])
        ema.apply(model)
    model.eval()
    return model, make_scheduler(ckpt["args"]), ckpt


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    names = [s["name"] for s in read_sessions(args.sessions)]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train_loader, _, valid_loader, _, stats, _, _ = build_chunk_dataloaders(
        Path(args.latent_dir), names, history_len=args.history_len,
        chunk_len=args.chunk_len, stride=args.stride, batch_size=args.batch_size)
    if not len(train_loader.dataset) or not len(valid_loader.dataset):
        raise ValueError("Nonempty training and validation splits are required")
    stats.save(out / "chunk_norm_stats.pt")
    coefs, intercepts = build_cursor_coef_tensors(names, load_cursor_model(Path(args.cursor_model_csv)))
    model = make_model(vars(args), names, device)
    scheduler = make_scheduler(vars(args))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6)
        if args.lr_schedule == "cosine" else None)
    ema = EMA(model, args.ema_decay) if args.ema else None
    best_loss, history = float("inf"), []

    def save(name, epoch):
        ckpt = dict(epoch=epoch, model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(), args=vars(args),
                    session_names=names, rng_states=dict(python=random.getstate(),
                    numpy=np.random.get_state(), torch_cpu=torch.get_rng_state()))
        if ema is not None:
            ckpt["ema_state_dict"] = ema.state_dict()
        if lr_scheduler is not None:
            ckpt["lr_scheduler_state_dict"] = lr_scheduler.state_dict()
        torch.save(ckpt, out / name)

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * (epoch + 1) / args.warmup_epochs
        metrics = train_one_epoch(model, scheduler, train_loader, optimizer, device,
            stats, coefs, intercepts, epoch, args.tf_start, args.tf_decay, args.tf_min,
            args.lambda_z, args.lambda_v, args.rollout_ddim_steps, args.grad_clip, ema)
        if lr_scheduler is not None and epoch >= args.warmup_epochs:
            lr_scheduler.step()
        record = {"epoch": epoch, "train": metrics}
        if epoch % args.eval_every == 0:
            val = evaluate_loss(model, scheduler, valid_loader, device, stats,
                                coefs, intercepts, args.lambda_z, args.lambda_v,
                                args.rollout_ddim_steps)
            record["valid"] = val
            if val["loss"] < best_loss:
                best_loss = val["loss"]
                save("best_ckpt.pt", epoch)
        save("latest_ckpt.pt", epoch)
        history.append(record)
        (out / "metrics.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(record), flush=True)
    return model, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("latent-dir", "sessions", "cursor-model-csv", "output"):
        p.add_argument("--" + name, required=True)
    for name, default in dict(latent_dim=16, chunk_len=8, history_len=16,
            model_dim=256, cond_dim=256, denoise_depth=6, n_heads=8,
            diffusion_steps=1000, epochs=200, batch_size=32, warmup_epochs=5,
            stride=1, rollout_ddim_steps=10, eval_every=5, seed=42).items():
        p.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    for name, default in dict(mlp_ratio=4., dropout=0., lr=1e-4,
            weight_decay=1e-4, grad_clip=1., lambda_z=1., lambda_v=1.,
            tf_start=1., tf_decay=.95, tf_min=.2, ema_decay=.9999).items():
        p.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    p.add_argument("--prediction-type", choices=["epsilon", "sample", "v_prediction"], default="epsilon")
    p.add_argument("--beta-schedule", default="squaredcos_cap_v2")
    p.add_argument("--lr-schedule", choices=["cosine", "constant"], default="cosine")
    p.add_argument("--ema", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.epochs <= args.warmup_epochs and args.lr_schedule == "cosine":
        p.error("cosine training requires epochs > warmup-epochs")
    train(args)


if __name__ == "__main__":
    main()
