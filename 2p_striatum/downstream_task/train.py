"""Linear/MLP velocity decoding with real, synthetic, fused and fine-tuned training."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .models import train_mlp_decoder, finetune_mlp_decoder, evaluate_mlp_predictions
from .metrics import compute_pred_metrics
from .data import load_split


def evaluate(model, splits, device):
    return {name: compute_pred_metrics(y, evaluate_mlp_predictions(model, x, device))
            for name, (x, y) in splits.items()}


def run_regimes(real_splits, synthetic, args, device, seed):
    """Fit each regime using real validation loss; test labels never select weights.

    Fine tuning keeps the pretrained model when its validation R2 is at least as
    good as the fine-tuned model, matching the original evaluation protocol.
    """
    valid = real_splits["valid"]
    real = real_splits["train"]
    fused = tuple(np.concatenate([synthetic[i], real[i]], axis=0) for i in (0, 1))
    results, models = {}, {}

    def fit(data):
        return train_mlp_decoder(*data, *valid, args, device, seed)

    syn, tr, va, ep = fit(synthetic)
    results["synthetic"] = dict(metrics=evaluate(syn, real_splits, device), best_epoch=ep,
                                train_loss=tr, valid_loss=va)
    models["synthetic"] = syn
    ft, tr, va, ep = finetune_mlp_decoder(copy.deepcopy(syn), *real, *valid,
                                         args, device, args.mlp_finetune_epochs)
    metrics = evaluate(ft, real_splits, device)
    retained_pretrained = metrics["valid"]["r2"] <= results["synthetic"]["metrics"]["valid"]["r2"]
    if retained_pretrained:
        ft = copy.deepcopy(syn)
        metrics = results["synthetic"]["metrics"]
    models["finetune"] = ft
    results["finetune"] = dict(metrics=metrics, best_epoch=ep, train_loss=tr,
                               valid_loss=va, retained_pretrained=retained_pretrained)
    for name, data in (("fused", fused), ("real", real)):
        model, tr, va, ep = fit(data)
        models[name] = model
        results[name] = dict(metrics=evaluate(model, real_splits, device),
                             best_epoch=ep, train_loss=tr, valid_loss=va)
    return results, models


def train(args):
    device = torch.device(args.device)
    root = Path(args.data_dir) / f"session_{args.session_index}"
    splits = {key: load_split(root / directory, args.horizon)
              for key, directory in (("train", "real_train"), ("valid", "valid"), ("test", "test"))}
    synthetic = load_split(root / "train", args.horizon)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for seed in range(args.seed, args.seed + args.num_seeds):
        results, models = run_regimes(splits, synthetic, args, device, seed)
        all_results[seed] = results
        for regime, model in models.items():
            torch.save(dict(model_state_dict=model.state_dict(), input_dim=synthetic[0].shape[1],
                output_dim=args.horizon, hidden_dim=args.mlp_hidden_dim, depth=args.mlp_depth,
                dropout=args.mlp_dropout, args=vars(args), seed=seed), out / f"{regime}_seed{seed}.pt")
    (out / "metrics.json").write_text(json.dumps(all_results, indent=2))
    return all_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--decoder", choices=["linear", "mlp"], default="linear")
    for name, default in dict(session_index=0, horizon=8, seed=12345, num_seeds=5,
            mlp_epochs=50, mlp_finetune_epochs=50, mlp_batch_size=512, mlp_hidden_dim=256).items():
        p.add_argument("--" + name.replace("_", "-"), type=int, default=default)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    p.add_argument("--mlp-weight-decay", type=float, default=0.)
    p.add_argument("--mlp-dropout", type=float, default=.1)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    a.mlp_depth = 0 if a.decoder == "linear" else 2
    a.disable_tqdm = True
    if min(a.horizon, a.num_seeds, a.mlp_epochs, a.mlp_finetune_epochs) < 1:
        p.error("horizon, seed count and epochs must be positive")
    train(a)


if __name__ == "__main__":
    main()
