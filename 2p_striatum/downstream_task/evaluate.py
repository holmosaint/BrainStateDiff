"""Evaluate a saved decoder on an exported real or synthetic split."""
import argparse
import json
from pathlib import Path
import torch
from .models import MLPDecoder, evaluate_mlp_predictions
from .metrics import compute_pred_metrics
from .data import load_split


def evaluate_checkpoint(checkpoint, split_dir, device="cpu"):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = MLPDecoder(ckpt["input_dim"], ckpt["hidden_dim"], ckpt["depth"],
                       ckpt["dropout"], ckpt["output_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    x, y = load_split(split_dir, ckpt["output_dim"])
    return compute_pred_metrics(y, evaluate_mlp_predictions(model, x, torch.device(device)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    result = evaluate_checkpoint(a.checkpoint, a.split_dir, a.device)
    Path(a.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
