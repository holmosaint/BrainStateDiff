"""
decoders/mlp_decoder.py
-----------------------

MLPDecoder model and training/fine-tuning/evaluation functions, plus
MLPDecoderWrapper which wraps them behind the :class:`AbstractDecoder` interface.
"""

from __future__ import annotations

import argparse
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm


from .utils import _seed_all

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLPDecoder model
# ---------------------------------------------------------------------------

class MLPDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float = 0.0, output_dim: int = 1):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(0, depth)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Training / fine-tuning / evaluation
# ---------------------------------------------------------------------------

def train_mlp_decoder(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    epochs: Optional[int] = None,
) -> Tuple[MLPDecoder, List[float], List[float], int]:
    """
    Train MLP decoder; re-seeds to ensure deterministic initialization per call.
    Returns model and (train_loss_history, val_loss_history, best_epoch).
    """
    _seed_all(seed)
    epochs = args.mlp_epochs if epochs is None else epochs

    K = y_train.shape[1] if y_train.ndim == 2 else 1
    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).float().reshape(-1, K)
    Xva = torch.from_numpy(X_val).float()
    yva = torch.from_numpy(y_val).float().reshape(-1, K)

    train_ds = torch.utils.data.TensorDataset(Xtr, ytr)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.mlp_batch_size,
        shuffle=True,
        generator=g,
    )

    model = MLPDecoder(
        input_dim=X_train.shape[1],
        hidden_dim=args.mlp_hidden_dim,
        depth=args.mlp_depth,
        dropout=args.mlp_dropout,
        output_dim=K,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    train_hist: List[float] = []
    val_hist: List[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n = 0
        for xb, yb in tqdm(train_loader, desc=f"MLP train ep{ep:03d}", leave=False, disable=getattr(args, "disable_tqdm", False)):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            bsz = xb.size(0)
            ep_loss += float(loss.item()) * bsz
            n += bsz
        ep_loss = ep_loss / max(n, 1)

        model.eval()
        with torch.no_grad():
            pred_val = model(Xva.to(device))
            val_loss = float(loss_fn(pred_val, yva.to(device)).item())

        train_hist.append(ep_loss)
        val_hist.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_hist, val_hist, best_epoch


def finetune_mlp_decoder(
    model: MLPDecoder,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
) -> Tuple[MLPDecoder, List[float], List[float], int]:
    """Continue training an existing MLP decoder without re-initializing weights."""
    loss_fn = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)

    K = y_train.shape[1] if y_train.ndim == 2 else 1
    Xtr = torch.from_numpy(X_train).float()
    ytr = torch.from_numpy(y_train).float().reshape(-1, K)
    Xva = torch.from_numpy(X_val).float()
    yva = torch.from_numpy(y_val).float().reshape(-1, K)
    train_ds = torch.utils.data.TensorDataset(Xtr, ytr)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.mlp_batch_size,
        shuffle=True,
    )

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    train_hist: List[float] = []
    val_hist: List[float] = []

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n = 0
        for xb, yb in tqdm(train_loader, desc=f"MLP finetune ep{ep:03d}", leave=False, disable=getattr(args, "disable_tqdm", False)):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            bsz = xb.size(0)
            ep_loss += float(loss.item()) * bsz
            n += bsz
        ep_loss = ep_loss / max(n, 1)

        model.eval()
        with torch.no_grad():
            pred_val = model(Xva.to(device))
            val_loss = float(loss_fn(pred_val, yva.to(device)).item())

        train_hist.append(ep_loss)
        val_hist.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = ep

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_hist, val_hist, best_epoch


def evaluate_mlp_predictions(model: MLPDecoder, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X).float().to(device)
        preds = model(x).cpu().numpy().astype(np.float32)
    return preds
