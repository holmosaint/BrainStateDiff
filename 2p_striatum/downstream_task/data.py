"""Load generated/roundtripped trials and form within-trial forecast targets."""
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch

def apply_decode_horizon(
    X_trials: List[np.ndarray],
    y_trials: List[np.ndarray],
    K: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert per-trial targets from (Li, 1) to (Li-K+1, K) rolling windows.

    Also trims X_trials to match.  Trials with Li < K are dropped.
    When K <= 1 the inputs are returned unchanged (backward compatible).
    """
    if K <= 1:
        return X_trials, y_trials

    X_out: List[np.ndarray] = []
    y_out: List[np.ndarray] = []
    n_dropped = 0
    for X_t, y_t in zip(X_trials, y_trials):
        Li = X_t.shape[0]
        if Li < K:
            n_dropped += 1
            continue
        y_flat = y_t.ravel()  # (Li,)
        y_rolled = np.lib.stride_tricks.sliding_window_view(y_flat, K)  # (Li-K+1, K)
        X_out.append(X_t[: Li - K + 1].astype(np.float32))
        y_out.append(y_rolled.copy().astype(np.float32))
    if n_dropped > 0:
        import logging
        logging.getLogger(__name__).warning(
            f"apply_decode_horizon: dropped {n_dropped}/{len(X_trials)} trials "
            f"shorter than K={K}"
        )
    return X_out, y_out


def load_split(split_dir, horizon=1):
    xs, ys = [], []
    for path in sorted(Path(split_dir).glob("trial_*.pt")):
        trial = torch.load(path, map_location="cpu", weights_only=False)
        if trial.get("discarded", False):
            continue
        if trial["length"]:
            xs.append(trial["x_recon"].numpy().astype(np.float32))
            ys.append(trial["v_raw"].numpy().astype(np.float32))
    xs, ys = apply_decode_horizon(xs, ys, horizon)
    if not xs:
        raise ValueError(f"No usable trials at {split_dir} for horizon {horizon}")
    return np.concatenate(xs), np.concatenate(ys)
