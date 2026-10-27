from typing import Dict
import numpy as np
from sklearn.metrics import r2_score

def compute_pred_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """
    targets, preds: (N,1) or (N,)
    """
    if targets.ndim == 1:
        targets = targets[:, None]
    if preds.ndim == 1:
        preds = preds[:, None]

    r2 = float(r2_score(targets, preds, multioutput="uniform_average"))
    mse = float(np.mean((targets - preds) ** 2))
    mae = float(np.mean(np.abs(targets - preds)))

    # Corr
    t0 = targets - targets.mean(axis=0, keepdims=True)
    p0 = preds - preds.mean(axis=0, keepdims=True)
    num = np.sum(t0 * p0, axis=0)
    den = np.sqrt(np.sum(t0 ** 2, axis=0) * np.sum(p0 ** 2, axis=0))
    corr = num / (den + 1e-12)
    corr = float(np.nanmean(corr))

    result = {"r2": r2, "mse": mse, "mae": mae, "corr": corr}
    K = targets.shape[1]
    if K > 1:
        result["r2_per_step"] = [float(r2_score(targets[:, k], preds[:, k])) for k in range(K)]
    return result

