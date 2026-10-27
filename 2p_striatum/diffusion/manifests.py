"""Matched-state and cross-state sample manifests."""
from pathlib import Path
from typing import Dict, List, Any
import logging
import numpy as np
import torch
logger = logging.getLogger(__name__)

def build_manifest_standard(
    sample_loader, session_idx: int, n_samples_per_trial: int,
) -> List[Dict[str, Any]]:
    """Build manifest for standard sampling. Deterministic from seed."""
    manifest = []
    history_idx = 0
    for batch in sample_loader:
        B = batch["z_hist"].shape[0]
        for b in range(B):
            sid = int(batch["session_id"][b].item())
            if sid != session_idx:
                continue
            tt = int(batch["trial_type"][b].item())
            for rep in range(n_samples_per_trial):
                manifest.append({
                    "history_idx": history_idx,
                    "batch_global_idx": b,  # not used for lookup, just metadata
                    "trial_type": tt,
                    "cursor_raw": None,  # use original cursor
                    "sample_rep": rep,
                })
            history_idx += 1
    return manifest


def build_manifest_cross(
    sample_loader, session_idx: int, cross_n: int,
    n_samples_per_trial: int, cursor_dist: Dict[int, np.ndarray], seed: int,
) -> List[Dict[str, Any]]:
    """Build manifest for cross sampling. Cursor values pre-sampled from seed."""
    rng = np.random.RandomState(seed)
    manifest = []
    history_idx = 0
    for batch in sample_loader:
        B = batch["z_hist"].shape[0]
        for b in range(B):
            sid = int(batch["session_id"][b].item())
            if sid != session_idx:
                continue
            for trial_type in [0, 1]:
                for cn in range(cross_n):
                    c_raw = float(rng.choice(cursor_dist[trial_type]))
                    for rep in range(n_samples_per_trial):
                        manifest.append({
                            "history_idx": history_idx,
                            "trial_type": trial_type,
                            "cursor_raw": c_raw,
                            "cross_idx": cn,
                            "sample_rep": rep,
                        })
            history_idx += 1
    return manifest


def collect_session_histories(sample_loader, session_idx: int, device: torch.device):
    """Collect all histories for a session, indexed sequentially."""
    histories = []
    for batch in sample_loader:
        B = batch["z_hist"].shape[0]
        for b in range(B):
            sid = int(batch["session_id"][b].item())
            if sid != session_idx:
                continue
            histories.append({
                "z_hist": batch["z_hist"][b:b+1].to(device),
                "v_hist": batch["v_hist"][b:b+1].to(device),
                "c_hist": batch["c_hist"][b:b+1].to(device),
                "trial_type": int(batch["trial_type"][b].item()),
                "subject_id": int(batch["subject_id"][b].item()),
                "session_id": sid,
            })
    return histories


def precompute_cursor_distribution(
    latent_dir: Path,
    n_sessions: int,
    anchor_frame: int = 45,
) -> Dict[int, np.ndarray]:
    """Precompute empirical distribution of cursor at anchor_frame per trial type.

    Scans all training trials across all sessions.

    Returns:
        {trial_type: np.ndarray of raw cursor values at anchor_frame}
    """
    cursor_by_type: Dict[int, List[float]] = {0: [], 1: []}

    for si in range(n_sessions):
        train_dir = latent_dir / f"session_{si}" / "train"
        if not train_dir.exists():
            continue

        for trial_path in sorted(train_dir.glob("trial_*.pt")):
            trial = torch.load(trial_path, map_location="cpu", weights_only=False)
            T = trial["length"]
            if T <= anchor_frame:
                continue

            tt = trial["trial_type"]
            c_val = trial["c_seq"][anchor_frame, 0].item()
            if tt in cursor_by_type:
                cursor_by_type[tt].append(c_val)

    result = {}
    for tt, values in cursor_by_type.items():
        arr = np.array(values, dtype=np.float32)
        logger.info(
            f"Cursor dist type={tt}: n={len(arr)}, "
            f"mean={arr.mean():.3f}+-{arr.std():.3f}, "
            f"range=[{arr.min():.3f}, {arr.max():.3f}]"
        )
        result[tt] = arr

    return result

