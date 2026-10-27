"""Extract framewise latent means from canonical, already-split 2p trial pickles."""
import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from .vae import load_frozen_stitched_vae, SessionBoundVAE, encode_frames_independent
from .cursor_rollout import load_cursor_model
from .train import read_sessions

TRIAL_TYPES = {"Cue-Left": 0, "Cue-Right": 1}


def recompute_cursor_for_trial(trial, coef, intercept):
    """Preserve frame 45's measured cursor; integrate velocity after that anchor."""
    T = trial["length"]
    v = trial["v_seq"].numpy().flatten()
    c_real = trial["c_seq"].numpy().flatten()
    c_est = np.zeros(T, dtype=np.float32)
    if T > 45:
        c_est[45] = c_real[45]
        for t in range(46, T):
            c_est[t] = c_est[t - 1] + coef * v[t] + intercept
    return torch.from_numpy(c_est).unsqueeze(-1)


@torch.no_grad()
def extract_session(payload, ae, session_idx, device="cpu", batch_size=32):
    """Yield (split, index, trial) from {split}_{neural,behav,types} arrays.

    Behavior channels are [cursor, velocity]. Canonical aligned behavior is
    required; there is no implicit switch to alternate ITI arrays.
    """
    for split in ("train", "valid", "test"):
        xs, bs, ts = [payload[f"{split}_{key}"] for key in ("neural", "behav", "types")]
        if not len(xs) == len(bs) == len(ts):
            raise ValueError(f"Mismatched {split} trial counts")
        for start in range(0, len(xs), batch_size):
            neural = [torch.as_tensor(x, dtype=torch.float32) for x in xs[start:start + batch_size]]
            padded = pad_sequence(neural, batch_first=True).to(device)
            latents = encode_frames_independent(ae, padded, sample_posterior=False).cpu()
            for b, x in enumerate(neural):
                i, length = start + b, len(x)
                behav = torch.as_tensor(bs[i], dtype=torch.float32)
                if len(behav) != length or behav.shape[1] < 2:
                    raise ValueError(f"Behavior/neural frame mismatch in {split} trial {i}")
                if not length:
                    continue
                tt = TRIAL_TYPES[ts[i]] if isinstance(ts[i], str) else int(ts[i])
                if tt not in (0, 1):
                    raise ValueError(f"Expected cue 0/1, received {tt}")
                yield split, i, dict(z_seq=latents[b, :length], neural_seq=x,
                    v_seq=behav[:, 1:2].clone(), c_seq=behav[:, :1].clone(),
                    trial_type=tt, session_id=session_idx, length=length)


def extract(sessions, checkpoint, cursor_model_csv, output, device="cpu", batch_size=32):
    vae, _, counts = load_frozen_stitched_vae(checkpoint, device)
    cursor_models = load_cursor_model(Path(cursor_model_csv))
    for index, spec in enumerate(read_sessions(sessions)):
        sid = spec["session_id"]
        if sid not in counts:
            raise KeyError(f"VAE has no session {sid}")
        with Path(spec["data_path"]).open("rb") as handle:
            payload = pickle.load(handle)
        ae = SessionBoundVAE(vae, sid)
        cursor = cursor_models[spec["name"]]
        for split, i, trial in extract_session(payload, ae, index, device, batch_size):
            trial["c_seq"] = recompute_cursor_for_trial(trial, cursor["coef"], cursor["intercept"])
            directory = Path(output) / f"session_{index}" / split
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(trial, directory / f"trial_{i:04d}.pt")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("sessions", "vae-checkpoint", "cursor-model-csv", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    a = p.parse_args()
    extract(a.sessions, a.vae_checkpoint, a.cursor_model_csv, a.output, a.device, a.batch_size)


if __name__ == "__main__":
    main()
