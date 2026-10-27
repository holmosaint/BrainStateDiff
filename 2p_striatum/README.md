# Two-photon striatum

Train neural-latent/velocity chunk diffusion, generate matched-state or cross-state
trials, and fit linear or MLP velocity decoders. Run commands from the `BrainStateDiff`
repository root. Python supports `python -m 2p_striatum...`; programmatic imports can
use `importlib.import_module("2p_striatum.diffusion.train")`.

## Prepare the inputs

Supply a pretrained stitched VAE checkpoint, already split session pickles, and a
CSV containing the cursor integration coefficients for each session. VAE fitting
is an upstream prerequisite. The checkpoint requires `model_state_dict`,
`session_neuron_counts`, and `config` as a dictionary or supported serialized
`AutoencoderKLConfig`.

Each session pickle contains `train_neural`, `train_behav`, and `train_types`, plus
the corresponding `valid_` and `test_` entries. Each list element is one trial:

- Neural activity: `(frames, session_neurons)`.
- Behavior: `(frames,2)` with channels **cursor, velocity**.
- Trial type: `"Cue-Left"`/`"Cue-Right"` or integer `0`/`1`.

Create `sessions.json`, keeping its order fixed throughout the pipeline:

```json
{"sessions": [
  {"name": "sw24_20231121", "session_id": "sw24_20231121",
   "data_path": "/data/sw24_20231121_dataloaders.pkl"}
]}
```

`name` identifies the session in the cursor CSV; its prefix before `_` identifies
the animal. `session_id` identifies the VAE input/output head. The CSV must contain
`session,r2,coef,intercept` columns and one row for every configured session.

## Extract latents and train diffusion

```bash
python -m 2p_striatum.diffusion.extract_latents \
  --sessions sessions.json --vae-checkpoint /models/stitched_vae.pt \
  --cursor-model-csv cursor_models.csv --output runs/striatum/latents --device cuda

python -m 2p_striatum.diffusion.train \
  --latent-dir runs/striatum/latents --sessions sessions.json \
  --cursor-model-csv cursor_models.csv --output runs/striatum/diffusion \
  --ema --device cuda
```

Latent extraction uses VAE posterior means for individual frames. It preserves
cursor position at frame 45 and integrates subsequent velocity using the session's
cursor coefficients. Trials are written to `session_i/{train,valid,test}/trial_XXXX.pt`.

Diffusion training uses 16 history frames and eight future frames per chunk,
starting at frame 46. Normalization is fitted on training data. The checkpoint
with the lowest validation loss is saved as `best_ckpt.pt`; `latest_ckpt.pt`,
`chunk_norm_stats.pt`, and `metrics.json` are also written. Use `--help` to change
model dimensions, training epochs, batch size, learning rate, and sampling options.

## Generate synthetic trials

```bash
python -m 2p_striatum.diffusion.sample \
  --latent-dir runs/striatum/latents --sessions sessions.json \
  --checkpoint runs/striatum/diffusion/best_ckpt.pt \
  --norm-stats runs/striatum/diffusion/chunk_norm_stats.pt \
  --vae-checkpoint /models/stitched_vae.pt --cursor-model-csv cursor_models.csv \
  --mode matched --samples-per-trial 1 --ema \
  --output runs/striatum/matched --device cuda

python -m 2p_striatum.diffusion.sample \
  --latent-dir runs/striatum/latents --sessions sessions.json \
  --checkpoint runs/striatum/diffusion/best_ckpt.pt \
  --norm-stats runs/striatum/diffusion/chunk_norm_stats.pt \
  --vae-checkpoint /models/stitched_vae.pt --cursor-model-csv cursor_models.csv \
  --mode cross --cross-n 2 --ema \
  --output runs/striatum/cross --device cuda
```

Matched-state sampling preserves the donor's history, cue, and cursor. The
cross-state dataset includes one matched trial per donor, then evaluates both cues
with cursor draws from their empirical training distributions. Only the last
cursor-history frame is replaced. With one sample per request, `--cross-n 2`
produces five requests per donor: one matched and four cross-state.

Rollouts update cursor position as
`cursor[t] = cursor[t-1] + coef * velocity[t] + intercept`. Generation stops at the
cursor target, after two consecutive chunks moving in the wrong direction, or at
the configured chunk limit. Repeated wrong-direction failures receive a discarded
marker. Use fresh output directories for each run.

Each session directory contains synthetic trials in `train/` and VAE-roundtripped
real trials in `real_train/`, `valid/`, and `test/`. Real reference trials begin at
frame 46. These outputs contain `x_recon`, `v_raw`, `c_raw`, and trial identifiers.
The manifest and sampling settings are saved alongside the trials.

## Train and evaluate decoders

Each training command fits all four regimes: real-only, synthetic-only,
synthetic-plus-real (`fused`), and synthetic pretraining followed by real finetuning.

```bash
python -m 2p_striatum.downstream_task.train \
  --data-dir runs/striatum/cross --session-index 0 --decoder linear --horizon 8 \
  --output runs/striatum/decoder_linear --device cuda

python -m 2p_striatum.downstream_task.train \
  --data-dir runs/striatum/cross --session-index 0 --decoder mlp --horizon 8 \
  --output runs/striatum/decoder_mlp --device cuda

python -m 2p_striatum.downstream_task.evaluate \
  --checkpoint runs/striatum/decoder_linear/finetune_seed12345.pt \
  --split-dir runs/striatum/cross/session_0/test \
  --output runs/striatum/decoder_linear/test_metrics.json --device cuda
```

The MLP has two 256-unit ReLU layers with dropout 0.1; the linear decoder has no
hidden layers. Defaults use AdamW, MSE, 50 training/finetuning epochs, and five seeds
starting at 12345. Checkpoints are saved as `<regime>_seed<seed>.pt`; `metrics.json`
contains the per-seed results. Finetuning keeps the pretrained model when its real
validation R² is at least as good as the finetuned model's.

For horizon `K`, features at frame `t` predict velocity at frames `t` through
`t+K-1`, with no windows crossing trial boundaries. Evaluation reports R², MSE,
MAE, mean Pearson correlation, and per-step R². Use real validation data for model
selection and real test data for final reporting. All commands also accept
`--device cpu`.
