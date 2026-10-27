# Widefield calcium imaging

Train a shared frame VAE, a state-conditioned diffusion model, and a 78-class image
decoder. Run all commands from the `BrainStateDiff` repository root. Replace
`--device cuda` with `--device cpu` for CPU execution.

## Prepare the inputs

Download the two-mouse dataset and restore its HDF5 files using the instructions in
[data/README.md](data/README.md). Each file contains `image_ids0` through
`image_ids77`, with float32 arrays shaped `(trials,28,96,96)`. The first 20 frames
are the history; frames 20–27 are the response.

Edit [example.json](example.json):

- Set `image_embeddings` to a PyTorch file containing a dictionary from each image
  key to frozen ViT-MAE token arrays shaped `(tokens,768)`. These tokens must be
  supplied separately from the neural recordings.
- Set the `subjects` train/validation/test paths. The example uses the restored
  `mouse1` and `mouse2` files.
- Set the VAE and diffusion model sizes, training options, and output checkpoint
  path. `vae.checkpoint` must point to the VAE trained in the first command below.

When loading a checkpoint, use its model architecture and diffusion schedule.
Subject names and their order identify the learned subject embeddings. For older
checkpoints, retain their saved internal subject names while pointing the data
paths to the renamed `mouse1`/`mouse2` files.

## Train the VAE and diffusion model

```bash
python -m widefield_calcium_imaging.diffusion.train --stage vae \
  --config widefield_calcium_imaging/example.json \
  --output runs/widefield/vae --device cuda

python -m widefield_calcium_imaging.diffusion.train --stage diffusion \
  --config widefield_calcium_imaging/example.json \
  --output runs/widefield/diffusion --device cuda
```

The VAE trains on individual frames with balanced sampling across mice. Diffusion
uses frozen VAE posterior means and normalization statistics fitted on the training
split. Training uses `train` and `val`; `test` is reserved for evaluation.

Each stage writes `best.pt`, `last.pt`, and `history.json`. Diffusion also writes
`latent_stats.pt`. The default objective predicts clean latents (`x0`); `epsilon`
is also supported. If EMA is enabled, select its weights during generation with
`--state-key ema_state_dict`.

## Generate synthetic responses

```bash
python -m widefield_calcium_imaging.diffusion.generate \
  --config widefield_calcium_imaging/example.json \
  --checkpoint runs/widefield/diffusion/best.pt \
  --mode matched_state --replicas 1 --steps 50 \
  --output runs/widefield/matched --device cuda

python -m widefield_calcium_imaging.diffusion.generate \
  --config widefield_calcium_imaging/example.json \
  --checkpoint runs/widefield/diffusion/best.pt \
  --mode cross_state --replicas 1 --steps 50 \
  --output runs/widefield/cross --device cuda
```

Matched-state sampling pairs each image with its recorded training history.
Cross-state sampling pairs each target image with every training history from the
same mouse, including histories recorded with that image. For balanced data,
cross-state generation produces 78 times as many responses per replica.

Outputs include `mouse1_cross_state.h5` and `mouse2_cross_state.h5`, or the
corresponding `matched_state` names. Each image group contains `samples` shaped
`(samples,8,96,96)`, source image/trial indices, and replica indices. Use fresh
output directories. `--cfg-scale 1` uses conditional generation; `--replicas`
controls repeated draws and `--seed` controls sampling randomness.

The configured crop must contain the complete history and response:
`frame_st <= stim_onset_frame - history_frames`, and any `frame_end` must be at
least `stim_onset_frame + target_frames`. The example uses `[0,28)`.

New checkpoints use `model_state_dict` or `ema_state_dict`. Older structured
checkpoints use `--state-key model_state` or `ema_state`; supply `--latent-stats`
for a separate statistics file and explicitly match the training timestep count
and beta schedule in the configuration.

## Train and evaluate the decoder

Edit [downstream_task/example.json](downstream_task/example.json) for one mouse.
Set `real_train`, `val`, `test`, and `synthetic_train`; the default synthetic path
matches the cross-state command above. Use a separate configuration and run
directory for mouse2. The supplied decoder architecture is in
[downstream_task/architecture.json](downstream_task/architecture.json).

```bash
python -m widefield_calcium_imaging.downstream_task.train \
  --config widefield_calcium_imaging/downstream_task/example.json \
  --mode real --output runs/widefield/decoder_real --device cuda

python -m widefield_calcium_imaging.downstream_task.train \
  --config widefield_calcium_imaging/downstream_task/example.json \
  --mode synthetic --output runs/widefield/decoder_synthetic --device cuda

python -m widefield_calcium_imaging.downstream_task.train \
  --config widefield_calcium_imaging/downstream_task/example.json \
  --mode mixed --output runs/widefield/decoder_mixed --device cuda

python -m widefield_calcium_imaging.downstream_task.train \
  --config widefield_calcium_imaging/downstream_task/example.json \
  --mode finetune --output runs/widefield/decoder_finetune --device cuda

python -m widefield_calcium_imaging.downstream_task.train \
  --config widefield_calcium_imaging/downstream_task/example.json \
  --evaluate-checkpoint runs/widefield/decoder_real/real/best.pt \
  --output runs/widefield/evaluation --device cuda
```

`real` and `synthetic` train on their respective datasets. `mixed` concatenates
real and synthetic trials. `finetune` first trains on synthetic trials, restores
the best validation checkpoint and Adam state, and then trains on real trials.
The configured epoch budget applies separately to each finetuning phase.

Real responses use frames `[20,28)` and synthetic responses use `[0,8)`. Keep the
same preprocessed response scale across training and evaluation. Checkpoints are
selected by real validation cross-entropy. Each run reports real test accuracy and
cross-entropy in `test_metrics.json`, with phase checkpoints and training histories
in subdirectories named for their training regime.
