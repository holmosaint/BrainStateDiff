# TVSD: V4 generation and image reconstruction

Run all commands from the `BrainStateDiff` repository root. Use `--device cpu`
in place of `--device cuda` for CPU execution.

## Data and pretrained weights

Set `TVSD_DATA` to a directory containing:

```text
ventral_stream_dataset.pickle
ventral_stream_timeseries_preprocessed_no_baseline_nb30.h5
ventral_stream/<stimulus name>.mp4
```

The pickle contains `train_stimuli`, `train_activity`, `test_stimuli`, and
`test_activity`; stimulus names omit the `.mp4` suffix. The HDF5 contains
`train_timeseries` and `test_timeseries` arrays shaped `(trials,30,315)`. Channels
0–197 belong to monkey N, and 198–314 to monkey F. The supplied split assigns
21,136 training and 1,112 validation images; the test split has 100 images.
Images are loaded from the first RGB frame of each stimulus video, resized to
256 pixels, and normalized to `[-1,1]`.

Neural diffusion uses the frozen `facebook/vit-mae-base` encoder. For offline use,
cache its weights or set `model_params.vit_model_id` to a local model directory.
Image reconstruction additionally requires a trained TVSD VQ-VAE checkpoint,
`vqvae_autoencoder_ckpt.pth`. Its architecture is included, but the checkpoint must
be supplied separately. LPIPS evaluation requires torchvision's pretrained VGG16
weights; cache `vgg16-397923af.pth` for offline use. The small LPIPS calibration
weights are included in the repository.

## Train neural diffusion

Edit [configs/diffusion.yaml](configs/diffusion.yaml) to set model and training
options, then run:

```bash
export TVSD_DATA=/path/to/tvsd
python -m TVSD.diffusion.train --config TVSD/configs/diffusion.yaml \
  --output runs/tvsd/generator --device cuda
```

The model generates mean evoked activity over bins `[15,25)` conditioned on mean
prestimulus activity over `[0,10)`, image tokens, and subject identity. Normalization
is fitted on training responses. Validation selects `best.pt`; sampling uses its
EMA weights unless `--no-ema` is specified. Training checkpoints contain model,
EMA, optimizer, and configuration state.

## Generate synthetic responses

```bash
python -m TVSD.diffusion.sample --config TVSD/configs/diffusion.yaml \
  --checkpoint runs/tvsd/generator/best.pt --output runs/tvsd/matched_x2.pickle \
  --mode matched --factor 2 --steps 50 --device cuda

python -m TVSD.diffusion.sample --config TVSD/configs/diffusion.yaml \
  --checkpoint runs/tvsd/generator/best.pt --output runs/tvsd/cross_x2.pickle \
  --mode cross --factor 2 --steps 50 --cross-seed 0 --device cuda
```

Matched sampling produces `factor` draws per observed training pair. Cross-state
sampling independently permutes image labels and neural histories for each round.
The output preserves the 315-channel scale/order and includes the real validation
and test responses and stimulus labels. `--steps` controls DDIM sampling steps;
`--guidance-scale` controls classifier-free guidance.

## Train image reconstruction

Edit [configs/reconstruction.yaml](configs/reconstruction.yaml), including
`train_params.ldm_epochs` and model dimensions matching the VQ-VAE checkpoint.
Train a separate reconstructor for each subject (`monkeyF` or `monkeyN`):

```bash
export TVSD_VQVAE=/path/to/vqvae_autoencoder_ckpt.pth

python -m TVSD.downstream_task.train --config TVSD/configs/reconstruction.yaml \
  --vae-checkpoint "$TVSD_VQVAE" --regime real --subject monkeyF \
  --output runs/tvsd/real_F --device cuda

python -m TVSD.downstream_task.train --config TVSD/configs/reconstruction.yaml \
  --vae-checkpoint "$TVSD_VQVAE" --regime synthetic \
  --synthetic-path runs/tvsd/cross_x2.pickle --subject monkeyF \
  --output runs/tvsd/synthetic_F --device cuda

python -m TVSD.downstream_task.train --config TVSD/configs/reconstruction.yaml \
  --vae-checkpoint "$TVSD_VQVAE" --regime fused \
  --synthetic-path runs/tvsd/cross_x2.pickle --subject monkeyF \
  --output runs/tvsd/fused_F --device cuda
```

`fused` concatenates synthetic and real training examples. Synthetic and fused
training use the real validation/test arrays retained in the synthetic pickle.
Real-only training uses averages over the configured HDF5 response bins; the real
portion of fused training uses the time-averaged pickle responses.

For synthetic pretraining followed by real finetuning, select the saved synthetic
epoch with the lowest validation LPIPS and supply it through `--resume`:

```bash
python -m TVSD.downstream_task.train --config TVSD/configs/reconstruction.yaml \
  --vae-checkpoint "$TVSD_VQVAE" --regime real --finetune \
  --resume /path/to/validation_selected_synthetic_epoch.pth --subject monkeyF \
  --output runs/tvsd/finetune_F --device cuda
```

Finetuning restores the model and Adam state, resets the epoch counter, and uses
pickle-averaged real responses. Its saved `config.yaml` records this data choice;
use that configuration when evaluating the finetuned model.

## Evaluate reconstructions

```bash
python -m TVSD.downstream_task.evaluate --config TVSD/configs/reconstruction.yaml \
  --checkpoint runs/tvsd/fused_F/epoch_0099.pth --vae-checkpoint "$TVSD_VQVAE" \
  --synthetic-path runs/tvsd/cross_x2.pickle --subject monkeyF --split val \
  --output runs/tvsd/fused_F/validation --device cuda
```

Supply the same `--synthetic-path` when evaluating synthetic or fused training.
For real-only or finetuned models, omit it; for finetuned models, also use
`--config runs/tvsd/finetune_F/config.yaml`.

The trainer writes `latest.pth`, `best.pth`, periodic `epoch_XXXX.pth` checkpoints,
and `config.yaml`. `best.pth` is selected by validation denoising loss. To select by
reconstruction quality, evaluate the saved epochs on validation data, choose the
maximum-PSNR or minimum-LPIPS checkpoint, and evaluate that checkpoint with
`--split test` in a separate output directory.

Evaluation writes `per_image.csv` and `metrics.json`. PSNR uses uint8 images with
data range 255; LPIPS uses unquantized images mapped to `[-1,1]`. Reconstruction
uses ancestral DDPM sampling over the configured diffusion schedule.
