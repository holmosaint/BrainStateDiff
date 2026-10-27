# BrainStateDiff

State-conditioned diffusion models for neural activity, with downstream decoder
training and evaluation on real and generated data.

| Application | Generative model | Downstream task | Instructions |
|---|---|---|---|
| Widefield calcium imaging | Shared frame VAE and latent diffusion | 78-way stimulus classification | [Widefield guide](widefield_calcium_imaging/README.md) |
| TVSD | Subject-conditioned neural diffusion | Neural-to-image reconstruction | [TVSD guide](TVSD/README.md) |
| Two-photon striatum | Autoregressive neural/velocity diffusion | Velocity decoding and forecasting | [Striatum guide](2p_striatum/README.md) |

Each application has a `diffusion/` directory and a `downstream_task/` directory.
The workflow is to prepare data, train the diffusion model, generate matched-state
or cross-state samples, and train/evaluate a downstream decoder. Widefield also
includes VAE training.

## Installation

Use Python 3.10 or later. Clone the repository and install its dependencies:

```bash
git clone git@github.com:holmosaint/BrainStateDiff.git
cd BrainStateDiff
python -m pip install -r requirements.txt
```

Run all commands below from this directory. Examples use `--device cuda`; use
`--device cpu` for CPU execution. CUDA requires a compatible PyTorch installation
and NVIDIA driver. Recordings and large pretrained checkpoints are supplied
separately. Every command accepts `--help` for its full argument list.

## Widefield calcium imaging

### Prepare the data and configuration

The public [DANDI dataset 002040](https://dandiarchive.org/dandiset/002040/draft)
contains train/validation/test NWB files for `mouse1` and `mouse2`: 12,480 trials,
each with 28 frames at 96×96 pixels. The data are licensed under
[CC BY 4.0](widefield_calcium_imaging/data/LICENSE.md).

Install the download/conversion dependencies, download the data, and restore the
HDF5 layout used by the models:

```bash
python -m pip install -e '.[data]'
mkdir -p datasets
dandi download -o datasets https://dandiarchive.org/dandiset/002040/draft
python -m widefield_calcium_imaging.restore_hdf5 \
  --nwb-dir datasets/002040 --output widefield_calcium_imaging/data
```

This creates `mouse1_{train,val,test}.h5` and `mouse2_{train,val,test}.h5`.
See the [data guide](widefield_calcium_imaging/data/README.md) for the array layout.

Edit [example.json](widefield_calcium_imaging/example.json) to set
`image_embeddings` to your frozen ViT-MAE token file. It must map `image_ids0`
through `image_ids77` to arrays shaped `(tokens, 768)`. These image tokens are a
separate input and are not part of the neural-data download. Adjust model sizes,
batch sizes, and epochs as needed. The example data paths and VAE checkpoint path
match the commands below.

### Train and sample

```bash
python -m widefield_calcium_imaging.diffusion.train --stage vae \
  --config widefield_calcium_imaging/example.json \
  --output runs/widefield/vae --device cuda
python -m widefield_calcium_imaging.diffusion.train --stage diffusion \
  --config widefield_calcium_imaging/example.json \
  --output runs/widefield/diffusion --device cuda
python -m widefield_calcium_imaging.diffusion.generate \
  --config widefield_calcium_imaging/example.json \
  --checkpoint runs/widefield/diffusion/best.pt \
  --mode cross_state --output runs/widefield/cross --device cuda
```

For matched-state samples, use `--mode matched_state` and a separate output
directory. Generated files are named `<subject>_<mode>.h5`.

### Train and evaluate the decoder

Edit [downstream_task/example.json](widefield_calcium_imaging/downstream_task/example.json)
to select the subject and real/synthetic input files. Its default paths use mouse1
and the cross-state output above.

```bash
python -m widefield_calcium_imaging.downstream_task.train \
  --config widefield_calcium_imaging/downstream_task/example.json \
  --mode real --output runs/widefield/decoder --device cuda
python -m widefield_calcium_imaging.downstream_task.train \
  --config widefield_calcium_imaging/downstream_task/example.json \
  --evaluate-checkpoint runs/widefield/decoder/real/best.pt \
  --output runs/widefield/evaluation --device cuda
```

Training modes are `real`, `synthetic`, `mixed`, and `finetune`. Evaluation reports
classification accuracy and cross-entropy. See the [widefield guide](widefield_calcium_imaging/README.md)
for checkpoint settings and sampling options.

## TVSD

Prepare the TVSD response pickle, neural time-series HDF5 file, stimulus videos,
and pretrained TVSD VQ-VAE checkpoint described in the [TVSD guide](TVSD/README.md).
Set the data directory and edit the two [configuration files](TVSD/configs/).

```bash
export TVSD_DATA=/path/to/tvsd
export TVSD_VQVAE=/path/to/vqvae_autoencoder_ckpt.pth
python -m TVSD.diffusion.train --config TVSD/configs/diffusion.yaml \
  --output runs/tvsd/diffusion --device cuda
python -m TVSD.diffusion.sample --config TVSD/configs/diffusion.yaml \
  --checkpoint runs/tvsd/diffusion/best.pt --mode cross --factor 2 \
  --output runs/tvsd/cross.pickle --device cuda
python -m TVSD.downstream_task.train --config TVSD/configs/reconstruction.yaml \
  --vae-checkpoint "$TVSD_VQVAE" --regime fused --subject monkeyF \
  --synthetic-path runs/tvsd/cross.pickle \
  --output runs/tvsd/reconstruction_F --device cuda
python -m TVSD.downstream_task.evaluate --config TVSD/configs/reconstruction.yaml \
  --checkpoint runs/tvsd/reconstruction_F/epoch_0099.pth \
  --vae-checkpoint "$TVSD_VQVAE" --subject monkeyF \
  --synthetic-path runs/tvsd/cross.pickle --split val \
  --output runs/tvsd/validation_F --device cuda
```

Use `--mode matched` for matched-state generation. Decoder regimes are `real`,
`synthetic`, and `fused`; real-data finetuning uses `--finetune --resume`.
Train separate reconstructors for `monkeyF` and `monkeyN`. Select saved epochs using
real validation PSNR/LPIPS, then evaluate the selected checkpoints with `--split test`.
`best.pth` is selected by denoising loss, so it need not be the best reconstruction
checkpoint. Frozen ViT-MAE and VGG16 weights must be cached or downloadable.

## Two-photon striatum

Prepare the split session pickles, frozen stitched VAE checkpoint, `sessions.json`,
and cursor-model CSV using the [striatum guide](2p_striatum/README.md). Keep the
session order identical across latent extraction, training, and sampling.

```bash
python -m 2p_striatum.diffusion.extract_latents \
  --sessions sessions.json --vae-checkpoint /path/to/stitched_vae.pt \
  --cursor-model-csv cursor_models.csv --output runs/striatum/latents
python -m 2p_striatum.diffusion.train \
  --latent-dir runs/striatum/latents --sessions sessions.json \
  --cursor-model-csv cursor_models.csv --output runs/striatum/diffusion \
  --ema --device cuda
python -m 2p_striatum.diffusion.sample \
  --latent-dir runs/striatum/latents --sessions sessions.json \
  --checkpoint runs/striatum/diffusion/best_ckpt.pt \
  --norm-stats runs/striatum/diffusion/chunk_norm_stats.pt \
  --vae-checkpoint /path/to/stitched_vae.pt --cursor-model-csv cursor_models.csv \
  --mode cross --cross-n 2 --ema --output runs/striatum/cross --device cuda
python -m 2p_striatum.downstream_task.train \
  --data-dir runs/striatum/cross --session-index 0 --decoder linear --horizon 8 \
  --output runs/striatum/decoder --device cuda
python -m 2p_striatum.downstream_task.evaluate \
  --checkpoint runs/striatum/decoder/finetune_seed12345.pt \
  --split-dir runs/striatum/cross/session_0/test \
  --output runs/striatum/test_metrics.json
```

Use `--mode matched` for matched-state sampling and `--decoder mlp` for the MLP.
The decoder training command runs real-only, synthetic-only, fused, and finetuned
regimes. It reports R², MSE, MAE, and correlation on real data. The name
`2p_striatum` is supported by `python -m` despite beginning with a digit.

## Outputs

Training and evaluation write checkpoints, settings, and metrics to the supplied
output directory. Use a separate directory for each experiment and sampling mode.
Binary recordings, generated datasets, and model checkpoints are excluded from Git.
