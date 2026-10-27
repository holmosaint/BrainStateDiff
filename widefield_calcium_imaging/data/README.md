# Widefield calcium imaging data

The dataset is available as six NWB files in
[DANDI Dandiset 002040](https://dandiarchive.org/dandiset/002040/draft), under
[CC BY 4.0](LICENSE.md). It contains 12,480 trials from two mice viewing 78 natural
images. The NWB download is approximately 9.40 GB; restoring the training HDF5
files requires approximately 12.88 GB of additional space.

## Download and prepare training files

Run from the `BrainStateDiff` repository root:

```bash
python -m pip install -e '.[data]'
mkdir -p /path/to/downloads
dandi download -o /path/to/downloads \
  https://dandiarchive.org/dandiset/002040/draft
python -m widefield_calcium_imaging.restore_hdf5 \
  --nwb-dir /path/to/downloads/002040 \
  --output widefield_calcium_imaging/data
```

The restoration command creates:

```text
mouse1_train.h5   mouse1_val.h5   mouse1_test.h5
mouse2_train.h5   mouse2_val.h5   mouse2_test.h5
```

It preserves every movie value and places trials back at their original image and
trial indices. Existing HDF5 files are not overwritten. The model example
configurations already point to these filenames.

## Array layout and splits

Each HDF5 file contains `image_ids0` through `image_ids77`. Arrays are float32 with
axes `(trial, frame, height, width)`:

| Split, per mouse | Trials per image | Total trials | Array shape per image |
|---|---:|---:|---|
| Training | 64 | 4,992 | `(64,28,96,96)` |
| Validation | 8 | 624 | `(8,28,96,96)` |
| Test | 8 | 624 | `(8,28,96,96)` |

Each clip contains frames 0–27. Frames 0–19 are the prestimulus history; frames
20–27 are the response. The original train/validation/test assignments are retained.

```python
import h5py

with h5py.File("widefield_calcium_imaging/data/mouse1_train.h5", "r") as data:
    trial = data["image_ids0"][0]  # (28, 96, 96)
    history = trial[:20]
    response = trial[20:28]
```

Mouse1 is male and mouse2 is female. Both mice were three months old (`P3M`), with
recording date April 24, 2025. The frame interval is 0.024 seconds, approximately
41.67 Hz. Trial-to-session assignments are unspecified. NWB stores the calendar
date at midnight UTC as a date-only convention; recording time of day and timezone
were not supplied.

## Read NWB directly

NWB paths follow `sub-mouse1/sub-mouse1_desc-train_ophys.nwb`, with corresponding
`val`, `test`, and `mouse2` files. Each file contains an `ophys/trial_movies`
processing table. A row stores a `calcium_movie` shaped `(28,96,96)`, `image_id`,
`source_trial_index`, and `split`. Shared relative frame times are stored separately.
Row order follows image ID and within-image trial index, not acquisition chronology.

```python
from pynwb import NWBHDF5IO

path = "/path/to/downloads/002040/sub-mouse1/sub-mouse1_desc-train_ophys.nwb"
with NWBHDF5IO(path, "r") as io:
    table = io.read().processing["ophys"]["trial_movies"]
    movie = table["calcium_movie"][0]
    image_id = table["image_id"][0]
    original_trial_index = table["source_trial_index"][0]
```

The neural data release does not include the image-token dictionary required by
the diffusion model. Supply that input as described in the
[widefield model instructions](../README.md).
