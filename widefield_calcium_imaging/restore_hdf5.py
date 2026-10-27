"""Restore the training HDF5 layout from the BrainStateDiff NWB release."""
import argparse
from pathlib import Path

import h5py
import numpy as np
from pynwb import NWBHDF5IO


def restore_file(nwb_path, output_path):
    with NWBHDF5IO(str(nwb_path), 'r', load_namespaces=True) as io:
        table = io.read().processing['ophys']['trial_movies']
        image_ids = table['image_id'].data[:]
        indices = table['source_trial_index'].data[:]
        movies = table['calcium_movie'].data
        with h5py.File(output_path, 'x') as output:
            for image_id in sorted(np.unique(image_ids)):
                rows = np.flatnonzero(image_ids == image_id)
                assert np.array_equal(np.sort(indices[rows]), np.arange(len(rows)))
                dataset = output.create_dataset(f'image_ids{image_id}',
                    shape=(len(rows), *movies.shape[1:]), dtype=movies.dtype)
                for row in rows:
                    dataset[indices[row]] = movies[row]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nwb-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('widefield_calcium_imaging/data'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for subject in ('mouse1', 'mouse2'):
        for split in ('train', 'val', 'test'):
            source = args.nwb_dir/f'sub-{subject}'/f'sub-{subject}_desc-{split}_ophys.nwb'
            destination = args.output/f'{subject}_{split}.h5'
            restore_file(source, destination)
            print(destination, flush=True)


if __name__ == '__main__':
    main()
