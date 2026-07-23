# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "h5py",
#     "numpy",
# ]
# ///
"""Extract a tiny env-server e2e fixture from a LIBERO demo HDF5.

The e2e replay needs only each demo's action sequence and its initial full state — a few KB per episode, not the
multi-GB benchmark. Run once on a box that has the demos, then commit the ``.npz`` next to the test::

    uv run --no-project positronic/simulator/libero/make_fixture.py \
        --demo-path "$LIBERO_DATASETS/libero_spatial/<task>_demo.hdf5" \
        --out positronic/simulator/libero/tests/libero_spatial_task0.npz
"""

import argparse

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description='Pack a few LIBERO demos into a tiny .npz e2e fixture.')
    parser.add_argument('--demo-path', required=True, help='path to a {name}_demo.hdf5')
    parser.add_argument('--out', required=True, help='destination .npz fixture')
    parser.add_argument('--num-demos', type=int, default=3, help='how many demos to pack (a few KB each)')
    args = parser.parse_args()

    arrays = {}
    with h5py.File(args.demo_path, 'r') as f:
        data = f['data']
        keys = sorted(data.keys(), key=lambda k: int(k.split('_')[1]))[: args.num_demos]
        for i, key in enumerate(keys):
            arrays[f'actions_{i}'] = data[key]['actions'][()]
            arrays[f'init_state_{i}'] = data[key]['states'][()][0]

    np.savez(args.out, **arrays)
    print(f'wrote {args.out}: {len(keys)} episodes')


if __name__ == '__main__':
    main()
