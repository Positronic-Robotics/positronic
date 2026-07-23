# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "h5py",
#     "numpy",
# ]
# ///
"""Extract a tiny env-server e2e fixture from a RoboLab recorded-demo HDF5.

The e2e replay needs only each demo's raw joint-position action log and its initial scene state — a few KB per
episode, not the multi-GB recording. Run once against a RoboLab recording (e.g. the repo's
``examples/recorded_data/RubiksCubeAndBananaTask/data.hdf5``), then commit the ``.npz`` next to the test::

    uv run --no-project positronic/simulator/robolab/tests/make_fixture.py \
        --demo-path <robolab>/examples/recorded_data/RubiksCubeAndBananaTask/data.hdf5 \
        --out positronic/simulator/robolab/tests/rubiks_cube_and_banana.npz
"""

import argparse

import h5py
import numpy as np


def _flatten(group: h5py.Group, prefix: str, arrays: dict) -> None:
    """The nested ``initial_state`` groups as flat ``'/'``-joined npz keys; the e2e replay un-flattens them.

    The recorder stacks one row per reset event (RoboLab's eval resets multiple times before stepping); the
    last row is the state the episode actually started from, kept as ``(1, D)`` — the per-env layout
    ``env.reset_to`` consumes on the single-env server.
    """
    for name, node in group.items():
        key = f'{prefix}/{name}'
        if isinstance(node, h5py.Group):
            _flatten(node, key, arrays)
        else:
            arrays[key] = node[()][-1:]


def main() -> None:
    parser = argparse.ArgumentParser(description='Pack RoboLab recorded demos into a tiny .npz e2e fixture.')
    parser.add_argument('--demo-path', required=True, help='path to a RoboLab recording data.hdf5')
    parser.add_argument('--out', required=True, help='destination .npz fixture')
    parser.add_argument('--num-demos', type=int, default=3, help='how many demos to pack (a few KB each)')
    args = parser.parse_args()

    arrays = {}
    with h5py.File(args.demo_path, 'r') as f:
        data = f['data']
        keys = sorted(data.keys(), key=lambda k: int(k.split('_')[1]))[: args.num_demos]
        for i, key in enumerate(keys):
            arrays[f'actions_{i}'] = data[key]['actions'][()]
            _flatten(data[key]['initial_state'], f'initial_state_{i}', arrays)

    np.savez(args.out, **arrays)
    print(f'wrote {args.out}: {len(keys)} episodes')


if __name__ == '__main__':
    main()
