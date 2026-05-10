"""Read a Lance-converted Positronic dataset — demo / sanity script.

Usage:
    uv run --extra lance python -m positronic.vendors.lance.demo_read \\
        /path/to/output_dir

Where `output_dir` is the path passed to `convert.py` (it contains both
`data.lance/` and `videos/`).
"""

import sys
from pathlib import Path

import av
import lance
import numpy as np


def main(output_dir: str) -> None:
    root = Path(output_dir)
    ds = lance.dataset(str(root / 'data.lance'))

    print(f'=== {root} ===')
    print(f'rows: {ds.count_rows()}')
    print()
    print('schema:')
    print(ds.schema.to_string(show_field_metadata=False))
    print()

    # Batched iteration — one episode per row.
    print('first batch (one row = one episode):')
    batch = next(iter(ds.to_batches(batch_size=4, columns=['trajectory_length', 'action', 'observation_state'])))
    for i, r in enumerate(batch.to_pylist()):
        action = np.asarray(r['action'])
        state = np.asarray(r['observation_state'])
        print(f'  row={i}  T={r["trajectory_length"]:4d}  action={action.shape}  observation_state={state.shape}')
    print()

    # Resolve a relative mp4 uri and decode the first frame.
    row = ds.to_table(
        limit=1,
        columns=[
            'current_task',
            'observation_images_left_uri',
            'observation_images_left_num_frames',
            'observation_images_left_width',
            'observation_images_left_height',
        ],
    ).to_pylist()[0]
    video_path = root / row['observation_images_left_uri']
    print(f'decoding first frame (task: {row["current_task"]!r})')
    print(f'  uri (relative):   {row["observation_images_left_uri"]}')
    print(f'  declared frames:  {row["observation_images_left_num_frames"]}')
    print(f'  declared size:    {row["observation_images_left_width"]}x{row["observation_images_left_height"]}')
    container = av.open(str(video_path))
    frame = next(iter(container.decode(video=0)))
    img = frame.to_ndarray(format='rgb24')
    print(f'  decoded frame:    shape={img.shape}, dtype={img.dtype}')
    container.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <output_dir>', file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])
