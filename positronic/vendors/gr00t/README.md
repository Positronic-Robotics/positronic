# GR00T N1.7 DROID

Positronic uses [`nvidia/GR00T-N1.7-DROID`](https://huggingface.co/nvidia/GR00T-N1.7-DROID)
through our [GR00T fork](https://github.com/Positronic-Robotics/gr00t).
The checkpoint defines the model architecture, image processor and relative-action conversion.
The adapter follows [upstream's DROID robot client](https://github.com/NVIDIA/Isaac-GR00T/tree/main/examples/DROID).

## Representation

- `droid`: wrist + one exterior camera, matching the published checkpoint.
- `droid_three_cameras`: wrist + two exterior cameras. Fine-tune with this layout, then serve that checkpoint.
- RGB images first use the client's bilinear padded resize to **320×180**. The checkpoint processor
  resizes the shortest edge to **256**, crops **95%**, resizes the shortest edge again, then runs
  its vision processor. Positronic does not apply an additional square crop.
- State is absolute tool position + row-based 6D rotation, gripper position and seven joint positions.
  Poses move from Positronic's default tool frame to `DROID_EE_FRAME`, then receive upstream's
  DROID rotation correction. Gripper convention is **1 = closed**.
- Conversion writes recorded state trajectories as absolute action labels. The checkpoint processor
  computes pose-relative EEF actions and joint offsets for training, then restores absolute actions
  at inference. Positronic does not subtract poses or rotations itself.
- Inference executes the first **15 of 40** predicted joint targets at **15 Hz**, with the DROID
  impedance settings and a gripper threshold of **0.5**, matching upstream's default execution horizon.

The published two-camera checkpoint does not consume an extra exterior view. Select
`droid_three_cameras` for both conversion and serving when fine-tuning with three views.
N1.6 checkpoints require an N1.6 image; their custom action schemas are incompatible with this adapter.

The default pose transform matches Franka rigs and the RoboLab adapter. The bundled MuJoCo
Panda reports poses at a different tool frame; using the default transform on its recordings
(including `sim_stack_cubes`) introduces a 45 mm offset. That frame alignment is tracked in
[#550](https://github.com/Positronic-Robotics/positronic/issues/550). Such datasets and their
serving codec require a matching simulator-specific `ee_frame`; the default is not a validated
native-checkpoint configuration for that simulator.

## Docker

Build the Positronic image using the published GR00T base:

```bash
make -C docker build-groot
cd docker
export IMAGE_TAG=local
```

`build-groot` pulls `positro/gr00t-base:latest`, published by the fork's GitHub workflow.

The GR00T environment is `/opt/gr00t-venv` (Python 3.12, upstream locked dependencies).
Positronic creates its separate environment at `/positronic/.venv` on startup using the mounted uv cache.
Training and serving require a CUDA GPU.

The checkpoint also loads the gated `nvidia/Cosmos-Reason2-2B` backbone. The Hugging Face account
must have access to it, with its token available inside the container through `HF_TOKEN`,
`HF_TOKEN_PATH`, or the mounted Hugging Face cache's `token` file.

## Convert and fine-tune

From Positronic's `docker` directory:

```bash
mkdir -p "$PWD/groot-data"
docker compose run --rm --pull never -v "$PWD/groot-data:/data" lerobot-0_3_3-convert convert \
  --dataset.codec=@positronic.vendors.gr00t.codecs.droid \
  --output_dir=/data/datasets/my_task

docker compose run --rm --pull never -v "$PWD/groot-data:/data" groot-train \
  --input_path=/data/datasets/my_task \
  --output_path=/data/checkpoints \
  --exp_name=my_task \
  --num_train_steps=10000
```

Supply the conversion command's dataset configuration for your recordings as usual.
For three views, replace the codec with `positronic.vendors.gr00t.codecs.droid_three_cameras`.
The launcher reads camera keys from `meta/modality.json`; no separate modality selection is needed.

`--base_model` defaults to `nvidia/GR00T-N1.7-DROID`. Standard controls are `--batch_size`,
`--learning_rate`, `--num_train_steps`, `--save_steps`, `--num_workers` and `--resume=True`.
Resume restores the latest saved training state in the experiment directory.
The fork retains checkpoint architecture and preprocessing while applying upstream's standard
fine-tuning settings. Dataset statistics are computed by GR00T.

## Serve

Published checkpoint, without fine-tuning:

```bash
docker compose run --rm --service-ports groot-server droid
```

Fine-tuned checkpoint:

```bash
docker compose run --rm --service-ports --pull never -v "$PWD/groot-data:/data" groot-server droid \
  --model.model_source=/data/checkpoints/my_task
```

Select `droid_three_cameras` for a checkpoint trained on three views.
Use `--model.checkpoint=10000` to select a saved step. Omit it to serve the latest.
A Hugging Face source uses `--model.model_source=hf://owner/model`.

## Client pipeline

The client registry supports two GR00T components, both at version 1:

- `gr00t_droid`, with `image_mappings`: the DROID observation and robot-command conversions.
- `gr00t_action_chunk`, without arguments: converts the native `(actions, info)` result into action
  rows. Arrays must have shape `(1, T, D)` and share a time horizon. Auxiliary `info` is excluded
  from robot commands.

A complete client pipeline can be constructed and serialized inside Positronic:

```python
from positronic.policy import Sequential
from positronic.policy.codec import EncodeImages
from positronic.policy.processors import ChunkedSchedule, PauseOnUnavailable
from positronic.vendors.gr00t.codecs import ActionChunk, droid

pipeline = Sequential(
    PauseOnUnavailable(),
    ChunkedSchedule(fps=15, horizon_sec=1.0),
    droid() | ActionChunk() | EncodeImages(quality=90),
)
description = pipeline.to_spec()
```

Use `droid_three_cameras()` for the three-camera layout. Image encoding follows observation conversion
and preserves the model's batch and time dimensions. `EncodeImages` selects all RGB images by default;
`paths=[["video", "wrist_image_left"]]` compresses only that camera, and `paths=[]` uses lossless arrays.

A [native model server](../../../model_server/README.md) sends this description as `Session.client_stack`.
Its script builds the same plain data with the wrapper's `component` and `sequence` helpers, without
importing Positronic. The client performs conversion, JPEG encoding, native-result decoding and scheduling.
The `groot-server` Docker entrypoint uses the legacy server and its server-side codecs.

## Adapter parity tests

The GR00T source is included in the image at `/gr00t`. From the image's `/positronic` directory:

```bash
uv sync --locked --python 3.12
GR00T_REFERENCE_ROOT=/gr00t uv run --no-sync --python 3.12 python -m pytest \
  -o addopts= positronic/vendors/gr00t/tests
```

The cross-repository tests compare encoded tool poses and image pixels directly against the
upstream DROID functions. Model forward and fine-tuning checks additionally require the checkpoint
weights and a GPU.
