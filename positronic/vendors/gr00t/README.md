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

## Docker

Build the fork's base image, then the Positronic image:

```bash
# In the GR00T fork
make -C docker build
# In Positronic
make -C docker build-groot GROOT_BASE_IMAGE=positro/gr00t-base:local
cd docker
export IMAGE_TAG=local
```

The GR00T environment is `/opt/gr00t-venv` (Python 3.12, upstream locked dependencies).
Positronic has a separate environment at `/positronic/.venv`. Training and serving require a CUDA GPU.

## Convert and fine-tune

From Positronic's `docker` directory:

```bash
docker compose run --rm --pull never lerobot-0_3_3-convert convert \
  --dataset.codec=@positronic.vendors.gr00t.codecs.droid \
  --output_dir=~/datasets/groot/my_task

docker compose run --rm groot-train \
  --input_path=~/datasets/groot/my_task \
  --output_path=~/checkpoints/groot \
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
docker compose run --rm --service-ports groot-server droid \
  --pipeline.source.checkpoints_dir=~/checkpoints/groot/my_task
```

Select `droid_three_cameras` for a checkpoint trained on three views.
Use `--pipeline.source.checkpoint=10000` to select a saved step. Omit it to serve the latest.
A Hugging Face source uses `--pipeline.source.checkpoints_dir=hf://owner/model`.

## Adapter parity tests

The GR00T source is included in the image at `/gr00t`. From the image's `/positronic` directory:

```bash
GR00T_REFERENCE_ROOT=/gr00t uv run --no-sync --python 3.12 pytest \
  -o addopts= positronic/vendors/gr00t/tests
```

The cross-repository tests compare encoded tool poses and image pixels directly against the
upstream DROID functions. Model forward and fine-tuning checks additionally require the checkpoint
weights and a GPU.
