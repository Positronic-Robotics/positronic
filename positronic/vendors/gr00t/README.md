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
serving codec require a matching simulator-specific `ee_frames` entry; the default is not a validated
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
The server runs there as `python -m gr00t_serving.server`. It imports the standalone `serving/` directory and the lightweight wrapper.
Positronic is not importable in this environment. The Docker build fails if installing the adapter requires changes to native package versions.
Training uses a separate Positronic environment at `/positronic/.venv`, created with the mounted uv cache.
Training and serving require a CUDA GPU.

The checkpoint also loads the gated `nvidia/Cosmos-Reason2-2B` backbone. The Hugging Face account
must have access to it, with its token available inside the container through `HF_TOKEN`,
`HF_TOKEN_PATH`, or the mounted Hugging Face cache's `token` file.

## Convert and fine-tune

From Positronic's `docker` directory:

```bash
mkdir -p "$PWD/groot-data"
docker compose run --rm --pull never -v "$PWD/groot-data:/data" lerobot-0_3_3-convert convert \
  --dataset=@positronic.cfg.ds.transform \
  --dataset.base=@positronic.cfg.ds.local \
  --dataset.base.path=/data/recordings \
  --dataset.transforms='["@positronic.vendors.gr00t.recipes.droid"]' \
  --output_dir=/data/datasets/my_task

docker compose run --rm --pull never -v "$PWD/groot-data:/data" groot-train \
  --input_path=/data/datasets/my_task \
  --output_path=/data/checkpoints \
  --exp_name=my_task \
  --num_train_steps=10000
```

Set `--dataset.base.path` to your recordings, or select another dataset configuration with `--dataset.base`.
For three views, set `--dataset.transforms='["@positronic.vendors.gr00t.recipes.droid_three_cameras"]'`.
The [training recipe](recipes.py) builds an episode transform. The [inference recipe](serving/recipe.py)
builds a JSON-compatible description using only the lightweight wrapper.
Both use [shared Python settings](serving/settings.py): observation keys, image size, camera mappings, tool frame,
rotation offset, and cadence.
`settings.droid()` returns a fresh dictionary, and `settings.droid_three_cameras()` derives the three-view variant.
This module uses only the standard library; the common model-server wrapper has no GR00T settings definitions.
The standalone recipe's observation names live in [`serving/keys.py`](serving/keys.py).
Tests check their agreement with the Positronic client without adding a server dependency on Positronic.
The settings also contain the control mode, which only inference uses.
The training transform preserves the recorded absolute pose, grip, and joint trajectories as action labels.
GR00T's checkpoint processor converts these labels to relative actions and restores absolute actions during inference.

The launcher reads camera keys from `meta/modality.json`; no separate modality selection is needed.
The exporter saves shared settings to `meta/positronic_model_settings.json`.
The file layout is a Positronic dataset convention defined beside the exporter metadata in
[`policy/codecs/metadata.py`](../../policy/codecs/metadata.py). The GR00T description names client
metadata fields without importing Positronic; its tests verify that the client and exporter read them.
Export frame rates must match the recipe's `action_fps`; change that setting to select a different rate.
Training copies this file into the experiment directory and, after a successful run, into each checkpoint.
Resume requires matching settings in the dataset and experiment directory, or no settings file in either.
If only one side has settings, resume is rejected before training or changing the recorded settings.
Datasets without this optional file can start new training runs or resume experiments without settings.

Customize settings in Python and pass the same dictionary to each recipe:

```python
from positronic.vendors.gr00t import recipes
from positronic.vendors.gr00t.serving import recipe
from positronic.vendors.gr00t.serving import settings as model_settings

settings = model_settings.droid()
settings[model_settings.IMAGE_SIZE] = [160, 90]
training = recipes.droid(settings=settings)
description = recipe.inference(settings)
```

The training and serving configurations use these dictionaries directly as Configuronic defaults.
Use `.override()` for named variants and dotted overrides for individual settings:
`--dataset.transforms.0.settings.image_size='[160, 90]'` for conversion and
`--pipeline.settings.image_size='[160, 90]'` for serving.

The JSON saved with datasets and checkpoints records the resolved settings for reproducibility.
For a fine-tuned checkpoint, configure serving to match those recorded settings.
Selecting a checkpoint does not select its settings.
The image size describes preprocessing before the native checkpoint processor, which also applies its own transforms.

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
The server resolves the selection once, before loading. Each server serves one checkpoint until it exits.
The source can name an experiment directory, a checkpoint directory, or their S3 paths.
A Hugging Face source uses `--model.model_source=hf://owner/model`; `--model.revision=<commit>` pins its revision.

The adapter checks camera roles, state fields, action layout and recorded model settings before it binds its listeners.
Conflicting model settings fail startup. Checkpoints without recorded settings use the selected Python preset.
Camera source names, observation source keys and controller settings can differ from training.

Native loading, warm-up and inference run on the wrapper's model thread, in the same process.
The `ready` handshake includes the client recipe and resolved session parameters.
The client performs pose conversion, image resize, optional JPEG encoding, action decoding and scheduling.
The server deserializes the model input and calls `Gr00tPolicy.get_action` directly.

Session query parameters can override `fps`, `horizon_sec`, `image_mappings`, `compress_images` and `jpeg_quality`.
The camera mapping must keep the checkpoint's camera roles. It can select different source observations for those roles.
These controls do not change native input dimensions, weights or model conventions.
Set launch defaults through `--pipeline.fps=10`, for example. The default execution horizon is one second.
Use `--pipeline.horizon_sec=None` to execute the full action chunk.

WebSocket listens on port 8000. Set `--websocket.served_address.port=9000` to choose another port.
Add `--grpc=@gr00t_serving.server.grpc --grpc.served_address.port=8001` to serve gRPC too.
`AUTH_TOKEN` enables bearer authentication. `--idle_timeout_min=20` enables idle shutdown.
Keepalive requests renew the idle timer. Docker stop sends SIGINT so the wrapper finishes active work and closes sessions.

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
