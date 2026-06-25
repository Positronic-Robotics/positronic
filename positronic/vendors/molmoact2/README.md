# MolmoAct2 in Positronic

> **Status: Work in Progress** – the in-process server loads the pretrained `allenai/MolmoAct2-DROID`
> checkpoint and runs `predict_action` end-to-end, but the policy has **not** been validated in evaluations.
> Sim eval also hits a known gripper-convention gap (the MuJoCo sim's grip is inverted relative to the
> DROID convention this model uses — [#456](https://github.com/Positronic-Robotics/positronic/issues/456)).

## What is MolmoAct2?

[MolmoAct2](https://huggingface.co/allenai/MolmoAct2-DROID) is AllenAI's vision-language-action model. The
`MolmoAct2-DROID` variant is a pretrained checkpoint for the DROID action space (3 cameras + joint/gripper
state). It is a HuggingFace `transformers` model loaded with `trust_remote_code=True`; Positronic serves it
**in-process** by calling its `predict_action` method — there is no Positronic fork.

This integration is **inference-only**: it serves a fixed pretrained checkpoint. There is no convert or train
step (and nothing on the Nebius pipeline) — you serve the public checkpoint and run a client against it.

## Hardware

A CUDA GPU. The model loads in `bfloat16` with `device_map='auto'`; first start downloads the checkpoint from
HuggingFace.

## Serve

The `molmoact2` extra conflicts with the `lerobot` / `lerobot_0_3_3` extras (incompatible `transformers`
pins), so it is installed and run on its own with `--extra molmoact2`.

Via Docker Compose ([`docker/docker-compose.yml`](../../../docker/docker-compose.yml)), publishing the
WebSocket API on `8000`:

```bash
cd docker
docker compose run --rm --service-ports molmoact2-server
```

Or from a checkout:

```bash
uv run --python 3.13 --extra molmoact2 python -m positronic.vendors.molmoact2.server
```

The server owns the codec, so clients send raw observations and receive decoded joint commands. Config and
defaults (`hf_repo`, `num_steps`, `norm_tag`, `port`, …) are in [`server.py`](./server.py). Sanity-check once
warm:

```bash
curl http://localhost:8000/api/v1/models
# {"models": ["MolmoAct2-DROID"]}
```

## Run inference

Point the unified `.remote` client at the server (same client as every other vendor):

```bash
uv run --locked positronic-inference sim \
  --policy=.remote --policy.host=localhost --policy.port=8000 \
  --show_gui=True --output_dir=~/datasets/molmoact2_run
```

The model is DROID-pretrained, so its native target is a real franka_droid-style robot. **Sim eval grips
backwards** until the convention is unified ([#456](https://github.com/Positronic-Robotics/positronic/issues/456)).
See the [Inference Guide](../../../docs/inference.md) for the remote-policy protocol and options.

## Codec

A [codec](../../../docs/codecs.md) maps raw recordings into the state/action space the model expects. MolmoAct2
ships one, `droid` (source: [`codecs.py`](./codecs.py)):

| Codec | Observation | Action |
|-------|-------------|--------|
| `droid` | 3 cameras (raw RGB, ordered `[exterior_1, exterior_2, wrist]`) + 8-D state `[joints(7), grip(1)]` + language task | Absolute joint positions (7) + grip → `JointPosition` command |

The observation encoder is dedicated (`MolmoAct2ObservationCodec`) rather than a config variant of the shared
`ObservationCodec`: `predict_action` takes images as a single **ordered positional list** of **raw** frames
(the model tiles/resizes to 378×378 itself), whereas `ObservationCodec` emits separate named, pre-resized
image keys. The action side reuses the shared `absolute_joints_action`.

## Technical details

- **Action space**: absolute joint positions (7) + gripper (1), in raw robot units; the 8-vector decodes
  straight into a `JointPosition` command (no IK at runtime).
- **Observation**: 3 cameras (2 exterior + 1 wrist) + 8-D state + language prompt. A single exterior camera is
  duplicated to fill DROID's two exterior slots.
- **Inference**: `norm_tag='franka_droid'`, `num_steps=10` sampling steps, continuous action mode. The model
  emits a 15-step action chunk at 15 Hz (1 s of motion); the default client executes the whole chunk, so no
  client-side `--wrap` is needed.
- **Server**: subclasses `VendorServer` ([`positronic/offboard/vendor_server.py`](../../../positronic/offboard/vendor_server.py))
  and loads the model **in-process** on a worker thread, streaming `{"status": "loading"}` so the WebSocket
  handshake doesn't time out during the (minutes-long) load. Single model; `model_id` is the repo leaf
  (`MolmoAct2-DROID`).
- **Gripper convention**: the codec passes grip through unchanged on the DROID convention (`0=open, 1=closed`),
  matching Positronic's real Robotiq/DH drivers. The MuJoCo sim uses the opposite convention, so sim eval is
  inverted until [#456](https://github.com/Positronic-Robotics/positronic/issues/456) unifies them.
- **Wire protocol**: Positronic's standard WebSocket protocol — see [Connect Your Model](../../../docs/connect-your-model.md).
