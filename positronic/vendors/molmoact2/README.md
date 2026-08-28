# MolmoAct2 in Positronic

> **Status: Work in Progress** – serving is wired up and runs end-to-end, but the policy has **not** been
> validated in evaluations.

## What is MolmoAct2?

[MolmoAct2](https://huggingface.co/allenai/MolmoAct2-DROID) is AllenAI's open vision-language-action model for
robot control. The `MolmoAct2-DROID` variant is fine-tuned on the DROID Franka dataset for absolute joint-pose
control. Positronic serves it directly from HuggingFace `transformers` (no fork), **inference-only** — there is
no convert or train step.

## Hardware

A ~5B-parameter model loaded in `bfloat16` (~10 GB of weights), so plan for a **16 GB+ GPU**. First start downloads the checkpoint from HuggingFace.

## Serve

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

The server serves a named policy pipeline — the codec plus the HuggingFace model source. MolmoAct2 ships one
pipeline, `droid`, which is the default subcommand. The codec lives server-side, so clients send raw
observations and receive decoded joint commands. `--host`, `--port`, `--recording_dir` and
`--idle_timeout_min` are the server's flags; the model is reached through the pipeline
(`--pipeline.source.hf_repo`, `.device_map`, `.norm_tag`, `.num_steps`), with defaults in
[`server.py`](./server.py). Sanity-check once warm:

```bash
curl http://localhost:8000/api/v1/models
# {"models": ["MolmoAct2-DROID"]}
```

## Run inference

Point the unified `.remote` client at the server (same client as every other vendor):

```bash
uv run --locked positronic eval run --eval=.sim.positronic.stack_cubes \
  --policy=.remote --policy.url=localhost:8000 \
  --output_dir=~/datasets/molmoact2_run
```

The model is DROID-pretrained, so its native target is a real franka_droid-style robot. **Sim eval grips
backwards** until the convention is unified ([#456](https://github.com/Positronic-Robotics/positronic/issues/456)).
See the [Inference Guide](../../../docs/inference.md) for the remote-policy protocol and options.

Codec arguments are tunable per session without restarting the server — the client passes them as query params
on the session URL (e.g. `--policy.url='localhost:8000?codec.fps=10&codec.flip_grip=true'`). The model source
(`hf_repo`, `device_map`, …) is fixed at launch and cannot be changed this way.
See the [offboard README](../../offboard/README.md) for the session-param rules.

## Codec

A [codec](../../../docs/codecs.md) maps raw recordings into the state/action space the model expects. MolmoAct2
ships one, `droid` (source: [`codecs.py`](./codecs.py)):

| Codec | Observation | Action |
|-------|-------------|--------|
| `droid` | 3 cameras (raw RGB, ordered `[exterior_1, exterior_2, wrist]`) + 8-D state `[joints(7), grip(1)]` + language task | Absolute joint positions (7) + grip → `JointPosition` command |

## Technical details

- **Action space**: absolute joint positions (7) + gripper (1), decoded straight into a `JointPosition`
  command (no IK at runtime).
- **Observation**: 3 cameras (2 exterior + 1 wrist) + 8-D state + language prompt.
- **Inference**: `norm_tag='franka_droid'`, continuous action mode; the model emits a 15-step action chunk at
  15 Hz, executed in full by the client's declared `ChunkedSchedule`.
- **Wire protocol**: Positronic's standard WebSocket protocol — see [Connect Your Model](../../../docs/connect-your-model.md).
