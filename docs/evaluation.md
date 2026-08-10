# Evaluation

Independent evaluation for VLA policies — in simulation and on real hardware, through one API.

## The problem

You ship a new checkpoint and want a clean answer to one question: is it actually better than last week's? On real hardware that's hard to get — building and maintaining a rig, operators, and stable baselines is a serious sink, and "looks better to three of us" isn't a number you can cite. So most teams either skip real-world eval or trust sim results that may not transfer.

## What you get

- **Sim and real through the same tasks and API.** Sim: LIBERO, RoboLab (NVIDIA Isaac Lab), MolmoSpaces. Real hardware: the DROID setup (Franka FR3 + Robotiq 2F-85), bimanual next. **An endpoint that runs against a sim target runs against the rig unchanged** — same wire protocol, same client, nothing to port. Sim for cheap, broad iteration; real hardware as ground truth.
- **Blinded A/B.** Your checkpoint against your own previous checkpoints, or against our maintained baselines (π0.5, GR00T, SmolVLA, ACT) — randomized and blinded, so lighting and setup drift don't bias the result.
- **Every run returned.** Multi-view video, full telemetry, and the complete run dataset — not just a success rate. Yours to analyze.
- **Latency-honest execution.** On real hardware, inference and network delay are real — a slow model is scored as slow. In sim the world pauses during inference by default (as in other harnesses), but you can charge the model's measured inference time with `--inference_latency=True`, so sim scores reflect the delay the robot would actually feel — something sim-only harnesses can't model.

## How it works

You keep the weights. Your model runs as an inference server behind one WebSocket endpoint; a lightweight client streams observations and executes the returned trajectory — identical for sim and real. See [Connect your model](connect-your-model.md) and [Inference](inference.md).

## One CLI, any benchmark

The same command runs any benchmark — only the `--eval` target changes, as long as the model behind the endpoint speaks that benchmark's observations. With no checkpoint of your own, serve a public reference policy: openpi fetches the checkpoint itself, nothing to mount. The server needs an NVIDIA GPU with the Container Toolkit, and holds the terminal — run the eval from a second one.

```bash
cd docker && docker compose run --rm --service-ports openpi-server libero
```

Score the suite, and browse every trial — video, robot state, per-trial success:

```bash
uv run positronic eval run --eval=.sim.libero.object \
  --policy=.remote --policy.url=localhost:8000 \
  --eval.trial_count=10 --output_dir=~/evals/libero

uv run positronic-server --dataset.path=~/evals/libero
```

Your own model is served the same way ([Connect your model](connect-your-model.md)) and the eval command does not change. The **deployment** does: a serving pipeline binds the codec its benchmark expects — `openpi-server libero` reads `image.agentview`, where a RoboLab target supplies the standard exterior image — so switch the deployment with the target (`openpi-server droid_jointpos` for RoboLab). Both of those presets serve public reference checkpoints; point `--pipeline.source.checkpoints_dir` at your own to evaluate it instead.

```bash
# LIBERO — the 40-task benchmark (four suites), in sim
uv run positronic eval run --eval=.sim.libero.all \
  --policy=.remote --policy.url=<gpu-host>:8000 \
  --eval.trial_count=10 --output_dir=~/evals/libero

# RoboLab — NVIDIA Isaac Lab, 120 DROID tasks, in sim
# Isaac Sim renders with RT cores, so RoboLab needs an RTX-class GPU host
# (L40S / L4 / RTX 40xx) — datacenter A100/H100 won't run it. One L40S fits
# the sim and a pi0.5-size policy on the same card.
uv run positronic eval run --eval=.sim.robolab.benchmark \
  --policy=.remote --policy.url=<gpu-host>:8000 \
  --eval.trial_count=10 --output_dir=~/evals/robolab
```

Narrow the scope to any sim target the catalog exposes — a single suite or category (`.sim.libero.spatial`, `.sim.robolab.visual`) or one task (`.sim.robolab.banana_in_bowl`). Add `--inference_latency=True` to charge the model's inference time in sim. Every trial is recorded as a Positronic dataset under `--output_dir`.

Real-hardware DROID evals take the same model endpoint, but we run them for you — operated and operator-scored on our fleet, not self-driven in sim. Write to hi@phail.ai for those.

## What a run cost

`--timing` records the wall-clock cost of a sim sweep — the split across reset, env step, inference and
recording, the machine's load, and the inference-latency distribution — into sidecar files beside the
dataset, and `positronic eval timing-report` reduces them. Sizing and performance work reads that; a run's
scores never do. See [Eval telemetry](telemetry.md).

## Three ways to start

1. **Run it yourself, in sim.** Self-serve, on your own compute — the reference-policy path above. RoboLab renders in Isaac Sim and wants an RTX-class host.
2. **Have us run the sim.** Point us at your endpoint and we run the suite on our GPUs and return the runs, so nobody on your side provisions a GPU or installs Isaac.
3. **Get evaluated on real hardware.** The same endpoint, on our rigs, operated and operator-scored. The first one is on us, with full results back within a day.

Both of the last two start at hi@phail.ai.

## Public or private

Results are private to you by default. If you want the visibility, opt into the public leaderboard at [phail.ai](https://phail.ai) alongside π0.5, GR00T, SmolVLA, and ACT. Methodology and trial-count detail are in the [paper](https://arxiv.org/abs/2605.29710).
