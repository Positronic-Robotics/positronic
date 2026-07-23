# Evaluation

Independent evaluation for VLA policies — in simulation and on real hardware, through one API.

## The problem

You ship a new checkpoint and want a clean answer to one question: is it actually better than last week's? On real hardware that's hard to get — building and maintaining a rig, operators, and stable baselines is a serious sink, and "looks better to three of us" isn't a number you can cite. So most teams either skip real-world eval or trust sim results that may not transfer.

## What you get

- **Sim and real through the same tasks and API.** Sim today: LIBERO, NVIDIA Isaac Lab, MuJoCo. Real hardware: the DROID setup (Franka FR3 + Robotiq 2F-85), bimanual next. Same model, same client — sim for cheap, broad iteration; real hardware as ground truth.
- **Blinded A/B.** Your checkpoint against your own previous checkpoints, or against our maintained baselines (π0.5, GR00T, SmolVLA, ACT) — randomized and blinded, so lighting and setup drift don't bias the result.
- **Every run returned.** Multi-view video, full telemetry, and the complete run dataset — not just a success rate. Yours to analyze.
- **Latency-honest execution.** On real hardware, inference and network delay are real — a slow model is scored as slow. In sim the world pauses during inference by default (as in other harnesses), but you can charge the model's measured inference time with `--inference_latency=True`, so sim scores reflect the delay the robot would actually feel — something sim-only harnesses can't model.

## How it works

You keep the weights. Your model runs as an inference server behind one WebSocket endpoint; a lightweight client streams observations and executes the returned trajectory — identical for sim and real. See [Connect your model](connect-your-model.md) and [Inference](inference.md).

## One CLI, any benchmark

The same command runs any benchmark — only the `--eval` target changes. Start your model as an inference server (see [Connect your model](connect-your-model.md)), then point the eval runner at it:

```bash
# LIBERO — the 40-task benchmark (four suites), in sim
uv run positronic eval run --eval=.sim.libero.all \
  --policy=.remote --policy.host=<gpu-host> \
  --eval.trial_count=10 --output_dir=~/evals/libero

# RoboLab — NVIDIA Isaac Lab, 120 DROID tasks, in sim
# Isaac Sim renders with RT cores, so RoboLab needs an RTX-class GPU host
# (L40S / L4 / RTX 40xx) — datacenter A100/H100 won't run it. One L40S fits
# the sim and a pi0.5-size policy on the same card.
uv run positronic eval run --eval=.sim.robolab.benchmark \
  --policy=.remote --policy.host=<gpu-host> \
  --eval.trial_count=10 --output_dir=~/evals/robolab
```

Narrow the scope to any sim target the catalog exposes — a single suite or category (`.sim.libero.spatial`, `.sim.robolab.visual`) or one task (`.sim.robolab.banana_in_bowl`). Add `--inference_latency=True` to charge the model's inference time in sim. Every trial is recorded as a Positronic dataset under `--output_dir`.

Real-hardware DROID evals take the same model endpoint, but we run them for you — operated and operator-scored on our fleet, not self-driven in sim. That's the paid path under [Two ways to start](#two-ways-to-start).

## Two ways to start

1. **Try it yourself, in sim.** A public checkpoint runs end-to-end in about ten minutes — see [Connect your model](connect-your-model.md).
2. **Get evaluated.** Send us a checkpoint or point us at an endpoint; the first real-hardware eval is on us, with full results back within a day. Reach out at hi@phail.ai.

## Public or private

Results are private to you by default. If you want the visibility, opt into the public leaderboard at [phail.ai](https://phail.ai) alongside π0.5, GR00T, SmolVLA, and ACT. Methodology and trial-count detail are in the [paper](https://arxiv.org/abs/2605.29710).
