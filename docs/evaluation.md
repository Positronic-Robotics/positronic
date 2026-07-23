# Evaluation

Independent evaluation for VLA policies — in simulation and on real hardware, through one API.

## The problem

You ship a new checkpoint and want a clean answer to one question: is it actually better than last week's? On real hardware that's hard to get — building and maintaining a rig, operators, and stable baselines is a serious sink, and "looks better to three of us" isn't a number you can cite. So most teams either skip real-world eval or trust sim results that may not transfer.

## What you get

- **Sim and real through the same tasks and API.** Sim today: LIBERO, NVIDIA Isaac Lab, MuJoCo. Real hardware: the DROID setup (Franka FR3 + Robotiq 2F-85), bimanual next. Same model, same client — sim for cheap, broad iteration; real hardware as ground truth.
- **Blinded A/B.** Your checkpoint against your own previous checkpoints, or against our maintained baselines (π0.5, GR00T, SmolVLA, ACT) — randomized and blinded, so lighting and setup drift don't bias the result.
- **Every run returned.** Multi-view video, full telemetry, and the complete run dataset — not just a success rate. Yours to analyze.
- **Latency-honest execution.** Actions carry absolute timestamps and run on the client's own clock, so a slow model is scored as a slow model — not silently frozen while the world waits. (Most eval harnesses pause the world during inference; we don't.)

## How it works

You keep the weights. Your model runs as an inference server behind one WebSocket endpoint; a lightweight client streams observations and executes the returned trajectory — identical for sim and real. See [Connect your model](connect-your-model.md) and [Inference](inference.md).

## Two ways to start

1. **Try it yourself, in sim.** A public checkpoint runs end-to-end in about ten minutes — see [Connect your model](connect-your-model.md).
2. **Get evaluated.** Send us a checkpoint or point us at an endpoint; the first real-hardware eval is on us, with full results back within a day. Reach out at hi@phail.ai.

## Public or private

Results are private to you by default. If you want the visibility, opt into the public leaderboard at [phail.ai](https://phail.ai) alongside π0.5, GR00T, SmolVLA, and ACT. Methodology and trial-count detail are in the [paper](https://arxiv.org/abs/2605.29710).
