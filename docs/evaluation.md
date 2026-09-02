# Evaluation

Independent evaluation for VLA policies — in simulation and on real hardware, through one API.

## The problem

You ship a new checkpoint and want a clean answer to one question: is it actually better than last week's? On real hardware that's hard to get — building and maintaining a rig, operators, and stable baselines is a serious sink, and "looks better to three of us" isn't a number you can cite. So most teams either skip real-world eval or trust sim results that may not transfer.

## What you get

- **One checkpoint, every target.** Sim: LIBERO, RoboLab (NVIDIA Isaac Lab), MolmoSpaces. Real hardware: the DROID setup (Franka FR3 + Robotiq 2F-85), bimanual next. Serve a DROID policy once and it runs across all of them, and on the rig, with nothing to port. Sim for cheap, broad iteration; real hardware as ground truth.
- **Native benchmarks, comparable scores.** Each benchmark runs with its own task definitions and horizons — the run reproduces the native benchmark rather than a re-interpretation of it — so your score is comparable with the benchmark's own published numbers.
- **Blinded A/B.** Your checkpoint against your own previous checkpoints, or against our maintained baselines (π0.5, GR00T, SmolVLA, ACT) — randomized and blinded, so lighting and setup drift don't bias the result.
- **Every run returned.** Multi-view video, full telemetry, and the complete run dataset — not just a success rate. Yours to analyze.
- **Latency-honest execution.** On real hardware, inference and network delay are real — a slow model is scored as slow. In sim the world pauses during inference by default (as in other harnesses), but you can charge the model's measured inference time with `--charge_inference_time=True`, so sim scores reflect the delay the robot would actually feel — something sim-only harnesses can't model.

## How it works

You keep the weights. Your model runs as an inference server behind one WebSocket endpoint; a lightweight client streams observations and executes the returned trajectory — identical for sim and real. See [Connect your model](connect-your-model.md) and [Inference](inference.md).

## Try it now

Two commands and you have a scored run: no checkpoint of your own, no rig. The first run pulls the image, the checkpoint and LIBERO itself, so give it a few minutes and some disk. Serve a public policy — Ubuntu with an NVIDIA card, and it holds its terminal, so run the eval from a second one.

```bash
cd docker && docker compose run --rm --service-ports openpi-server libero
```

Score a suite, and browse every trial — video, robot state, per-trial success:

```bash
# --policy.url is where the server is — <remote-server>:8000 if it runs on another machine
uv run positronic eval run --eval=.sim.libero.object \
  --policy=.remote --policy.url=localhost:8000 \
  --eval.trial_count=10 --output_dir=~/evals/libero

uv run positronic-server --dataset.path=~/evals/libero \
  --ep_table_cfg=@positronic.cfg.server.eval_table
```

**One server, many targets.** Every target publishes the same observation keys (see [the wire format](connect-your-model.md#the-wire-format)), so a DROID policy served once is scored on LIBERO, RoboLab, MolmoSpaces and our rig without a restart. Running everywhere is not the same as being comparable everywhere: each benchmark points `image.exterior` at its own camera, so a checkpoint scored on a target it was never trained for measures the viewpoint gap as much as the policy. Your own model is served the same way — [Connect your model](connect-your-model.md).

`--eval` takes any target the catalog exposes: a whole benchmark (`.sim.libero.all`), a suite or category (`.sim.robolab.visual`), or one task (`.sim.robolab.banana_in_bowl`). Add `--charge_inference_time=True` to charge the model's inference time in sim. Every trial is recorded as a Positronic dataset under `--output_dir`, carrying whatever verdict its benchmark reported.

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
