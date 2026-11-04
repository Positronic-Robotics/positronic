# DreamZero in Positronic

[DreamZero](https://github.com/dreamzero0/dreamzero) predicts actions and future video from recent
observations. The standalone server supports the public `GEAR-Dreams/DreamZero-DROID` checkpoint
(wan2.1, 14B): seven joint targets and gripper commands, using two exterior cameras and a wrist camera.

## Self-contained DROID image

The serving image includes the checkpoint, encoder/VAE weights and tokenizer. It needs no cache mount
or network access to load the model. The native CUDA environment comes from `positro/dreamzero-base`.
Build from the repository root:

```bash
make -C docker build-serve-dreamzero-droid

docker run --rm --gpus all --ipc=host -p 8000:8000 positro/dreamzero-droid:local

# Check that model loading and warmup work without network access.
docker run --rm --network none --gpus all --ipc=host positro/dreamzero-droid:local
```

[`Dockerfile.serve-droid`](./Dockerfile.serve-droid) pins each model repository to a commit. Its independent
weights stage downloads only inference assets; the full DROID checkpoint supplies the diffusion weights,
so the native loader skips preloading a second WAN backbone. `COPY --link` keeps weights separate from
serving code. Code-only rebuilds reuse the weight blobs, including when pushing to the same registry.
`make -C docker push-serve-dreamzero-droid IMAGE_TAG=<tag>` imports and exports registry build cache.

The image serves one model on one GPU, with one active session. Allow room for native inference caches
as well as weights. Inference speed must be measured on the intended GPU; a large enough GPU is not
necessarily fast enough to keep up with playback.

## Run an evaluation

```bash
uv run --locked positronic eval run --eval=.sim.robolab.banana_in_bowl \
  --policy=.remote --policy.address.host=<gpu-host> --policy.address.port=8000 \
  --eval.trial_count=2
```

RoboLab also needs an NVIDIA GPU for rendering. It can share a sufficiently large GPU with the policy.

Image resizing, JPEG encoding and action decoding run on the client. Every `sampling_period_sec`
(default `1/15`), the client sends the latest observation through ordinary inference. The recipe names
the timestamp field explicitly. The server retains a bounded FIFO history; new arrivals evict old
frames, and closing the session releases history and native caches. Between predictions it returns
`None`, leaving playback unchanged. Each prediction request includes its latest frame.

History offsets are in seconds: `[-23/15, -16/15, -8/15, 0]`. The first prediction uses only the current
frame. The camera mapping can send one exterior camera to both exterior slots on a two-camera rig.

The session query selects `mode=chunked` (default) or `mode=rtc`. Both use the same scheduler and player.
Chunked playback appends after the previous trajectory. After its first chunk, RTC-like playback starts
at the observation time and skips elapsed actions. This is video continuation; it does not freeze
previous actions like GR00T RTC.

`prediction_period_sec` defaults to 1.6 seconds (24 actions at 15 Hz). For overlapping continuations:

```bash
# Add to the evaluation command:
--policy.address.query='mode=rtc&prediction_period_sec=0.8'
```

Chunked playback is limited to the prediction period to keep the future plan bounded. At the outstanding
call limit or runtime capacity, the client skips new samples and counts them in `skipped_samples` metadata.
To compare behavior on slower hardware, `--charge_inference_time=False` pauses simulation during inference;
that comparison does not establish real-time performance.

## Training

Training is separate from the DROID serving image. [`Dockerfile.training`](./Dockerfile.training)
includes the Positronic training stack; `make -C docker build-dreamzero` builds it. The native base is
built with `make -C docker build-dreamzero-base` from [`Dockerfile`](./Dockerfile).

The following examples use the Docker Compose training services on an H100 host. With a remote Docker
context, set `CACHE_ROOT` to that host's home directory so its Hugging Face and AWS caches are mounted.
Replace the internal S3 paths with your own. Fine-tuned checkpoints are training outputs; this vendor's
serving image supports the public DROID checkpoint only.

### 1. Convert the dataset

The DreamZero codecs reuse the shared `lerobot_0_3_3` converter (a CPU step — run it on any box, or locally
with `uv run`). The sim stack-cubes dataset is converted with the `joints_ik_sim` codec (joint targets
reconstructed from recorded EE-pose targets via the `dm_control` IK solver):

```bash
cd docker
CACHE_ROOT=/home/<user> docker --context <h100> compose run --rm lerobot-0_3_3-convert convert \
  --dataset.dataset=@positronic.cfg.ds.sim.sim_stack_cubes \
  --dataset.codec=@positronic.vendors.dreamzero.codecs.joints_ik_sim \
  --output_dir=s3://interim/sim_stack/dreamzero/joints_ik/
```

### 2. Train

`dreamzero-train` runs `python -m positronic.vendors.dreamzero.train`, streaming the dataset from
`--input_path`. Presets (defined in [`train.py`](./train.py)) fix backbone + architecture + GPU count:

> Only `wan22_full` (wan2.2 full fine-tune) is validated. LoRA and `wan2.1` training presets are unverified.

| Preset | Backbone | Arch | GPUs | Resumable |
|--------|----------|------|------|-----------|
| `wan22_full_h100x1` | wan2.2 (5B) | full | 1 | No — warm-start to continue |
| `wan22_full_h100x8` | wan2.2 (5B) | full | 8 | Optimizer state not restored |
| `wan22_lora_h100x1` | wan2.2 (5B) | LoRA | 1 | LoRA trains from zero |
| `wan22_lora_h100x8` | wan2.2 (5B) | LoRA | 8 | Yes (optimizer restored) |
| `wan21_*` | wan2.1 (14B) | full / LoRA | 1 / 8 | As above |

**Fresh run** (from the base backbone):

```bash
cd docker
CACHE_ROOT=/home/<user> docker --context <h100> compose run --rm dreamzero-train \
  wan22_full_h100x1 \
  --input_path=s3://interim/sim_stack/dreamzero/joints_ik \
  --output_path=s3://checkpoints/sim_stack/dreamzero/ \
  --exp_name=<exp_name> \
  --max_steps=30000 --save_steps=2500 --gradient_accumulation_steps=4 --save_total_limit=9999
```

Checkpoints land at `s3://checkpoints/sim_stack/dreamzero/<exp_name>/checkpoint-<step>`.

**Warm-start** from a prior run's weights to continue training. The DreamZero trainer does not persist
optimizer state, so resuming restarts Adam from scratch (the optimizer state is lost) — warm-starting just
loads the weights and accepts a fresh optimizer + schedule, so aim to reach `--max_steps` in one run:

```bash
  ... dreamzero-train wan22_full_h100x1 ... \
  --init_from_checkpoint=s3://checkpoints/sim_stack/dreamzero/<prior_exp>/checkpoint-<step>
```

Multi-GPU presets (`*_h100x8`) run `torchrun --nproc_per_node=8`, so use them on an 8-GPU box.
