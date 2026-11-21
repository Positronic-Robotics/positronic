# pi0.5 ABC bottles checkpoint in Positronic

This adapter serves [MikhailBird/pi05-abc-130k-task-finetune](https://huggingface.co/MikhailBird/pi05-abc-130k-task-finetune)
through Positronic's session protocol. The checkpoint is pi0.5 (openpi, JAX), fine-tuned on the real teleoperation
data of the [ABC-130k release](https://huggingface.co/datasets/XDOF/ABC-130k) for one bimanual YAM task, "put the
plastic bottles in the bin", at step 16000. Its training code is the author's fork
[yam-abc-reproduce](https://github.com/ninghaolu/yam-abc-reproduce), and its inference config is `pi05_abc130k`.

## Licences

The checkpoint card states no licence, and the Hub repository holds no licence file. The fork is under the Apache
License 2.0. The openpi repository holds its code under the Apache License 2.0, and it also carries a copy of the
Gemma Terms of Use in `LICENSE_GEMMA.txt`. pi0.5 builds on PaliGemma, a Gemma model.

## How it runs

The policy runs in the fork's openpi environment, as the author's own server snapshot, `inference/openpi_server.py`
in the Hub repository. The [server](server.py) starts that snapshot on `127.0.0.1` as a child process, with the task's
norm stats from `assets/put_the_plastic_bottles_in_the_bin/norm_stats.json`. The snapshot loads the checkpoint and
compiles the model before it listens. The server then sends one warm-up request, and then each encoded observation.

The [codec](codecs.py) follows the ABC-DiT codec of [`abc_dit`](../abc_dit/README.md), with the prompt of ABC's
real data. It holds its own copy, because no vendor imports another:

| Direction | Conversion |
| --- | --- |
| Views | `image.exterior`, `image.wrist_left`, `image.wrist_right` → the snapshot's `top`, `left`, `right`, resized with pad to 224x168, channels first. openpi pads each view to 224x224 |
| State | Per arm, 6 joints then the gripper, left arm first: 14 values |
| Gripper state | Positronic `0=open, 1=closed` → the checkpoint's `1=open, 0=closed`, using `1 - grip` |
| Prompt | `put the plastic bottles in the bin`. The codec refuses any other task instruction |
| Action | 50 absolute actions of 14 values → one `JointPosition` and one grip target per arm, with `1 - width` for the grip |

The client plays the first 15 actions of each chunk at 30 Hz, and then asks for the next chunk, as the author's
evaluation does. The gripper outputs can go outside 0 to 1.

ABC's cameras are RealSense D405s at 640x480. A session picks the crop of the yambox views with the `crop` query
parameter, as for ABC-DiT: `none`, `yambox` or `yambox_wide_top` (the default). The [ABC-DiT README](../abc_dit/README.md)
gives the boxes and the reason for each. For example:

    ws://<host>:<port>/api/v1/session?crop=yambox

The session metadata names the crop boxes under `pi05_abc.crops`.

## Setup

Set up the fork's environment at the commit that the checkpoint card names. Its `i2rt` submodule URL needs an SSH
key, so take it over HTTPS:

```bash
git clone https://github.com/ninghaolu/yam-abc-reproduce && cd yam-abc-reproduce
git checkout fa9c447e89beb978c44d9e06806ec161f61bba31
git -c url."https://github.com/".insteadOf=git@github.com: submodule update --init --recursive
uv sync --frozen --no-dev --extra deploy --group openpi
```

Download the inference files of the checkpoint (12.4 GB). The training state (30 GB) is not necessary:

```bash
hf download MikhailBird/pi05-abc-130k-task-finetune --revision 2e7ac0d02158cbe7b64926831d6f6748ef5fd45c \
    --local-dir <dir>/pi05-abc-130k-task-finetune-16000 \
    --include 'params/*' 'assets/*' 'inference/*' '_CHECKPOINT_METADATA' 'checkpoint_info.json' 'README.md'
```

openpi reads the PaliGemma tokenizer from `$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model`, and downloads it
when it is absent. Serve the checkpoint on a CUDA GPU:

```bash
uv run --locked python -m positronic.vendors.pi05_abc.server bottles_16k \
    --model.checkpoint=<dir>/pi05-abc-130k-task-finetune-16000 --model.fork=<yam-abc-reproduce>
```

## Docker image

[`docker/Dockerfile.serve-pi05-abc`](../../../docker/Dockerfile.serve-pi05-abc) builds a policy image that the
platform runs as a `kind: image` endpoint: the fork's environment, the checkpoint's inference files and the PaliGemma
tokenizer inside, offline, serving on `:8000`. `make -C docker push-serve-pi05-abc` builds it and pushes it to the
private Nebius registry, as `pi05-abc-yam`. The session query sets another crop, and arguments after the image name
override the command's flags, for example `docker run … <image> --pipeline.crop=yambox`.
