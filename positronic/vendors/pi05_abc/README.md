# pi0.5 ABC checkpoints in Positronic

This adapter serves pi0.5 (openpi, JAX) checkpoints fine-tuned on the real teleoperation data of the
[ABC-130k release](https://huggingface.co/datasets/XDOF/ABC-130k) through Positronic's session protocol. Their training
code is the fork [yam-abc-reproduce](https://github.com/ninghaolu/yam-abc-reproduce), and their config is
`pi05_abc130k`. Each command serves one checkpoint:

| Command | Checkpoint | Tasks | Norm stats in the checkpoint |
| --- | --- | --- | --- |
| `bottles_16k` | [MikhailBird/pi05-abc-130k-task-finetune](https://huggingface.co/MikhailBird/pi05-abc-130k-task-finetune), step 16000 | put the plastic bottles in the bin | `assets/put_the_plastic_bottles_in_the_bin/norm_stats.json` |
| `abc4_30k` | A full fine-tune of `pi05_abc130k` from the pi0.5 base on the real data of four tasks, step 29999 (a local directory) | the four tasks below | `assets/abc4_yam/norm_stats.json` |

## Licences

The checkpoint card states no licence, and the Hub repository holds no licence file. The fork is under the Apache
License 2.0. The openpi repository holds its code under the Apache License 2.0, and it also carries a copy of the
Gemma Terms of Use in `LICENSE_GEMMA.txt`. pi0.5 builds on PaliGemma, a Gemma model.

## How it runs

The policy runs in the fork's openpi environment, as the fork's own server,
`yam_abc_reproduce/deploy/servers/openpi_server.py` at commit `e0aedd27`. That file is the same as the server snapshot
`inference/openpi_server.py` in the Hub repository of the bottles checkpoint. The [server](server.py) starts it on
`127.0.0.1` as a child process, with the norm stats of the command's checkpoint. It loads the checkpoint and compiles
the model before it listens. The server then sends one warm-up request, and then each encoded observation.

The [codec](codecs.py) follows the ABC-DiT codec of [`abc_dit`](../abc_dit/README.md), with the prompt of ABC's
real data. It holds its own copy, because no vendor imports another:

| Direction | Conversion |
| --- | --- |
| Views | `image.exterior`, `image.wrist_left`, `image.wrist_right` → the snapshot's `top`, `left`, `right`, resized with pad to 224x168, channels first. openpi pads each view to 224x224 |
| State | Per arm, 6 joints then the gripper, left arm first: 14 values |
| Gripper state | Positronic `0=open, 1=closed` → the checkpoint's `1=open, 0=closed`, using `1 - grip` |
| Prompt | The prompt of the task instruction, from the table below. The codec refuses any other task instruction |
| Action | 50 absolute actions of 14 values → one `JointPosition` and one grip target per arm, with `1 - width` for the grip |

The prompt is the task folder name of the ABC-130k release, with underscores as spaces, as the training's
`prompt_from_task` reads it. The codec takes the task instruction that ABC's sim reports:

| Task instruction | Prompt | `bottles_16k` | `abc4_30k` |
| --- | --- | --- | --- |
| `put the plastic bottles in the bin` | `put the plastic bottles in the bin` | yes | yes |
| `sweep away paper scraps from the table` | `sweep away the paper scraps from the table` | no | yes |
| `turn mug right side up` | `turn the mug right side up` | no | yes |
| `load plates into tabletop dish rack` | `load the plates into the dish rack` | no | yes |

The views of the release are 224x224 frames that hold the 640x480 camera view at 224x168, with black rows above and
below. The codec sends 224x168, and openpi pads it to the same 224x224 frame.

The client plays the first 15 actions of each chunk at 30 Hz, and then asks for the next chunk, as the author's
evaluation does. The gripper outputs can go outside 0 to 1.

ABC's cameras are RealSense D405s at 640x480. A session picks the crop of the yambox views with the `crop` query
parameter, as for ABC-DiT: `none`, `yambox` or `yambox_wide_top` (the default). The [ABC-DiT README](../abc_dit/README.md)
gives the boxes and the reason for each. For example:

    ws://<host>:<port>/api/v1/session?crop=yambox

The session metadata names the crop boxes under `pi05_abc.crops`.

## Setup

Set up the fork's environment at commit `e0aedd27`. Its openpi sources and its lock are the same as at `fa9c447e`, the
commit that the bottles checkpoint card names, and its server reads a norm stats file. Its `i2rt` submodule URL needs
an SSH key, so take it over HTTPS:

```bash
git clone https://github.com/ninghaolu/yam-abc-reproduce && cd yam-abc-reproduce
git checkout e0aedd27ed97a60b8b79cb813d6372dece292b6d
git -c url."https://github.com/".insteadOf=git@github.com: submodule update --init --recursive
uv sync --frozen --no-dev --extra deploy --group openpi
```

Download the inference files of the bottles checkpoint (12.4 GB). The training state (30 GB) is not necessary:

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

`abc4_30k` needs these files of its checkpoint directory: `params/`, `assets/abc4_yam/norm_stats.json` and
`_CHECKPOINT_METADATA`. The training state is not necessary.

## Docker image

[`docker/Dockerfile.serve-pi05-abc`](../../../docker/Dockerfile.serve-pi05-abc) builds a policy image that the
platform runs as a `kind: image` endpoint: the fork's environment, the checkpoint's inference files and the PaliGemma
tokenizer inside, offline, serving on `:8000`. `make -C docker push-serve-pi05-abc` builds it and pushes it to the
private Nebius registry, as `pi05-abc-yam`. The default build serves `bottles_16k` from the Hub. A build with
`--build-arg WEIGHTS=local --build-arg COMMAND=abc4_30k --build-arg CHECKPOINT=<name>` takes a local checkpoint
directory as the build context `checkpoint` (`--build-context checkpoint=<dir>`); the Dockerfile header gives the
command. The session query sets another crop, and arguments after the image name
override the command's flags, for example `docker run … <image> --pipeline.crop=yambox`.
