# ABC-DiT in Positronic

ABC-DiT is the bimanual YAM policy of the [ABC project](https://github.com/amazon-far/abc). This adapter serves its
multi-task sim checkpoint, `abc_dit_xl_200k_model.pt`, through Positronic's session protocol. The checkpoint trained
on real teleoperation data and on sim data of five ABC tasks:

| Task instruction | Prompt it trained under |
| --- | --- |
| put the plastic bottles in the bin | sim throw plastic bottles in bin |
| sweep away paper scraps from the table | sim sweep away paper scraps from the table |
| turn mug right side up | sim turn mug right side up |
| load plates into tabletop dish rack | sim load plates into tabletop dish rack |
| hang mug on mug rack | sim hang the mug on the mug rack |

The codec sends the trained prompt for each instruction and refuses an instruction outside this table. The release
sidecar `abc_dit_xl_200k_model.json` records the same map by task name. ABC's sim has no success evaluator for
`hang_mug_on_mug_rack`, so that task scores no successes.

## Licences

The ABC code and the DiT checkpoints are under the Apache License 2.0. The checkpoints embed a vision backbone derived
from DINOv3, and the [DINOv3 License](LICENSE-DINOv3.md) governs it. A publication of results from this model must
acknowledge the use of DINOv3. An image that holds the weights must carry a copy of the DINOv3 License. A service
that makes the policy available to others must display "Built with DINOv3". See [NOTICE](NOTICE).

## How it runs

The policy runs in ABC's own interpreter, as ABC's websocket policy server `deploy/serve_policy.py`. The
[backend](backend.py) starts that server on `127.0.0.1` only. The
[server](server.py) starts it as a child process, waits until it listens, sends one warm-up request, and then sends it
each encoded observation. Each request carries standard normal flow noise from one generator, seeded at 0, as ABC's
sim eval draws it.

The [codec](codecs.py) handles all embodiment conversion:

| Direction | Conversion |
| --- | --- |
| Views | `image.exterior`, `image.wrist_left`, `image.wrist_right` → ABC's `top`, `left`, `right`, resized with pad to 224x168, channels first |
| State | Per arm, 6 joints then the gripper width, left arm first: 14 values |
| Gripper state | Positronic `0=open, 1=closed` → ABC `1=open, 0=closed`, using `1 - grip` |
| Prompt | The task instruction → the prompt that the checkpoint trained under |
| Action | 30 actions of 14 values → one `JointPosition` and one grip target per arm, with `1 - width` for the grip |

The client plays the first 15 actions of each chunk at 30 Hz, and then asks for the next chunk. This is the
synchronous mode of ABC's sim eval with no action prefix. ABC's own sim eval runs real-time chunking by default, which
this adapter does not do, so its numbers can differ from ABC's published numbers.

ABC renders its sim training views at 224x168. A view of another aspect ratio is padded, not stretched, so the scene
fills less of the image than in training. Render the cameras at 224x168, or at the same 4:3 aspect ratio, to match
the training data, or crop each view with a `CropBox` in the codec's `crops`.

ABC's cameras are RealSense D405s with a 58 degree vertical field of view, at 640x480. The `multitask_200k_yambox_crop`
config crops the views of the yambox ZED cameras (960x600, focal length 367 px) to 4:3 before the resize:

| View | Crop, in pixels of 960x600 | Why |
| --- | --- | --- |
| `image.exterior` | x 256-800, y 192-600 | 58 degrees vertical, the field of view of ABC's top view, over the arms and the table |
| `image.wrist_left`, `image.wrist_right` | x 80-880, y 0-600 | the open fingertips lie as far apart, as a share of the width, as in ABC's wrist views |

That config has the client send each view at up to 448x336, so each crop still holds at least 224x168 pixels.

ABC trained the five sim tasks under prompts that start with `sim `, within about 3553 hours of real teleoperation
data and 98 hours of sim data. The codec's `prompt_prefix` sets that prefix. `multitask_200k_yambox_crop_no_sim_prefix`
sends the same prompts without it.

## Setup

Set up ABC's interpreter at the commit whose server fields and layout this adapter follows. Then download the
checkpoint (8.1 GB):

```bash
git clone https://github.com/amazon-far/abc && cd abc
git checkout 6c467cebcecf16a4dce79e6fd87a7ca2281c3ef0
uv sync
uv pip install websockets opencv-python-headless msgpack  # ABC's policy server imports all three
uv run prepare.py --pretrained                    # writes cache/abc_dit_xl_200k_model.pt and its sidecar
```

ABC's server downloads the CLIP text encoder to `~/.cache/clip` at its first start. An image that serves without
network access must hold that cache.

Serve the checkpoint on a CUDA GPU:

```bash
uv run --locked python -m positronic.vendors.abc_dit.server multitask_200k \
    --model.abc_python=<abc>/.venv/bin/python --model.checkpoint=<abc>/cache/abc_dit_xl_200k_model.pt
```

`--model.fast_inference=False` runs the DiT in float32 with no compilation. That mode also runs on a CPU, slowly.
