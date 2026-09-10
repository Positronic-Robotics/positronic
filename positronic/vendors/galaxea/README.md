# Galaxea G0.5-DROID in Positronic

**For internal, non-commercial evaluation only.** This integration is not intended
for production, customer demonstrations, or hosted services. Galaxea's model code,
weights, and associated materials are governed by [LICENSE-G0.5](LICENSE-G0.5),
including its Authorized Hardware and Authorized Image conditions for company
evaluation. Confirm that your environment qualifies with Galaxea before running
the model. See [NOTICE](NOTICE). The adapter does not grant additional model rights.

## Scope

Pretrained DROID/Franka inference through Positronic's standard remote policy API.
The backend calls Galaxea's `PolicyInferencer` and returns the **entire predicted
chunk in one response**. It does not use the upstream step cache or its
`action_steps` setting. The checkpoint determines the prediction length; the
adapter neither truncates nor repeats steps. GPU and real-robot evaluation are
required to establish performance on a particular rig.

The server-side [codec](codecs.py) handles all embodiment conversion:

| Direction | Conversion |
| --- | --- |
| Observation | Exterior + wrist RGB from HWC to CHW, zero dummy wrist view, 7 joint angles, instruction, 15 Hz |
| Gripper state | Positronic `0=open, 1=closed` → Galaxea `1=open, 0=closed`, using `1 - grip` |
| Arm action | Absolute joint targets in radians → `JointPosition` with DROID impedance gains |
| Gripper action | Galaxea `1=open, 0=closed` → Positronic `0=open, 1=closed`, using `clip(1 - predicted_grip, 0, 1)` |
| Missing gripper prediction | Emit no new gripper command; the driver retains its last target |
| Missing arm / malformed prediction | Fail the request |

Galaxea's DROID dataset loader flips both recorded gripper state and action with
`1 - x` before training. Its evaluation client applies the same conversion when
sending observations and receiving predictions. The model therefore represents
openness, while Positronic and DROID's robot interface represent closure.
See upstream [`DroidLerobotDataset._slice_meta_feature`](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/src/g05/data/droid/droid_lerobot_dataset.py#L319).

Galaxea's processor performs image resizing, state normalization, and action
denormalization. There is no gripper conversion on the robot client. Every step
receives its 15 Hz timestamp and the standard end-of-chunk timestamp. Positronic's
`ChunkedSchedule` executes the complete chunk before asking for a new prediction.
The backend has no per-episode action cache; closing or cancelling a session
cannot carry cached actions into another episode.

## Docker setup

The dedicated `positro/galaxea` image contains Galaxea's Python 3.10 environment
and a separate Positronic Python 3.13 environment. Galaxea is pinned to
`89f2322b4ad016e192437adc1a2c253b05bab246`. Its [inference dependencies](requirements-inference.txt)
are constrained to the versions in that revision's lockfile. PyTorch supplies CUDA
12.8 runtime libraries; the image uses an Ubuntu base and omits Galaxea's training
and simulation packages. Positronic installs its frozen lockfile without extras.
Model weights are mounted separately.
Inference requires a CUDA GPU; Galaxea's DROID guide estimates about 12 GB of free
GPU memory. RoboLab also needs GPU memory and graphics support.

Build from the Positronic repository root:

```bash
make -C docker build-galaxea
```

Galaxea's image targets are opt-in; aggregate image publishing does not include
this evaluation-only vendor.

### Checkpoint access for all users

Positronic users outside the company download directly from
[OpenGalaxea/G05](https://huggingface.co/OpenGalaxea/G05) using their own Hugging Face
account. No Positronic S3 credentials are needed. Hugging Face grants gated model
access [to individual users](https://huggingface.co/docs/hub/models-gated), so each
user must accept Galaxea's conditions and obtain access for their account.
That access remains subject to the model license, including the hardware and
image conditions above.

On the machine holding the Docker bind-mounted cache, log in and check the account:

```bash
uvx hf auth login
uvx hf auth whoami
```

If downloading returns `403` with a request to enable public gated repositories,
edit the active token in [Hugging Face settings](https://huggingface.co/settings/tokens)
and enable **Read access to contents of all public gated repos you can access**.
The account must also have access to G05; the token permission alone does not grant it.

Download the pinned DROID checkpoint and shared resources (about 12 GB):

```bash
uvx hf download OpenGalaxea/G05 \
  --revision e312be81e90c56a55bcb26b57429bd39a335b449 \
  --local-dir "$HOME/.cache/galaxea/checkpoints" \
  --include 'g05-droid/*' \
  --include 'action_tokenizer.pt' \
  --include 'qwen3_5_2b_base_processor/*' \
  --include 'licenses/*'
```

### Private company cache

The same checkpoint bundle is stored at
`s3://checkpoints/droid/galaxea/g05-droid/` for internal evaluation. Its
`provenance.json` records the Hugging Face revision and SHA-256 checksum of each
upstream file; `NOTICE` and `licenses/` carry the usage terms. This cache is not
an external download service. Under the Internal Evaluation PoC terms, company
access is restricted to the corporate group; outside users obtain their own
copies from Galaxea through the flow above.

With company S3 credentials, populate the same Docker cache:

```bash
aws s3 sync s3://checkpoints/droid/galaxea/g05-droid/ \
  "$HOME/.cache/galaxea/checkpoints/" \
  --endpoint-url https://storage.eu-north1.nebius.cloud
```

### Start the server

Keep `.hydra/config.yaml`, `dataset_stats.json`, and `checkpoints/model_state_dict.pt`
inside `checkpoints/g05-droid/`. Start the server on the GPU host:

```bash
IMAGE_TAG=local docker compose -f docker/docker-compose.yml \
  run --rm --service-ports galaxea-server --port=8000
```

With a remote Docker context, set `CACHE_ROOT` to the cache owner's home directory
on that host. At startup the server loads `g05-droid` by launching
[backend.py](backend.py) on the container's private `127.0.0.1:9000` endpoint.
The HTTP API, including `/api/v1/models`, becomes available after the model is ready.
Unloading the policy stops the child process. Each request runs fresh inference and returns
the full chunk. The first request can include model compilation latency; set
`--pipeline.source.infer_timeout=300` if needed.

The `droid` pipeline places the vendor codec after the remote boundary, so existing
clients use `.remote` without Galaxea dependencies. For a source installation,
create Galaxea's `.venv` with its locked dependencies. Link or copy the complete
downloaded checkpoint bundle to `<galaxea_root>/checkpoints`. The upstream config
resolves the shared action tokenizer and processor files from this location. Pass
`--pipeline.source.galaxea_root=/path/to/GalaxeaVLA` and
`--pipeline.source.checkpoint_path=/path/to/GalaxeaVLA/checkpoints/g05-droid/checkpoints/model_state_dict.pt` to
`uv run --locked python -m positronic.vendors.galaxea.server` in the Positronic environment.

Use the existing DROID evaluation configuration:

```bash
uv run --locked positronic eval run --eval=.real.droid.pick_place \
  --policy=.remote --policy.url=localhost:8000 \
  --output_dir=/path/to/evaluation-recordings
```

For RoboLab, run its existing image on a GPU host with RTX graphics support.
Use the Galaxea host's address, reachable from the evaluation container, and a
unique output directory for each run:

```bash
IMAGE_TAG=latest docker compose -f docker/docker-compose.yml run --rm robolab-eval \
  --eval=.sim.robolab.banana_in_bowl --eval.trial_count=1 \
  --policy=.remote --policy.url=<galaxea-host>:8000 \
  --output_dir=s3://inference/tmp/galaxea-robolab/<run-id>/
```

The private backend protocol is `galaxea-full-chunk-v1`; connecting this adapter to
Galaxea's stock step-serving endpoint raises a protocol error. Transport timeouts
close the connection, and an in-flight result cancelled by the caller is discarded.
Reconnect after a transport failure. Fine-tuning and dataset conversion are outside
this vendor's scope.

## Validation

```bash
uv run --locked pytest positronic/vendors/galaxea/tests positronic/tests/test_vendor_boundary.py
```

CPU tests cover full-chunk conversion, the live WebSocket adapter, action timing,
gripper conventions, omitted predictions, cancellation, and subprocess ownership.
They do not establish GPU compatibility or robot performance.

Upstream references: [DROID deployment](https://github.com/OpenGalaxea/GalaxeaVLA/tree/89f2322b4ad016e192437adc1a2c253b05bab246/experiments/droid),
[model weights](https://huggingface.co/OpenGalaxea/G05),
[inferencer](https://github.com/OpenGalaxea/GalaxeaVLA/blob/89f2322b4ad016e192437adc1a2c253b05bab246/src/g05/models/g05/inferencer.py).
