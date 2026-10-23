# Submit a policy image

Positronic runs your policy for you. You provide a Docker image that runs an inference server: an
HTTP server on port 8000 that speaks the positronic
[session protocol](../positronic/offboard/README.md). The platform pulls the image, starts it on a
GPU beside a simulator, plays an eval against it, and scores the result. An eval is a named set of
tasks from the platform's catalog.

The container gets no network access. Bake every package, weight and tokenizer into the image, and
test that the image starts with the network denied before you submit
([Test the image before you submit](#test-the-image-before-you-submit)).

## Example images

The openpi and GR00T recipes build on a `positro/<vendor>-base` image on Docker Hub, which carries
the vendor stack. The MolmoAct2 recipe builds on a Python 3.13 image with uv: MolmoAct2 has no
vendor stack. The FLUX 3 Action and Cosmos3-Nano recipes build the vendor stack themselves. Each
recipe adds the weights, the positronic source with an offline environment, `EXPOSE 8000` and a
start command. The layers go from the least often changed to the most: base, dependencies, source,
weights. The source and the weights come in through `COPY --link` with no step after them, so a
source change builds one new layer and reuses the rest from a cache. CI publishes all five images
to Docker Hub. Build from the root of a positronic checkout:

| Model | Recipe | Base | Serves |
|---|---|---|---|
| openpi π0.5 DROID | [`docker/Dockerfile.serve-pi05-droid`](../docker/Dockerfile.serve-pi05-droid), published as `positro/pi05-droid` | `positro/openpi-base` | `pi05_droid_jointpos`, the public checkpoint |
| GR00T N1.7 DROID | [`docker/Dockerfile.serve-gr00t-n17-droid`](../docker/Dockerfile.serve-gr00t-n17-droid), published as `positro/gr00t-n17-droid` | `positro/gr00t-base` | `nvidia/GR00T-N1.7-DROID` at a pinned revision |
| MolmoAct2 DROID | [`docker/Dockerfile.serve-molmoact2-droid`](../docker/Dockerfile.serve-molmoact2-droid), published as `positro/molmoact2-droid` | `ghcr.io/astral-sh/uv:python3.13-bookworm` | `allenai/MolmoAct2-DROID` at a pinned revision |
| FLUX 3 Action DROID | [`docker/Dockerfile.serve-flux3-action`](../docker/Dockerfile.serve-flux3-action), published as `positro/flux3-action` | `python:3.12-slim-bookworm` | `black-forest-labs/flux-3-action-droid`, `variants/gd`; see [FLUX 3 Action](#flux-3-action) |
| Cosmos3-Nano DROID | [`docker/Dockerfile.serve-cosmos3-nano`](../docker/Dockerfile.serve-cosmos3-nano), published as `positro/cosmos3-nano` | `nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04` | `nvidia/Cosmos3-Nano-Policy-DROID`; see [Cosmos3-Nano](#cosmos3-nano) |

CI builds each image when its recipe or the code it installs changes, and tags each build `main`, `latest`
and the commit. A release also tags each of them `v<version>`. CI does not test an
image: run the checks in [Test the image before you submit](#test-the-image-before-you-submit).
Read the digest of a published image with `docker/read_image_digest.sh positro/<image>:main`, and
pin it.

The header of each recipe gives its build command. The comments in each recipe say where a
checkpoint of your own goes and how the server is pointed at it. Loading GR00T needs about 15 GB of CPU RAM
before anything reaches the GPU.

The GR00T recipe downloads `nvidia/GR00T-N1.7-DROID` (6.9 GB) and its backbone
`nvidia/Cosmos-Reason2-2B` (4.9 GB). The backbone repository is gated: accept NVIDIA's terms on its
Hub page, then put a read token in `$HOME/.hf_token`, or in `HF_TOKEN` for
`make -C docker build-serve-gr00t-n17-droid`. The build reads the token through a secret mount, and
it enters no layer.

Both GR00T models are under the NVIDIA Open Model License. `positro/gr00t-n17-droid` is Built on
NVIDIA Cosmos. The image carries the agreement and its notice in `/opt/gr00t`, as the licence
requires of each copy of the models.

The MolmoAct2 recipe downloads `allenai/MolmoAct2-DROID` (21.8 GB in float32). The build needs no
token: the repository is not gated. The image is 24.4 GB compressed.

Other models:

- **DreamZero.** The public `GEAR-Dreams/DreamZero-DROID` checkpoint is 65 GB on the Hub, and the
  `positro/dreamzero` base is 20 GB compressed. Together they exceed the 50 GB budget. Serve
  DreamZero on your own GPU and file an eval plan with a `remote` endpoint
  ([Eval plans](../client/README.md#eval-plans)).
- **A model of your own.** Write an inference server ([Connect your model](connect-your-model.md))
  and hold the image to the rules in [Life of a submission](#life-of-a-submission).

### What the example images use

Approximate values, from one run of each image in October 2026. A later build of an image can use
different values. The platform gives each image 24 GB of GPU memory, so the last two images do not fit.

| Image | Disk, pulled image | Memory, test without a GPU | GPU memory |
|---|---|---|---|
| `positro/pi05-droid` | 50 GB | 20 GB | 10 GB |
| `positro/gr00t-n17-droid` | 40 GB | needs a GPU | 10 GB |
| `positro/molmoact2-droid` | 35 GB | 12 GB | 15 GB |
| `positro/flux3-action` | 40 GB | needs a GPU | 35 GB, over the 24 GB |
| `positro/cosmos3-nano` | 55 GB | needs a GPU | 35 GB, over the 24 GB |

### Three traps in the `positro/*` bases

**`uv run` needs the network.** The `positro/<vendor>` images carry the positronic tree at
`/positronic` and no environment for it. The repository's `docker/docker-compose.yml` starts every
server with `uv run`, which builds that environment at container start. With the network denied
the container dies in seconds on a DNS error. Each recipe runs
`uv sync --locked --python 3.13 --no-dev` at build time and starts the interpreter the sync made,
`/positronic/.venv/bin/python`.

**Hugging Face asks the Hub about a local checkpoint.** `transformers` and `huggingface_hub` send
a HEAD request per file, each retried five times, about 23 s per file against a resolver that
cannot answer. Each recipe sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. GR00T needs one
more variable, `GROOT_PATCH_MISTRAL=1`, because `transformers` also asks the Hub about the
backbone's tokenizer with no cache fallback. The offline variables alone fail at once with
`OfflineModeIsEnabled`. The patch alone times out after 600 s of retried HEAD requests. Both
together load the model in 151 s.

**openpi resolves the container's hostname.** openpi's `scripts/serve_policy.py` calls
`gethostbyname` on the hostname before it serves. Under `--network none`, Docker writes no record
of the hostname into `/etc/hosts`, and the server dies with `socket.gaierror: Temporary failure in
name resolution`. A Docker network with no egress has the record, so only `--network none` shows
the trap. The openpi recipe installs `libnss-myhostname`, which resolves the hostname to a local
address.

### FLUX 3 Action

`positro/flux3-action` holds FLUX 3 Action DROID, the serving code of Black Forest Labs (BFL), and
every weight the server loads. It serves the session protocol, and the platform runs the published image as a
submission.

- The image holds two Python environments. positronic's server answers on port 8000. It starts BFL's
  own server in BFL's environment, on port 9000 inside the container, and sends it each observation.
- The image serves the `droid` command. It sends the policy the wrist view and `image.exterior` in
  both exterior slots, so it runs on every eval. An eval with two exterior views, such as RoboLab or
  a DROID rig, runs better on `droid_3cam`: to select it in your own image, change `droid` to
  `droid_3cam` in the recipe's `ENTRYPOINT`.
- BFL's server compiles the model for about 140 s, and port 8000 binds after it. The GPU memory peaks
  at about 33 000 MiB.
- `black-forest-labs/flux-3-action-droid` and `black-forest-labs/flux-3-action-base` are gated. Accept
  BFL's terms on the Hub page of each, then put a read token in `$HOME/.hf_token`, or in `HF_TOKEN` for
  `make -C docker build-serve-flux3-action`. The build reads the token through a secret mount, and it
  enters no layer.

### Cosmos3-Nano

`positro/cosmos3-nano` holds Cosmos3-Nano-Policy-DROID, NVIDIA's action server with its environment,
and the Wan2.2 VAE the model loads. It serves the session protocol, and the platform runs the
published image as a submission.

- The image holds two Python environments. positronic's server answers on port 8000. It starts
  NVIDIA's server in NVIDIA's environment, on port 9000 inside the container, and sends it each
  observation.
- The image serves the `droid` command. It sends the policy the wrist view and `image.exterior` in
  both exterior slots, so it runs on every eval. An eval with two exterior views, such as RoboLab or
  a DROID rig, runs better on `droid_3cam`: to select it in your own image, change `droid` to
  `droid_3cam` in the recipe's `ENTRYPOINT`.
- The client plays each chunk of 32 actions in full, and a gripper value above 0.5 closes the
  gripper, as in NVIDIA's RoboLab client.
- The guardrails are off. Their model repository is gated, and they serve only the video and text
  generation paths, which the action server does not use.
- The GPU memory peaks at about 32 500 MiB.

### Build and push

- `--platform linux/amd64` names the architecture the platform runs. Both `positro/*` bases
  publish amd64 only, so a build on an ARM host resolves nothing without it.
- `--provenance=false --sbom=false` makes buildx push one image manifest. Without them it pushes a
  manifest index with an `unknown/unknown` attestation entry beside the image.
- Push to Docker Hub when your base is `positro/*`. The base layers cross-mount from the public
  repository, so only your layers upload. Another registry re-uploads all of them.
- A public repository needs nothing else. A private one takes a credential, below.

### A private image

Push to a private repository and hand the platform a read-only credential for it. The run pulls
the image from your registry with that credential.

Write the password into a file of its own and name the file:

```bash
mkdir -p ~/.config/positronic
install -m 600 /dev/null ~/.config/positronic/registry-password
printf 'Registry token: '; IFS= read -rs token; echo        # paste the token at the prompt
printf %s "$token" > ~/.config/positronic/registry-password; unset token
uv run positronic eval run --eval=<eval> \
    --policy-image=<registry>/<you>/<image>@sha256:<digest> \
    --registry-username=<user> \
    --registry-password-file=~/.config/positronic/registry-password
```

- The password is in the file and never on the command line, where every `ps` on the machine
  reads it and every shell history keeps it. `read -s` keeps it off the screen and out of a
  terminal or agent transcript, so run it in your own terminal.
- One trailing line ending goes, so a file written with `echo` needs no trimming. Everything
  else in the file is the password, a leading or trailing space included.
- Give the credential read on that one repository. A registry issues a token for exactly this:
  Docker Hub calls it an access token, and a cloud registry a service account key.

What the platform does with the credential:

1. At submission, the platform reads the image with it. A credential your registry refuses fails
   the submission with `image_unpullable`, charged.
2. After that read, the platform stores the credential as one secret for this submission. The
   submission record holds a reference to the secret. No response and no log holds the password.
3. When the run starts, the platform copies the credential into the run's own secret. The policy
   VM of that run reads it before it denies the container the network.
4. The VM writes the credential to a file that only root reads, and logs in to your registry. It
   deletes the file, pulls the digest, and logs out. The policy container never gets the credential.
5. When the submission ends, the platform deletes both secrets. The secret store keeps a deleted
   secret until it purges it.

- Only the platform's run services, the policy VM of that run and a platform administrator can
  read the credential. They can read it until the purge.
- Keep the image and the credential valid until the submission ends. A run that cannot pull
  the image fails with `image_unpullable`. Revoke the token after the submission ends.
- A retry with the same transaction key returns the first submission, also with a changed
  credential. To submit with a new credential, use a new transaction key.

## Life of a submission

1. The platform resolves your image reference to a digest at submission and records it as
   `policy_image_digest`. The run pulls that digest from your registry when its VM starts. The
   platform keeps no copy of the image.
2. It refuses an image it cannot pull — anonymously, or with the credential you named
   (`image_unpullable`) — and one whose compressed size, config and layers summed, is over 50 GB
   (`image_too_large`). Both are charged to your quota. The image has two more budgets, in
   [The container](#the-container).
3. It runs the image on a GPU VM with **no arguments**. Your `CMD` or `ENTRYPOINT` starts the
   server. The platform passes no flags and no secrets. It sets one variable, `AUTH_TOKEN`, the
   run's bearer token. An image with no start command runs the base image's `CMD ["bash"]`, which
   exits, and the run fails with `policy_setup_crash`.
4. It denies all network egress from the container for the whole run. Only the simulator can
   reach your container, on port 8000. A download at start hangs or fails.
5. It waits for `POST /api/v1/keepalive` to answer on port 8000. VM boot, the image pull and your
   server's start share one provisioning deadline of 1800 s. A 25 GB image takes about 10 minutes
   to pull.
6. It opens one WebSocket session per episode at `/api/v1/session`, with the bearer token.
7. It fails the run with `policy_setup_crash` if a route refuses the run's own token. The vendor
   servers read `AUTH_TOKEN` and check it. A server of your own may check it or ignore it.
8. The GPU is one `1g.24gb` slice of an RTX PRO 6000: 24192 MiB of VRAM and a quarter of the card's
   compute. The container runs under the
   limits in [The container](#the-container).

The RTX PRO 6000 is a Blackwell card with compute capability 12.0 (`sm_120`). Each framework in the
image needs kernels for `sm_120`. Without them, the first inference fails with
`CUDA error: no kernel image is available for execution on the device`. The PyPI build of torch 2.7.1
is a CUDA 12.6 build, and its kernels stop at `sm_90`. Install a PyTorch build for CUDA 12.8 or newer,
for example from the `https://download.pytorch.org/whl/cu128` index, as the MolmoAct2 recipe does. A
CUDA 12.8 build needs NVIDIA driver 570 or newer. `docker/test_policy_image.sh` runs no inference, so
check the kernels on a machine with a GPU:

```bash
docker run --rm --gpus device=0 --entrypoint python docker.io/<you>/<image>:v1 \
  -c "import torch; print(torch.cuda.get_arch_list())"
```

The list must contain `sm_120`. If `python` in your image is not the interpreter your server uses,
give the full path to that interpreter.

## The container

[`client/platform_client/policy_container.py`](../client/platform_client/policy_container.py)
holds each value below. The platform runs every image under them.

| Limit | Value | Constant |
|---|---|---|
| The port the server listens on | 8000 | `POLICY_PORT` |
| The one variable the platform sets | `AUTH_TOKEN` | `AUTH_TOKEN_ENV` |
| The deadline for VM boot, the pull and the server's start | 1800 s | `PROVISIONING_DEADLINE_S` |
| The GPU slice | `1g.24gb` | `MIG_PROFILE` |
| VRAM | 24192 MiB | `VRAM_MIB` |
| The compressed image | 50 GB | `COMPRESSED_IMAGE_BYTES` |
| The unpacked image | 55 GB | `UNPACKED_IMAGE_BYTES` |
| The files in the image | 3,300,000 | `IMAGE_FILES` |
| The block an unpacked entry is counted in | 4096 bytes | `IMAGE_BLOCK_BYTES` |
| The image store | 120 GiB | `IMAGE_STORE_BYTES` |
| Memory, with no swap | 150 GiB | `MEMORY_BYTES` |
| CPUs | 14 | `CPUS` |
| Processes and threads | 4,096 | `PIDS` |
| The container log | 200 MiB | `LOG_BYTES` |
| Disk reads | 500 MiB/s | `DISK_READ_BYTES_PER_S` |
| Disk writes | 100 MiB/s | `DISK_WRITE_BYTES_PER_S` |
| Disk read operations | 2,000/s | `DISK_READ_IOPS` |
| Disk write operations | 1,000/s | `DISK_WRITE_IOPS` |
| The send rate | 500 Mbit/s | `SEND_BITS_PER_S` |
| The send burst | 5 MiB | `SEND_BURST_BYTES` |

- The container has no Linux capabilities and runs with `no-new-privileges`. It has no route out,
  and a name lookup gets no answer. `docker_limit_flags(disk)` gives the `docker run` flags that set
  the limits. The send rate is a policer on the host, and a packet above it is dropped.
- The compressed size is the config blob and every layer, as the registry stores them
  (`ImageManifest.compressed_size`).
- The unpacked size counts every entry of every layer: its size rounded up to a whole 4096-byte
  block, and at least one block. The file count counts every entry of every layer: files,
  directories, links and deletions. A file that a later layer replaces or deletes counts in each
  layer that carries it.
- The image store is an ext4 filesystem. Its metadata and the 5% that ext4 keeps for root leave
  119.78 GB that any writer can use. It holds the compressed layers beside the unpacked ones, the
  writable layer of the container and its log, so an image inside both budgets leaves the writable
  layer about 14.5 GB. A process that runs as root in the container can also use the 6 GiB reserve.
- An image over a budget fails with `image_too_large`, charged. A container over its memory is
  killed, and the run fails with `policy_oom`. The other limits slow the server, or stop it with
  `policy_setup_crash` or `policy_inference_crash`.

## Test the image before you submit

Run it with the network denied, and wait for its server. This reproduces the platform's own
conditions and needs no GPU. It needs an amd64 machine: on an ARM machine, such as a Mac with Apple
silicon, Docker emulates amd64, and the model does not load in the time limit. The script does not
start the test there.

```bash
bash docker/test_policy_image.sh docker.io/<you>/<image>:v1
```

[`docker/test_policy_image.sh`](../docker/test_policy_image.sh) starts the image on a Docker network with
no egress, with no capabilities, and with the token `test` in `AUTH_TOKEN`. Every 5 s, it calls the
keepalive route from outside the container, as the platform does. It waits up to 900 s, because a model
loads more slowly on a CPU. A ready server prints `the server answered: {"alive_seconds": ...}`. A server
that stops prints `the server exited`, and a hung one prints `no answer in 900 s`. A server that refuses
the request prints `the server answered HTTP <code>`, and a `401` means that it does not accept the token
in `AUTH_TOKEN`. A server that listens only on `127.0.0.1` cannot answer the platform, and the script says
so. The log follows each of them.

In a correct image, the server pins its checkpoint and starts the model process. The openpi and
MolmoAct2 recipes then load the model on the CPU and serve. The GR00T recipe stops on the missing GPU
with `Flash Attention 2 is not available on CPU`. For the FLUX 3 Action recipe, BFL's server exits on
the missing GPU, and the server reports `FLUX 3 Action backend exited with code 1`. For the Cosmos3-Nano
recipe, NVIDIA's server exits the same way, and the server reports `Cosmos3 backend exited with code 1`.
These three stop before they read the weights, so for them, also run the test on a GPU below. The log
must not show `NameResolutionError`, `dns error` or `OfflineModeIsEnabled`.
albumentations prints a `UserWarning` about fetching its version; ignore it. MolmoAct2 prints
`A new version of the following files was downloaded` when `transformers` copies the model's code
into its module cache; no download occurs.

On a machine with a GPU, add `--gpu`:

```bash
bash docker/test_policy_image.sh --gpu docker.io/<you>/<image>:v1
```

The script then waits up to 600 s: FLUX 3 Action binds port 8000 after about 140 s, and GR00T loads in
about 151 s. `no answer in 600 s` means a hung server, or one too slow for the provisioning deadline,
which it shares with VM boot and the pull.

The platform gives the image one GPU with 24 GB of memory. On a larger GPU, use `nvidia-smi` to make
sure that your model uses less. A memory setting that is a fraction of the GPU, such as
`XLA_PYTHON_CLIENT_MEM_FRACTION`, is a fraction of the 24 GB.

After the push of a public image, read the digest and the compressed size the way the platform
does, anonymously:

```bash
docker/read_image_digest.sh <you>/<image>:v1
```

[`docker/read_image_digest.sh`](../docker/read_image_digest.sh) prints the `docker-content-digest`
header, which names the manifest the registry served. Pin that digest. The `config.digest` inside
the manifest names the config blob, and the registry refuses a reference to it. The size adds the layers and the config blob, which is the count the platform makes against
the 50 GB budget. The platform sees the same `401` or `404`: the image is not public, or the name
is wrong.

The script reads a public image only. For a private image, log in with the credential you give
the platform, and read the same two values with it:

```bash
docker login <registry> --username=<user> --password-stdin < ~/.config/positronic/registry-password
docker buildx imagetools inspect <registry>/<you>/<image>:v1                  # the digest
docker buildx imagetools inspect --raw <registry>/<you>/<image>:v1 \
  | jq '([.layers[].size] | add) + .config.size'                             # the compressed size
```

The platform sees the same `401` or `404`: the credential does not read the image, or the name is
wrong.

## Submit

The commands ship with `positronic`, and a checkout of this repository carries the platform
client at the version it pins. Run them from the checkout:

```bash
uv run platform-register --alias="<display name>"      # a device sign-in; prints the key once
export POSITRONIC_PLATFORM_API_KEY=<the key it printed>
uv run positronic eval catalog                          # the evals your key may name
uv run positronic eval run --eval=<eval> \
    --policy-image=docker.io/<you>/<image>@sha256:<digest> \
    --transaction-key=<a name for this attempt>
uv run positronic eval status --id=<hex id>
uv run positronic eval list
```

- Pin by digest. A tag is resolved at submission, so a tag can name bytes you did not test.
- A private image adds `--registry-username` and `--registry-password-file`; see
  [A private image](#a-private-image).
- Reuse the transaction key on a retry. The same key returns the original submission; a retry
  without one spends quota again. The same key with a different request is refused as a conflict.
- `positronic eval catalog` prints the evals your key may name, and the tasks each one runs. The
  catalog changes, so read the names from it. Start with the smallest eval it offers: it answers
  whether the image serves at all.
- A competition submission needs an entry from your team:
  [Enter the competition](nebius-competition.md#enter-the-competition).
- `users.me` reports your quota. A competition submission counts against a daily limit:
  [Submissions per day](nebius-competition.md#submissions-per-day). An
  [eval plan](../client/README.md#eval-plans) runs on the lab rig, a real robot, and does not count
  against it.

## Read the outcome

The lifecycle is `pending -> running -> finished | errored | cancelled`. A `blocked` run waits on
what its `reason` names. A `running` run reports its `stage`:

| stage | meaning |
|---|---|
| `provisioning` | the VMs boot and pull your image |
| `evaluating` | your server answered on port 8000 and episodes run |
| `persisting` | the run writes its files |

Reaching `evaluating` proves the image works. A run can go back to `provisioning` from
`evaluating`: a lost simulator VM is replaced, and `result.json` records which attempt scored.
`episodes` and `runs` on a submission are fields of a plan the lab rig runs. An image run reports
`0/0` and an empty list, while it runs and after it finishes.

A `finished` submission carries `scores.primary`, the value a leaderboard ranks on, and
`artifacts` with signed links:

| link | content | present |
|---|---|---|
| `result` | the run id, the attempt that scored, and `scores` with `primary`, `episodes` and `unscored` | on a finished run, and on an errored run that wrote its records |
| `policy_log` | your container's stdout and stderr, from the attempt that decided the run | on the same runs as `result`, when the container printed anything |
| `diagnostics` | why the run failed, and the state of the box | on an errored run whose record was written |

The links expire after 15 minutes. Read `submissions.get` again for fresh ones. `submissions.get`
returns these three files. The episodes are a separate call, below.

Read `policy_log` first on a failed run. `diagnostics` answers the questions the
log cannot:

```json
{
  "reason_code": "policy_setup_crash",
  "reason": "the policy container exited with 25 MiB of its 24192 MiB slice in use, before it served",
  "egress_probe": "denied",
  "pull": "pulled",
  "container_started": true,
  "container_exit_code": 1,
  "container_oom_killed": false,
  "vram_peak_mib": 25,
  "vram_capacity_mib": 24192,
  "serving": false,
  "served_on_boot": 0,
  "token_rejected": false,
  "startup_log": "..."
}
```

- `vram_peak_mib` near 25 means the container never reached the GPU. The failure is in startup,
  before the model loads. Read `policy_log` for the exception.
- `container_oom_killed: false` with a low `vram_peak_mib` rules out both memory limits at once.

### The episodes

`submissions.artifacts` lists what a finished run wrote, one page at a time, and it is the only
route to an episode: a signed link covers one key, and the bucket refuses a listing. The
client calls it with `list_artifacts`, and `positronic eval` has no command for it.

- Read `result.json` first. Its `attempt_location` names the attempt that scored. A reprovisioned
  run writes one tree per attempt, and only that tree holds the episodes that count.
- The route's `prefix` filter is read under the submission's own prefix, so one attempt's
  episodes are `attempts/<n>/episodes/`. The same attempt holds `logs/` and `timing.jsonl`.
  `control/` is never listed.
- The route pages: pass `after` with the `next` of the page before it. These links expire after
  15 minutes, like the three above.
- `attempts/<n>/episodes/` is a positronic dataset root. Download the whole tree, and positronic
  reads it as a dataset, the same as one a local `positronic eval run` writes
  ([Evaluation](evaluation.md)). The eval's privileged ground-truth is stripped from it.
- A plan the lab rig ran is refused here: its episodes land under the prefix `submissions.get`
  names.

### Failure reasons

`reason_code` is a closed set, split by fault. A caller fault is charged to your quota; a platform
fault is not.

| reason_code | fault | first thing to check |
|---|---|---|
| `image_unpullable` | caller | is the image public, or does your credential read it? Is the digest right? |
| `image_too_large` | caller | the compressed size, against 50 GB |
| `policy_setup_crash` | caller | `policy_log`: the server did not come up, or refused the run's own token |
| `policy_inference_crash` | caller | `policy_log`: the server died after it served |
| `policy_oom` | caller | `diagnostics.container_oom_killed`; the model against the 24192 MiB slice |
| `latency_budget_exceeded` | caller | inference time per step |
| `wall_clock_exceeded` | caller | the run passed its ceiling |
| `invalid_flags` | caller | the plan the platform recorded |
| `quota_exceeded` | platform | the platform's own capacity; resubmit, then report it |
| `provision_wedged` | platform | resubmit with a new transaction key |
| `runner_unresponsive` | platform | resubmit |
| `internal_error` | platform | resubmit, then report it |

A `policy_setup_crash` whose `policy_log` ends in a readiness timeout of the model process is a
model that loaded too slowly. Check the offline variables above first.

## See also

- [Eval plans](../client/README.md#eval-plans) — compose your own tasks and endpoints; needs a
  customer grant.
- [Examples](../positronic/cli/examples/README.md) — the same flow from Python.
- [Evaluation](evaluation.md) — the evals, and the local run that needs no platform.
