# Submit a policy image

The platform pulls a container image that serves your model, runs it against a named eval in
simulation, and scores it. This page is the path from a model you hold to a scored run.

Most of the path exists. Every vendor server in `positronic/vendors/<vendor>/server.py` speaks the
[session protocol](../positronic/offboard/README.md), and every `positro/<vendor>` image on Docker
Hub carries the vendor stack and the positronic tree. What you add is the last mile: the weights,
the offline environment, `EXPOSE 8000`, and a start command. Two recipes ship in `docker/`:

| Model | Recipe | Base image | What it serves |
|---|---|---|---|
| openpi π0.5 DROID | [`docker/Dockerfile.submit-openpi`](../docker/Dockerfile.submit-openpi) | `positro/openpi` | `pi05_droid_jointpos`, the public checkpoint |
| GR00T N1.7 DROID | [`docker/Dockerfile.submit-gr00t`](../docker/Dockerfile.submit-gr00t) | `positro/gr00t` | `nvidia/GR00T-N1.7-DROID` |

For a fine-tuned checkpoint of one of these families, copy the recipe and replace the weights step.
For a model of another family, see [Other models](#other-models).

## What the platform does with your image

1. It resolves your image reference to a digest at submission and records it as
   `policy_image_digest`. The run uses those bytes.
2. It refuses an image whose compressed size, config and layers summed, is over 30 GB
   (`image_too_large`), and one it cannot pull anonymously (`image_unpullable`). Both are
   charged to your quota.
3. It runs the image on a GPU VM with **no arguments**. Your `CMD` or `ENTRYPOINT` starts the
   server. The platform passes no flags and no secrets. The one variable it sets is `AUTH_TOKEN`,
   the run's bearer token.
4. It denies all network egress from the container for the whole run. Only the simulator can
   reach your container, on port 8000.
5. It waits for `GET /api/v1/models` to answer on port 8000. VM boot, the image pull and your
   server's start share one provisioning deadline of 1800 s. A 25 GB image takes about 10 minutes
   to pull.
6. It opens one WebSocket session per episode at `/api/v1/session`, with the bearer token.
7. It fails the run if a route serves a caller without the token, or refuses the run's own token.
   The vendor servers read `AUTH_TOKEN` and check it; a server of your own must do the same.
8. The GPU is one `3g.40gb` slice of an H100: 40448 MiB of VRAM.

## The requirements

| # | Requirement | What happens if you miss it |
|---|---|---|
| 1 | The image starts the server itself: `CMD` plus `EXPOSE 8000` | `policy_setup_crash` after `bash` exits |
| 2 | Every weight and tokenizer is in the image | a download at start hangs or fails |
| 3 | Nothing at start needs the network | `uv run` and Hugging Face both do, see below |
| 4 | The image is public, and pinned by digest when you submit | `image_unpullable`, or a run of bytes you did not test |
| 5 | The server honours `AUTH_TOKEN` | `policy_setup_crash` |
| 6 | The image is under 30 GB compressed | `image_too_large` |

Requirement 4 has a consequence: a gated checkpoint has to be baked into a public image. That is a
licence decision to make before you build.

## Two traps at start

Both are properties of the `positro/*` images, measured on `positro/openpi:latest` with the network
denied.

**`uv run` needs the network.** The `positro/<vendor>` images carry the positronic tree at
`/positronic` and no environment for it. `docker-compose.yml` starts every server with `uv run`,
which builds that environment at container start. With the network denied the container dies in
seconds on a DNS error. Build the environment in the image and call its interpreter:

```dockerfile
# Wrong: resolves the project and fetches an interpreter at every start.
CMD ["uv", "run", "--python", "3.13", "python", "-m", "positronic.vendors.openpi.server", "droid_jointpos"]

# Right: `uv sync` at build time, then the interpreter it made.
RUN uv sync --locked --python 3.13 --extra openpi && uv cache clean
CMD ["/positronic/.venv/bin/python", "-m", "positronic.vendors.openpi.server", "droid_jointpos"]
```

**Hugging Face asks the Hub about a local checkpoint.** `transformers` and `huggingface_hub` send a
HEAD request per file, each retried five times, about 23 s per file against a resolver that cannot
answer. Set the offline variables in the image:

```dockerfile
ENV HF_HOME=/opt/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

GR00T needs one more variable, `GROOT_PATCH_MISTRAL=1`, because `transformers` also asks the Hub
about the backbone's tokenizer with no cache fallback. `Dockerfile.submit-gr00t` sets all four. This was
measured in a container with the network denied. The offline variables alone fail at once with
`OfflineModeIsEnabled`. The patch alone times out after 600 s of retried HEAD requests. Both
together load the model in 151 s.

## Build the image

### openpi π0.5 DROID

```bash
docker buildx build --provenance=false --sbom=false \
    -f docker/Dockerfile.submit-openpi -t docker.io/<you>/pi05-droid:v1 --push docker
```

The recipe fetches the public `pi05_droid_jointpos` checkpoint and the PaliGemma tokenizer in a
stage of its own. It resolves the positronic environment with `--extra openpi`. It also resolves
openpi's own environment, because the server starts `serve_policy.py` through
`uv run --project /openpi`. Then it serves `droid_jointpos`. The checkpoint sits under a directory
named `0`: a local `checkpoints_dir` lists its subdirectories as model ids, and an id has to be
digits.

For a checkpoint of your own, replace the `assets` stage with a `COPY` of the checkpoint directory
into `/opt/positronic/checkpoints/<name>/0`. Name your pipeline in the `ENTRYPOINT`. The
[openpi guide](../positronic/vendors/openpi/README.md) lists the pipelines.

### GR00T N1.7 DROID

```bash
docker buildx build --secret id=hf_token,src=$HOME/.hf_token --provenance=false --sbom=false \
    -f docker/Dockerfile.submit-gr00t -t docker.io/<you>/gr00t-droid:v1 --push docker
```

The recipe resolves the positronic environment, then downloads `nvidia/GR00T-N1.7-DROID` (6.9 GB)
and the backbone `nvidia/Cosmos-Reason2-2B` (4.9 GB) into `HF_HOME`. The backbone repository is
gated: accept NVIDIA's terms on its Hub page, then put a read token in `$HOME/.hf_token`. The token
enters no layer. The image serves the `droid` pipeline, which is the base checkpoint.

For a fine-tuned checkpoint, `COPY` its `checkpoint-<step>` directories into the image and add
`--pipeline.source.model_source=<their parent>` to the `CMD`. The base checkpoint's backbone is
still the one the recipe bakes. Loading the model needs about 15 GB of CPU RAM before anything
reaches the GPU.

### Other models

- **DreamZero.** The public `GEAR-Dreams/DreamZero-DROID` checkpoint is 65 GB on the Hub, and
  the `positro/dreamzero` base is 20 GB compressed. Together they do not fit the 30 GB budget.
  Serve DreamZero on your own GPU and file an eval plan with a `remote` endpoint instead
  ([Eval plans](../client/README.md#eval-plans)).
- **A model of your own.** Write a server that speaks the session protocol
  ([Connect your model](connect-your-model.md)), and hold the image to the requirements above.

### Push and check

- Push to Docker Hub when your base is `positro/*`. The base layers cross-mount from the public
  repository, so only your layers upload. Another registry re-uploads all of them.
- `--provenance=false --sbom=false` makes buildx push one image manifest. Without them it pushes a
  manifest index with an `unknown/unknown` attestation entry beside the image.
- Read the digest and the compressed size the way the platform does, anonymously:

```bash
TOK=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:<you>/<image>:pull" | jq -r .token)
curl -s -H "Authorization: Bearer $TOK" \
  -H "Accept: application/vnd.oci.image.manifest.v1+json" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
  "https://registry-1.docker.io/v2/<you>/<image>/manifests/v1" \
  | jq '{digest: .config.digest, compressed_bytes: ([.layers[].size] | add)}'
```

A `401` or a `404` here is what the platform sees too: the image is not public, or the name is
wrong.

## Test the image before you submit

Run it with the network denied. This reproduces the platform's own conditions and needs no GPU:

```bash
docker run --rm --network none -e AUTH_TOKEN=test docker.io/<you>/<image>:v1
```

What a correct image prints: the server pins its checkpoint, starts the model process, reads the
weights from the image, and then fails on the missing GPU. For the openpi recipe that is jax on
CPU reading the checkpoint under `/opt/positronic/checkpoints`. For the GR00T recipe it is
`Flash Attention 2 is not available on CPU`. Everything you control is then correct. What you must
not see: `NameResolutionError`, `dns error`, `OfflineModeIsEnabled`, or a hang. One line to
ignore: albumentations prints a `UserWarning` about fetching its version, which is harmless.

On a machine with a GPU, serve it with the network denied and dial the models route from inside
the container with the token:

```bash
docker network create --internal noegress
docker run -d --name policy --network noegress --gpus all -e AUTH_TOKEN=test docker.io/<you>/<image>:v1
docker exec policy /positronic/.venv/bin/python -c "import urllib.request as u; \
  print(u.urlopen(u.Request('http://127.0.0.1:8000/api/v1/models', headers={'Authorization': 'Bearer test'})).read())"
```

The route answers `{"models": [...]}` with the token and `401` without it.

## Register and submit

The client installs from the repository and puts `platform-register` on your path:

```bash
uv add "positronic-platform-client @ git+https://github.com/Positronic-Robotics/positronic@main#subdirectory=client"
uv run platform-register --alias="<display name>"      # GitHub's device flow; prints the key once
export POSITRONIC_PLATFORM_API_KEY=<the key it printed>
```

The commands that drive an eval ship with `positronic`, so run them from a checkout:

```bash
uv run positronic eval catalog                          # the evals your key may name
uv run positronic eval run --eval=<eval> \
    --policy-image=docker.io/<you>/<image>@sha256:<digest> \
    --transaction-key=<a name for this attempt>
uv run positronic eval status --id=<hex id>
uv run positronic eval list
```

- Pin by digest. A tag is resolved at submission, so a tag can name bytes you did not test.
- Reuse the transaction key on a retry. The same key returns the original submission; a retry
  without one spends quota again. The same key with a different request is refused as a conflict.
- The catalog offers `molmo.franka_pick_mini` (20 episodes) and `molmo.franka_pick_mini_smoke`
  (5 episodes) today. Read the names from the catalog; both provision the same GPU VMs. Run the
  smoke eval first: it answers whether the image serves at all.
- `users.me` reports your quota. The default is 2 image submissions per day. An eval plan the
  lab rig runs does not count against it.

## Read the result

The lifecycle is `pending -> running -> finished | errored | cancelled`. A `blocked` run waits on
what its `reason` names. A `running` run reports its `stage`:

| stage | meaning |
|---|---|
| `provisioning` | the VMs boot and pull your image |
| `evaluating` | your server answered on port 8000 and episodes run |
| `persisting` | the run writes its files |

Reaching `evaluating` is the proof that the image works. A run can go back to `provisioning` from
`evaluating`: a lost simulator VM is replaced, and `result.json` records which attempt scored.
`episodes` and `runs` on a submission are the lab rig's fields. An image run reports `0/0` and an
empty list, while it runs and after it finishes.

A `finished` submission carries `scores.primary`, the value a board ranks on, and `artifacts`
with signed links:

| link | content | present |
|---|---|---|
| `result` | the run id, the attempt that scored, and `scores` with a `per_task` breakdown | on a finished or an errored run |
| `policy_log` | your container's stdout and stderr, from the attempt that decided the run | on a finished or an errored run, when the container printed anything |
| `diagnostics` | why the run failed, and the state of the box | on an errored run whose record was written |

The links expire after 15 minutes. Read `submissions.get` again for fresh ones. `submissions.get`
returns these three files. The episode recordings stay in the platform's storage.

`policy_log` is the first thing to read on a failed run. `diagnostics` answers the questions the
log cannot:

```json
{
  "reason_code": "policy_setup_crash",
  "reason": "the policy container exited with 25 MiB of its 40448 MiB slice in use, before it served",
  "egress_probe": "denied",
  "pull": "pulled",
  "container_started": true,
  "container_exit_code": 1,
  "container_oom_killed": false,
  "vram_peak_mib": 25,
  "vram_capacity_mib": 40448,
  "serving": false,
  "served_on_boot": 0,
  "token_rejected": false,
  "serving_unauthenticated": false,
  "startup_log": "..."
}
```

Two readings that save time:

- `vram_peak_mib` near 25 means the container never reached the GPU. The failure is in startup,
  before the model loads. Read `policy_log` for the exception.
- `container_oom_killed: false` with a low `vram_peak_mib` rules out both memory limits at once.

### Failure reasons

`reason_code` is a closed set, split by fault. A caller fault is charged to your quota; a platform
fault is not.

| reason_code | fault | first thing to check |
|---|---|---|
| `image_unpullable` | caller | is the image public, and is the digest right? |
| `image_too_large` | caller | the compressed size, against 30 GB |
| `policy_setup_crash` | caller | `policy_log`: the server did not come up, or served without the token |
| `policy_inference_crash` | caller | `policy_log`: the server died after it served |
| `policy_oom` | caller | `diagnostics.container_oom_killed`; the model against the 40448 MiB slice |
| `latency_budget_exceeded` | caller | inference time per step |
| `wall_clock_exceeded` | caller | the run passed its ceiling |
| `invalid_flags` | caller | the plan the platform recorded |
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
