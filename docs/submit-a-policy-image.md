# Submit a policy image

Positronic runs your policy for you. You provide a Docker image that runs an inference server: an
HTTP server on port 8000 that speaks the positronic
[session protocol](../positronic/offboard/README.md). The platform pulls the image, starts it on a
GPU beside a simulator, plays an eval against it, and scores the result. An eval is a named set of
tasks from the platform's catalog.

The container gets no network access. Bake every package, weight and tokenizer into the image, and
test that the image starts with the network denied before you submit:
`docker run --rm --network none <image>`.

## Example images

Every `positro/<vendor>` image on Docker Hub carries a vendor stack and an inference server for it.
Each recipe below adds the weights, an offline environment, `EXPOSE 8000` and a start command:

| Model | Recipe | Base | Serves |
|---|---|---|---|
| openpi π0.5 DROID | [`docker/Dockerfile.submit-openpi`](../docker/Dockerfile.submit-openpi) | `positro/openpi` | `pi05_droid_jointpos`, the public checkpoint |
| GR00T N1.7 DROID | [`docker/Dockerfile.submit-gr00t`](../docker/Dockerfile.submit-gr00t) | `positro/gr00t` | `nvidia/GR00T-N1.7-DROID` |

The header of each recipe gives its build command. The comments in each recipe say where a
checkpoint of your own goes and which flag names it. Loading GR00T needs about 15 GB of CPU RAM
before anything reaches the GPU.

The GR00T recipe downloads `nvidia/GR00T-N1.7-DROID` (6.9 GB) and its backbone
`nvidia/Cosmos-Reason2-2B` (4.9 GB). The backbone repository is gated: accept NVIDIA's terms on its
Hub page, then put a read token in `$HOME/.hf_token`. The build reads the token through a secret
mount, and it enters no layer.

Other models:

- **DreamZero.** The public `GEAR-Dreams/DreamZero-DROID` checkpoint is 65 GB on the Hub, and the
  `positro/dreamzero` base is 20 GB compressed. Together they exceed the 30 GB budget. Serve
  DreamZero on your own GPU and file an eval plan with a `remote` endpoint
  ([Eval plans](../client/README.md#eval-plans)).
- **A model of your own.** Write an inference server ([Connect your model](connect-your-model.md))
  and hold the image to the rules in [Life of a submission](#life-of-a-submission).

### Two traps in the `positro/*` bases

Both recipes handle both traps. Both were measured on `positro/openpi:latest` with the network
denied.

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

### Build and push

- `--platform linux/amd64` names the architecture the platform runs. Both `positro/*` bases
  publish amd64 only, so a build on an ARM host resolves nothing without it.
- `--provenance=false --sbom=false` makes buildx push one image manifest. Without them it pushes a
  manifest index with an `unknown/unknown` attestation entry beside the image.
- Push to Docker Hub when your base is `positro/*`. The base layers cross-mount from the public
  repository, so only your layers upload. Another registry re-uploads all of them.
- The image must be public. Under that rule a gated checkpoint goes into a public image, which is
  a licence decision to make before you build.

## Life of a submission

1. The platform resolves your image reference to a digest at submission and records it as
   `policy_image_digest`. The run uses those bytes.
2. It refuses an image it cannot pull anonymously (`image_unpullable`), and one whose compressed
   size, config and layers summed, is over 30 GB (`image_too_large`). Both are charged to your
   quota.
3. It runs the image on a GPU VM with **no arguments**. Your `CMD` or `ENTRYPOINT` starts the
   server. The platform passes no flags and no secrets. It sets one variable, `AUTH_TOKEN`, the
   run's bearer token. An image with no start command runs the base image's `CMD ["bash"]`, which
   exits, and the run fails with `policy_setup_crash`.
4. It denies all network egress from the container for the whole run. Only the simulator can
   reach your container, on port 8000. A download at start hangs or fails.
5. It waits for `GET /api/v1/models` to answer on port 8000. VM boot, the image pull and your
   server's start share one provisioning deadline of 1800 s. A 25 GB image takes about 10 minutes
   to pull.
6. It opens one WebSocket session per episode at `/api/v1/session`, with the bearer token.
7. It fails the run with `policy_setup_crash` if a route serves a caller without the token, or
   refuses the run's own token. The vendor servers read `AUTH_TOKEN` and check it; a server of
   your own must do the same.
8. The GPU is one `3g.40gb` slice of an H100: 40448 MiB of VRAM.

## Test the image before you submit

Run it with the network denied. This reproduces the platform's own conditions and needs no GPU:

```bash
docker run --rm --network none -e AUTH_TOKEN=test docker.io/<you>/<image>:v1
```

In a correct image, the server pins its checkpoint, starts the model process, reads the weights
from the image, and then fails on the missing GPU. For the openpi recipe that is jax on CPU
reading the checkpoint under `/opt/positronic/checkpoints`. For the GR00T recipe it is
`Flash Attention 2 is not available on CPU`. Everything you control is then correct. The run must
not print `NameResolutionError`, `dns error` or `OfflineModeIsEnabled`, and it must not hang.
albumentations prints a `UserWarning` about fetching its version; ignore it.

On a machine with a GPU, serve it with the network denied and dial the models route from inside
the container with the token:

```bash
docker network create --internal noegress
docker run -d --name policy --network noegress --gpus all -e AUTH_TOKEN=test docker.io/<you>/<image>:v1
docker exec policy /positronic/.venv/bin/python -c "import urllib.request as u; \
  print(u.urlopen(u.Request('http://127.0.0.1:8000/api/v1/models', headers={'Authorization': 'Bearer test'})).read())"
```

The route answers `{"models": [...]}` with the token and `401` without it.

After the push, read the digest and the compressed size the way the platform does, anonymously:

```bash
docker/read_image_digest.sh <you>/<image>:v1
```

[`docker/read_image_digest.sh`](../docker/read_image_digest.sh) prints the `docker-content-digest`
header, which names the manifest the registry served. Pin that digest. The `config.digest` inside
the manifest names the config blob, and the registry refuses a reference to it. The size adds the layers and the config blob, which is the count the platform makes against
the 30 GB budget. The platform sees the same `401` or `404`: the image is not public, or the name
is wrong.

## Submit

The commands ship with `positronic`, and a checkout of this repository carries the platform
client at the version it pins. Run them from the checkout:

```bash
uv run platform-register --alias="<display name>"      # GitHub's device flow; prints the key once
export POSITRONIC_PLATFORM_API_KEY=<the key it printed>
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
- `positronic eval catalog` prints the evals your key may name, and the tasks each one runs. The
  catalog changes, so read the names from it. Start with the smallest eval it offers: it answers
  whether the image serves at all.
- `users.me` reports your quota. The default is 2 image submissions per day. An
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
| `result` | the run id, the attempt that scored, and `scores` with a `per_task` breakdown | on a finished or an errored run |
| `policy_log` | your container's stdout and stderr, from the attempt that decided the run | on a finished or an errored run, when the container printed anything |
| `diagnostics` | why the run failed, and the state of the box | on an errored run whose record was written |

The links expire after 15 minutes. Read `submissions.get` again for fresh ones. `submissions.get`
returns these three files. The episodes are a separate call, below.

Read `policy_log` first on a failed run. `diagnostics` answers the questions the
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
  episodes are `attempts/<n>/episodes/`. The same attempt holds `scores.json`, `logs/` and
  `timing.jsonl`. `control/` is never listed.
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
| `image_unpullable` | caller | is the image public, and is the digest right? |
| `image_too_large` | caller | the compressed size, against 30 GB |
| `policy_setup_crash` | caller | `policy_log`: the server did not come up, or served without the token |
| `policy_inference_crash` | caller | `policy_log`: the server died after it served |
| `policy_oom` | caller | `diagnostics.container_oom_killed`; the model against the 40448 MiB slice |
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
