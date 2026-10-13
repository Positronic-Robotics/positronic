# Enter the Physical AI Competition

The [competition site](https://physical-ai-competition.positronic.ro/index.html) has the rules, the dates
and the leaderboard.

A coding agent can do most of the work. You approve the sign-in, and you choose where to push your image.

## What you need

You can install the commands, register and submit on any computer. To build and test your image, use a
Linux machine with an x86-64 (amd64) processor, such as a [Nebius](https://nebius.com) virtual machine.

- The platform runs only amd64 images. On a Mac with Apple silicon, or on another ARM machine, Docker
  emulates amd64. With emulation, the test is very slow, and it can fail.
- The test loads your model into memory. Give the machine enough memory and free disk space for your
  model. For the π0.5 DROID example image, use 32 GB of memory and 100 GB of free disk space.
- The first test needs no GPU. A GPU is necessary for [Test the image on a GPU](#test-the-image-on-a-gpu),
  and to train your policy.

## Use a coding agent

Paste this prompt into Claude Code, Codex or a similar agent:

```prompt
Help me enter the Nebius × Positronic Physical AI Competition.
Follow this guide step by step:
https://github.com/Positronic-Robotics/positronic/blob/main/docs/nebius-competition.md

1. Install the commands and register me. Ask me for a team name first.
   Show me the sign-in code and link, and wait while I approve it.
   Keep the API key in my environment. Never print it again.
2. Help me build a policy image, and push it to a container registry
   that I choose.
3. Test the image with the network denied.
4. Submit it, pinned by digest, and check its status until the scores
   arrive.
```

The agent needs git, [uv](https://docs.astral.sh/uv/) 0.8.5 or later, Docker and push access to a
container registry, such as Docker Hub. Run the agent on the machine that builds and tests the image, as
[What you need](#what-you-need) says.

## Get started by hand

### Install the commands

The commands need [uv](https://docs.astral.sh/uv/) 0.8.5 or later, the first version with
`--with-executables-from`. Run `uv --version` to check. To update uv, run `uv self update`, or use the
tool that installed uv.

This command installs the commands from the positronic repository on GitHub. It uses about 2 GB of disk
space.

```bash
uv tool install --with-executables-from positronic-platform-client \
    git+https://github.com/Positronic-Robotics/positronic
```

To update the commands later, run `uv tool upgrade positronic`.

### Register

Choose a team name. The command prints a code and a link. Open the link, sign in with GitHub, and enter
the code. The command then prints your API key one time.

```bash
platform-register --alias="<team name>"
export POSITRONIC_PLATFORM_API_KEY=<the key it printed>
```

- Keep the key secret. If you lose it, run `platform-register --rotate` to get a new key.
- When you register, you accept the [terms](https://physical-ai-competition.positronic.ro/terms.html) of
  the competition.

### Build and push your image

Your image runs an inference server for your policy. [Your policy image](#your-policy-image) says what the
platform needs from the image, and how to build one from an example.

### Test the image

Start the image with the network denied, and wait for its server. This test needs no GPU, and it uses
no submission. It needs an amd64 machine with enough memory, as [What you need](#what-you-need) says.
`--pull=always` makes Docker test the image in your registry, not an older local copy.

```bash
probe='import sys, urllib.error, urllib.request as u
try:
    print(u.urlopen(u.Request("http://127.0.0.1:8000/api/v1/keepalive", method="POST",
          headers={"Authorization": "Bearer test"}), timeout=5).read().decode())
except urllib.error.HTTPError as e:
    print("the server answered HTTP", e.code)
except OSError as e:
    reason = e.reason if isinstance(e, urllib.error.URLError) else e
    if not isinstance(reason, (ConnectionError, TimeoutError)):
        raise
    sys.exit(3)'
arch=$(docker version -f '{{.Server.Arch}}')
if [ "$arch" != amd64 ]; then
  echo "Docker runs on '$arch'. Run this test on an amd64 machine."
elif cid=$(docker run -d --pull=always --network none -e AUTH_TOKEN=test docker.io/<you>/<image>:v1); then
  deadline=$((SECONDS + 900))
  while :; do
    [ "$(docker inspect -f '{{.State.Running}}' "$cid")" = true ] || { echo "the server exited"; break; }
    [ "$SECONDS" -lt "$deadline" ] || { echo "no answer in 900 s"; break; }
    docker exec "$cid" /positronic/.venv/bin/python -c "$probe"
    [ $? = 3 ] || break
    sleep 5
  done
  docker logs --tail 30 "$cid"
  docker rm -f "$cid"
fi
```

The loop calls the keepalive route every 5 s, for up to 900 s. A model loads more slowly on a CPU than on
a GPU. The loop ends with one of these lines:

- `Docker runs on '<arch>'`: Docker does not run amd64 images natively, and the test does not start. Use
  an amd64 machine, as [What you need](#what-you-need) says. An empty `<arch>` means that Docker does not
  answer.
- `{"alive_seconds": ...}`: the server loaded its weights, and it serves on the CPU.
- `the server exited`: read the log. A correct image loads its weights, and then it stops on the missing
  GPU.
- `the server answered HTTP <code>`: the server runs, but it refused the request. A `401` means that the
  server does not accept the token in `AUTH_TOKEN`. On the platform, this fails the run with
  `policy_setup_crash`.
- `no answer in 900 s`: the server hung. Read the log for the step where it stopped.
- An error from `docker exec` or a Python traceback: the probe failed, and the loop stopped at once. For
  example, the image has no Python at `/positronic/.venv/bin/python`.

The probe exits with code 3 when the port is closed or the server does not answer in 5 s. The loop then
waits quietly, because a server that loads its weights does not open the port yet.

The log must not show a DNS error. The `docker exec` line uses the Python of the example images. In your
own image, use any HTTP client that the image has. The probe runs inside the container, so it also gets
an answer from a server that listens only on `127.0.0.1`. The platform connects from outside the
container, so such a server fails with `policy_setup_crash`. The full test on a GPU machine is in
[Test the image on a GPU](#test-the-image-on-a-gpu).

### Submit

Read the name of the competition eval, and the digest of your image:

```bash
positronic eval catalog
docker buildx imagetools inspect docker.io/<you>/<image>:v1
```

Submit the image to the competition eval, pinned by its digest. Then follow the submission:

```bash
positronic eval run --eval=<eval> \
    --policy-image=docker.io/<you>/<image>@sha256:<digest> \
    --transaction-key=<a name for this attempt>
positronic eval status --id=<submission id>
positronic eval list
```

- Pin the image by its digest. A tag can move to an image that you did not test.
- Reuse the transaction key only to retry the same submission. The same key returns the same submission,
  and it does not count again. A new image or another eval needs a new key: the platform refuses a changed
  request under a used key with `transaction_conflict`.
- `positronic eval catalog` also prints a `tasks: forbidden` line. That line is correct for a participant
  key.
- You can make one submission per day. [Submissions per day](#submissions-per-day) says what counts.
- [Read the outcome](#read-the-outcome) says what the status tells you.

### Choose your final entry

Before submissions close, choose your final entry, as the
[rules](https://physical-ai-competition.positronic.ro/rules.html#finale) say.

### Ask for help

Ask your questions on the [Positronic Discord](https://discord.gg/PXvBy4NBgv). The commands above call
the platform's [HTTP API](https://platform.positronic.ro/docs), and your own tools can call it too.

## Train on Nebius

To train on Nebius, sign up for Nebius Builder with the email address of the GitHub account that you
register with.

---

<!-- The competition site shows the sections above this line in full, and each section below it as a block
that opens and closes. -->

## Your policy image

The platform pulls your image and starts it on a GPU beside the simulator. Then it plays the competition
eval against the server in the image.

### What the platform runs

- The server listens on port 8000 on all interfaces (`0.0.0.0`), and speaks the positronic
  [session protocol](https://github.com/Positronic-Robotics/positronic/blob/main/positronic/offboard/README.md).
- The platform starts the image with no arguments. Your `ENTRYPOINT` or `CMD` starts the server.
- The platform sets one environment variable, `AUTH_TOKEN`, the bearer token of the run. It passes no
  flags and no secrets.
- The container has no network access at any time. Put every package, weight and tokenizer into the
  image.
- The platform waits until `POST /api/v1/keepalive` answers on port 8000. Then it opens one WebSocket
  session for each episode at `/api/v1/session`.
- If a route refuses the token of the run, the run fails with `policy_setup_crash`.
- The GPU is one `3g.40gb` slice of an NVIDIA H100, with 40448 MiB of memory.
- The platform runs the image digest that it found when you submitted. It pulls that digest from your
  registry when the run starts, and it keeps no copy of the image.
- The [rules](https://physical-ai-competition.positronic.ro/rules.html#limits) give the other limits on
  the image.

### Start from an example image

Each example image serves a public DROID checkpoint, such as π0.5 DROID or GR00T N1.7 DROID. CI publishes
each one to Docker Hub as a `positro/` image, from a recipe in the positronic repository.
[Example images](https://github.com/Positronic-Robotics/positronic/blob/main/docs/submit-a-policy-image.md#example-images)
lists them, with the recipe for each.

- The comments in each recipe say where a checkpoint of your own goes.
- The GR00T recipe needs a Hugging Face read token, because the repository of its backbone is gated.
- To write a server for a model of your own, read
  [Connect your model](https://github.com/Positronic-Robotics/positronic/blob/main/docs/connect-your-model.md#implement-your-own-server).
- [Three traps in the `positro/*` bases](https://github.com/Positronic-Robotics/positronic/blob/main/docs/submit-a-policy-image.md#three-traps-in-the-positro-bases)
  are failures that show only when the network is denied.

### Build and push

The recipes build from the root of the positronic source. The header of each recipe gives its build
command. For example:

```bash
git clone https://github.com/Positronic-Robotics/positronic
cd positronic
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
    -f docker/Dockerfile.serve-pi05-droid -t docker.io/<you>/<image>:v1 --push .
```

- `--platform linux/amd64` names the architecture that the platform runs.
- `--provenance=false --sbom=false` makes buildx push one image manifest, with no attestation beside it.
- Push to Docker Hub when your base is a `positro/*` image. Then only your own layers upload, because the
  base layers come from the public repository.
- A public repository needs nothing more. A private repository needs a credential, as
  [A private image](#a-private-image) shows.

### A private image

Push to a private repository, and give the platform a read-only credential for it. First write the
password into a file of its own. Run these commands in your own terminal, paste the token at the prompt,
and press Enter:

```bash
mkdir -p ~/.config/positronic
install -m 600 /dev/null ~/.config/positronic/registry-password
printf 'Registry token: '; IFS= read -rs token; echo
printf %s "$token" > ~/.config/positronic/registry-password; unset token
```

`read -s` keeps the token off the screen and out of a terminal or agent transcript.

Then add `--registry-username=<user>` and `--registry-password-file=~/.config/positronic/registry-password`
to `positronic eval run`.

- Give the credential read access to that one repository. Docker Hub calls this type of token an access
  token.
- The password stays in the file. It is not on the command line, where other processes and the shell
  history can read it.
- The command removes one line ending from the end of the file. All other characters are the password.

What the platform does with your credential:

- When you submit, the platform reads your image with the credential. If your registry refuses it, the
  submission fails with `image_unpullable`, and the submission counts.
- The platform keeps the credential as a secret for that one submission. No API response and no log
  shows the password.
- When the run starts, the machine that runs your image logs in to your registry with the credential. It
  pulls the image, and then it logs out. Your container never gets the credential.
- The platform deletes the credential when the submission ends. The secret store keeps a deleted secret
  until it purges it.
- Only the services that start the run, the machine that runs your image and the platform administrators
  can read the credential.
- Keep the image and the credential valid until the submission ends. Then you can revoke the token.
- A retry with the same `--transaction-key` returns the first submission, also with a new credential. To
  use a new credential, use a new transaction key.

## Test the image on a GPU

On a machine with a GPU, serve the image with the network denied. Then wait until the keepalive route
answers, from inside the container, with the token:

```bash
docker network create --internal noegress
probe='import sys, urllib.error, urllib.request as u
try:
    print(u.urlopen(u.Request("http://127.0.0.1:8000/api/v1/keepalive", method="POST",
          headers={"Authorization": "Bearer test"}), timeout=5).read().decode())
except urllib.error.HTTPError as e:
    print("the server answered HTTP", e.code)
except OSError as e:
    reason = e.reason if isinstance(e, urllib.error.URLError) else e
    if not isinstance(reason, (ConnectionError, TimeoutError)):
        raise
    sys.exit(3)'
arch=$(docker version -f '{{.Server.Arch}}')
if [ "$arch" != amd64 ]; then
  echo "Docker runs on '$arch'. Run this test on an amd64 machine."
elif cid=$(docker run -d --pull=always --network noegress --gpus all -e AUTH_TOKEN=test docker.io/<you>/<image>:v1); then
  deadline=$((SECONDS + 600))
  while :; do
    [ "$(docker inspect -f '{{.State.Running}}' "$cid")" = true ] || { echo "the server exited"; break; }
    [ "$SECONDS" -lt "$deadline" ] || { echo "no answer in 600 s"; break; }
    docker exec "$cid" /positronic/.venv/bin/python -c "$probe"
    [ $? = 3 ] || break
    sleep 5
  done
  docker logs --tail 30 "$cid"
  docker rm -f "$cid"
fi
```

- `Docker runs on '<arch>'` means that the machine is not amd64. The test does not start.
- The loop calls the route every 5 s, for up to 600 s. A slow model binds port 8000 late: FLUX 3 Action
  binds it after about 140 s.
- The route answers `{"alive_seconds": ...}` when the model has loaded and warmed.
- `the server exited` means that the server crashed. The log shows why.
- `the server answered HTTP <code>` means that the server refused the request. A `401` means that it does
  not accept the token in `AUTH_TOKEN`.
- `no answer in 600 s` means that the server hung, or that it starts too slowly. On the platform, machine
  boot, the image pull and the server's start share one deadline of 1800 s.
- An error from `docker exec` or a Python traceback means that the probe failed. The loop stops at once.
- A server that checks the token answers `401` when the request has no token.
- The `docker exec` line uses the Python of the example images. In your own image, use any HTTP client
  that the image has.

After you push the image, read its compressed size. The size limit counts the layers and the config, as
this command adds them. The command needs [jq](https://jqlang.org/download/).

```bash
docker buildx imagetools inspect --raw docker.io/<you>/<image>:v1 \
  | jq '([.layers[].size] | add) + .config.size'
```

## Submissions per day

You can make one submission per day. The count starts again at 00:00 UTC each day, so you can submit at
23:00 UTC and again at 01:00 UTC on the next day.

- A submission counts from the time that the platform accepts it. Your cancel never gives it back, also
  when you cancel before the run starts.
- When the platform cannot start a run for a submission and ends the submission itself, it gives the
  submission back. That submission ends `errored` with `internal_error`.
- A submission that fails also counts. If the fault is the platform's, the organizers give the submission
  back.
- A retry with the same `--transaction-key` returns the first submission, and it does not count again.
- The platform refuses a submission over the limit with `quota_exceeded`.
- The competition can change the limit during the season. It announces each change on the
  [competition site](https://physical-ai-competition.positronic.ro/index.html).

## Read the outcome

A submission goes from `pending` to `running`, and then to `finished`, `errored` or `cancelled`. Before
it ends, a submission can also show `blocked`. A `blocked` submission waits on what its `reason` names,
and the platform changes its status later. `positronic eval status` prints the `reason`.

A `running` submission reports its `stage`:

| stage | meaning |
|---|---|
| `provisioning` | the machines start and pull your image |
| `evaluating` | your server answered on port 8000, and the episodes run |
| `persisting` | the run writes its files |

When a submission gets to `evaluating`, your image works. A lost simulator machine is replaced, so a run
can go back from `evaluating` to `provisioning`. An image submission reports `episodes` as `0/0` and
`runs` as an empty list, because those fields do not apply to it.

A `finished` submission has `scores.primary`, the value that the leaderboard ranks. It also has
`artifacts`, with signed links to the files of the run. An `errored` submission has `artifacts` only if
the run wrote its files. If the run failed before that, the submission has no links, and `reason_code`
and `reason` tell you why it failed.

| link | content | when it is there |
|---|---|---|
| `result` | the run, the attempt that scored, and `scores` with `primary`, `episodes` and `unscored` | in all `artifacts` |
| `policy_log` | the stdout and stderr of your container | when your container printed something |
| `diagnostics` | why the run failed, and the state of the machine | on a failed run, when the run wrote it |

The links expire after 15 minutes. Run `positronic eval status` again to get new links. On a failed run,
read `policy_log` first, if it is there. In `diagnostics`, a `vram_peak_mib` near 25 means that the
container failed before the model got to the GPU. `container_oom_killed` tells you if the memory limit
stopped the container.

### Failure reasons

`reason_code` tells you why a run failed. These reasons are faults of the image, and they count against
your submissions:

| reason_code | first thing to check |
|---|---|
| `image_unpullable` | Is the image public, or does your credential read it? Is the digest correct? |
| `image_too_large` | the compressed size of the image, against the limit in the rules |
| `policy_setup_crash` | `policy_log`: the server did not start, or it refused the token of the run |
| `policy_inference_crash` | `policy_log`: the server stopped after it started to serve |
| `policy_oom` | `diagnostics.container_oom_killed`, and the size of the model against the memory limits |
| `latency_budget_exceeded` | the inference time for each step |
| `wall_clock_exceeded` | the run went past its time limit |

The reasons `provision_wedged`, `runner_unresponsive`, `internal_error` and `quota_exceeded` are faults of
the platform. As a `reason_code`, `quota_exceeded` means that the platform had no capacity for the run.
Tell the organizers about a fault of the platform on the [Positronic Discord](https://discord.gg/PXvBy4NBgv).
