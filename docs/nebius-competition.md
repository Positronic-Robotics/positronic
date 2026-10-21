# Enter the Physical AI Competition

The [competition site](https://competition.phail.ai/) has the rules, the dates
and the leaderboard.

A coding agent can do most of the work. You approve the sign-in, and you choose where to push your image.

Six steps: 1 [Install](#install-the-commands) · 2 [Register](#register) ·
3 [Enter](#enter-the-competition) · 4 [Build](#build-and-push-your-image) · 5 [Test](#test-the-image) ·
6 [Submit](#submit)

## Before you start

- A machine to build and test on: Linux on amd64, with enough memory and free disk space for your model.
  [What the example images use](https://github.com/Positronic-Robotics/positronic/blob/main/docs/submit-a-policy-image.md#what-the-example-images-use)
  gives values for each example image. Some images also need a GPU.
  - No such machine? [Build and test on a Nebius VM](#build-and-test-on-a-nebius-vm) creates one with one
    command. Sign up for Nebius Builder with the email address of the account that you register with.
    Nebius also has the GPUs to train your policy.
  - Do not use a Mac with Apple silicon, or another ARM machine. Docker emulates amd64 there, and the test
    is very slow and can fail.
- On that machine: git, [uv](https://docs.astral.sh/uv/) 0.8.5 or later, and Docker.
- Push access to a container registry, such as Docker Hub.

You can install the commands, register, enter and submit on any computer. Do these four steps on the same
computer, because the API key stays on the computer that registered. Build and test, and run your coding
agent, on the build machine.

## Use a coding agent

Paste this prompt into Claude Code, Codex or a similar agent:

```prompt
Help me enter the Nebius × Positronic Physical AI Competition.
Follow this guide step by step:
https://github.com/Positronic-Robotics/positronic/blob/main/docs/nebius-competition.md

1. Install the commands and register me. Ask me for a team name first.
   Run the register command in the background, and show me its
   sign-in line at once. The link expires in 5 minutes.
   Wait while I sign in.
   The command saves my API key on this computer. Never print the key.
2. Enter me in the competition. Show me what becomes public and
   where to read how my data is used, and ask whether to enter.
   Never enter without explicit human approval.
3. Help me build a policy image, and push it to a container registry
   that I choose.
4. Test the image with the network denied.
5. Before the first submission, show me the terms link and ask
   whether I accept the terms for my team. Never accept the terms
   without explicit human approval.
6. Submit it, pinned by digest, and check its status until the scores
   arrive.
```

## Get started by hand

### Install the commands

The commands need [uv](https://docs.astral.sh/uv/) 0.8.5 or later. Run `uv --version` to check. To update
uv, run `uv self update`, or use the tool that installed uv.

This command installs `positronic` from the `main` branch of its repository. It uses about 2.5 GB of disk
space.

```bash
uv tool install --managed-python --python 3.13 'positronic @ git+https://github.com/Positronic-Robotics/positronic'
```

`--managed-python --python 3.13` makes uv install Python 3.13 and use it. The Python of Ubuntu cannot
build one of the dependencies, and `positronic` does not run on Python 3.14 or later.

To update the commands later, run `uv tool upgrade positronic`.

### Register

Choose a team name. The command prints a sign-in link and a code, and waits while you sign in. The link
expires in 5 minutes. If it expires, run the command again. Open the link and sign in. The link carries the
code, so the page shows it: make sure that it is the code that the command printed. If the page asks for
the code, enter it. The sign-in page shows the ways to sign in that the platform takes. The command saves
your API key on this computer, and prints the path of the file. The other commands read the key from that
file.

```bash
positronic account register --alias="<team name>"
```

- Keep the key file secret. If you lose it, run `positronic account register --rotate` to get a new key.
  This stops the old key.
- If your account already has a key, the command does not show the key again. Set
  `POSITRONIC_PLATFORM_API_KEY` to the key that you have.

### Enter the competition

The command shows what becomes public about your team. With `--yes`, your team enters. The entry asks
for no terms: your team accepts the [terms](https://competition.phail.ai/terms) with its first submission
(see [Submit](#submit)). The section "Your data" of the terms tells how the organizers use your data.

```bash
competition-enter
competition-enter --yes
```

- When you enter, the competition publishes your team name and the date you entered. When a run is
  scored, it publishes the scores and the videos of the run. All of this stays public after the
  competition.
- The platform refuses a competition submission from an account that has not entered.

### Build and push your image

Your image runs an inference server for your policy. Start from an example image: each one serves a public
DROID checkpoint, and its recipe says where your own checkpoint goes.
[Your policy image](#your-policy-image) lists the examples, and says what the platform needs from an
image.

The recipes build from the root of the positronic source. The header of each recipe gives its build
command. For example, for π0.5 DROID:

```bash
git clone https://github.com/Positronic-Robotics/positronic
cd positronic
docker login
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
    -f docker/Dockerfile.serve-pi05-droid -t docker.io/<you>/<image>:v1 --push .
```

- `--platform linux/amd64` names the architecture that the platform runs.
- `--provenance=false --sbom=false` makes buildx push one image manifest, with no attestation beside it.
- Push to Docker Hub when your base is a `positro/*` image. Then only your own layers upload, because the
  base layers come from the public repository.
- A public repository needs nothing more. A private repository needs a credential, as
  [A private image](#a-private-image) shows.

After the push, read the compressed size of the image. The size limit counts the layers and the config, as
this command adds them. The command needs [jq](https://jqlang.org/download/).

```bash
docker buildx imagetools inspect --raw docker.io/<you>/<image>:v1 \
  | jq '([.layers[].size] | add) + .config.size'
```

### Test the image

Start the image as the platform starts it, with the network denied, and wait for its server. The test uses
no submission. Run it on the build machine that [Before you start](#before-you-start) describes, in the
positronic source:

```bash
bash docker/test_policy_image.sh docker.io/<you>/<image>:v1
```

The script pulls the image from your registry, so it tests what the platform pulls. It calls the keepalive
route from outside the container every 5 s, for up to 900 s, because a model loads more slowly on a CPU
than on a GPU. Then it prints one of these lines, and the end of the log:

- `the server answered: {"alive_seconds": ...}`: the server loaded its weights. The test passed.
- `the server exited`: read the log. An image that needs a GPU stops with an error about CUDA. The GR00T
  N1.7, FLUX 3 Action and Cosmos3-Nano example images stop in this way. Test such an image on a GPU.
- `the server answered HTTP <code>`: the server runs, but it refused the request. A `401` means that the
  server does not accept the token in `AUTH_TOKEN`. On the platform, this fails the run with
  `policy_setup_crash`.
- `the server answered 200 without {"alive_seconds": ...}`: the keepalive route must answer a JSON object
  whose `alive_seconds` is an integer or `null`. On the platform, any other answer fails the run.
- `the server listens only on 127.0.0.1`: the platform connects from outside the container, so the run
  fails with `policy_setup_crash`. Make the server listen on `0.0.0.0`.
- `no answer in 900 s`: the server hung. Read the log for the step where it stopped.

The log must not show a DNS error.

On a machine with a GPU, add `--gpu`. The script then gives the server 600 s:

```bash
bash docker/test_policy_image.sh --gpu docker.io/<you>/<image>:v1
```

- The platform gives the image one GPU with 40 GB of memory. On a larger GPU, use `nvidia-smi` to make
  sure that your model uses less.
- On the platform, machine boot, the image pull and the server's start share one deadline of 1800 s.

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

- The platform refuses the first submission of your team until your team accepts the terms. Read the
  [terms](https://competition.phail.ai/terms), then run `competition-enter --accept-terms=<version>` with
  the version that you read, and submit again. The platform refuses a version that is not current, and it
  records the version and the time.
  A team that accepted the terms submits again with no new acceptance.
- Pin the image by its digest. A tag can move to an image that you did not test.
- Reuse the transaction key only to retry the same submission. The same key returns the same submission,
  and it does not count again. A new image or another eval needs a new key: the platform refuses a changed
  request under a used key with `transaction_conflict`.
- You can make one submission per day. [Submissions per day](#submissions-per-day) says what counts.
- [Read the outcome](#read-the-outcome) says what the status tells you.

### Choose your final entry

Before submissions close, choose your final entry, as the
[rules](https://competition.phail.ai/rules#finale) say.

### Ask for help

Ask your questions on the [Positronic Discord](https://discord.gg/PXvBy4NBgv). The commands above call
the platform's [HTTP API](https://platform.positronic.ro/docs), and your own tools can call it too.

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
- The [rules](https://competition.phail.ai/rules#limits) give the other limits on
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

## Build and test on a Nebius VM

[`docker/nebius_vm.sh`](https://github.com/Positronic-Robotics/positronic/blob/main/docker/nebius_vm.sh)
creates a Nebius virtual machine for the build and the tests, and deletes it. The VM has 32 GB of memory,
a 200 GB disk, one NVIDIA L40S GPU with 48 GB of GPU memory, Docker with GPU access, and uv.

Before you start:

- Sign up for Nebius Builder, as [Before you start](#before-you-start) says.
- Install the [Nebius CLI](https://docs.nebius.com/cli/install), and run `nebius profile create`. The
  script creates the VM in the project of the profile.
- If you have no SSH key, make one with `ssh-keygen -t ed25519`.

```bash
git clone https://github.com/Positronic-Robotics/positronic
bash positronic/docker/nebius_vm.sh create
```

When the VM is ready, the command prints `ssh ubuntu@<address>`. Connect to the VM, and do the steps of
this guide on it. Run your coding agent on the VM too.

When you finish, delete the VM. Nebius charges for the VM until you delete it, also when it does nothing.

```bash
bash positronic/docker/nebius_vm.sh delete
```

- [Nebius prices](https://nebius.com/prices) give the cost of each VM type.
- For a VM with no GPU, set `NEBIUS_PLATFORM=cpu-d3 NEBIUS_PRESET=8vcpu-32gb`. The header of the script
  gives all the settings.
- If Nebius has no free L40S, the command stops with `NotEnoughResources`, and it deletes the VM. Try
  again later, or set `NEBIUS_PLATFORM` and `NEBIUS_PRESET` to another GPU type.
- `nebius compute platform list` shows the VM types and presets in your region.
- If Nebius refuses the VM because of a [quota](https://docs.nebius.com/overview/quotas), ask Nebius
  for a larger quota.

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
  [competition site](https://competition.phail.ai/).

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
