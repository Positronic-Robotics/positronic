# Score a policy on MolmoSpaces on your machine

Run a public MolmoSpaces eval on your own machine, and get its score. The platform runs the same episodes
under the same name, and scores them with the same rule.

| Eval | Episodes | What it is for |
|---|---|---|
| `molmo.franka_pick_mini_smoke` | 5 | A first run, and a quick check of a change |
| `molmo.franka_pick_mini` | 20 | A score to compare policies on |

Both evals run the DROID Franka in MolmoSpaces scenes, one trial for each episode. The competition eval
scores other episodes, and its episodes are not public.

With no GPU of your own, [score on Nebius Serverless](#score-on-nebius-serverless-with-no-gpu-of-your-own).

## Before you start

- Linux on amd64, with an NVIDIA GPU for the policy server and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- git, [uv](https://docs.astral.sh/uv/) 0.8.5 or later, Docker, and a C compiler. On Ubuntu, install the
  compiler with `sudo apt install build-essential`. The install builds one dependency, `evdev`, from source.
- About 80 GB of free disk space: about 50 GB for the `positro/pi05-droid` image of step 3, 13 GB for the
  MolmoSpaces assets, about 2.5 GB for `positronic`, and about 7 GB for the simulator's Python environment.

## 1. Install the commands

```bash
uv tool install --managed-python --python 3.13 'positronic @ git+https://github.com/Positronic-Robotics/positronic'
```

`--managed-python --python 3.13` makes uv install Python 3.13 and use it. The Python of Ubuntu cannot
build one of the dependencies.

## 2. Choose where the MolmoSpaces assets go

The simulator reads its scenes, objects and robots from two directories. Set both, and make them:

```bash
export MLSPACES_ASSETS_DIR=~/molmospaces/assets
export MLSPACES_CACHE_DIR=~/molmospaces/cache
mkdir -p "$MLSPACES_ASSETS_DIR" "$MLSPACES_CACHE_DIR"
```

The first run downloads the assets into the cache, about 13 GB, and links them into the assets directory.
It also clones MolmoSpaces at a pinned commit, and makes its Python 3.11 environment in
`~/.cache/positronic/molmospaces`. Later runs use both again. Set the two variables in each new shell.

## 3. Serve a policy

Start the policy server. Use an example image for the first run:

```bash
docker run --rm --gpus all -p 8000:8000 positro/pi05-droid:main
```

The server is ready when this command answers `{"alive_seconds": ...}`:

```bash
curl -X POST http://localhost:8000/api/v1/keepalive
```

To score your own policy, start your own image in the same way. [Submit a policy image](submit-a-policy-image.md)
tells how to build one from an example recipe.

## 4. Run the eval

Run the eval from a second terminal, with the two variables of step 2 set:

```bash
positronic eval run --eval=molmo.franka_pick_mini_smoke \
  --policy=.remote --policy.address.host=localhost --policy.address.port=8000 \
  --output_dir=~/evals/molmo_smoke
```

The last line that the command prints is the score:

```
score 0.600 over 5 episodes, 0 unscored
```

- The score is the share of the episodes that succeeded, from 0 to 1.
- `unscored` counts the episodes that the scoring rule cannot read. The score does not count them.
- The output directory keeps every episode. A second run into the same directory adds its episodes, and
  its score counts only its own episodes.
- The command warns when your `positronic` is not the commit that the platform runs the eval at. The two
  runs can then differ.

To run the simulator with no GPU, render on the CPU. It is slower:

```bash
export MUJOCO_GL=egl EGL_PLATFORM=surfaceless LIBGL_ALWAYS_SOFTWARE=1
```

## 5. Look at each episode

The dataset viewer shows the video, the robot state and the result of each episode:

```bash
positronic-server --dataset.path=~/evals/molmo_smoke --ep_table_cfg=@positronic.cfg.server.eval_table
```

Open https://localhost:8400. The server makes a self-signed certificate, so the browser asks you to accept it.

## Test this guide

[`integration_tests/molmo_local_eval.py`](../integration_tests/molmo_local_eval.py) runs step 4 and checks
the result. It needs a clone of this repository, the variables of step 2 and a policy server on port 8000.
See [Integration tests](../integration_tests/README.md#local-molmospaces-eval).

## Score on Nebius Serverless, with no GPU of your own

One command on your laptop scores a policy image on a public eval in the Nebius cloud. It serves your image as
a Serverless Endpoint. It runs the eval as a Serverless Job against the endpoint, from the public image
`positro/molmospaces`. It prints the score, deletes the endpoint and the job, and checks that neither is left.

### Before you start

- A Nebius account, the [Nebius CLI](https://docs.nebius.com/cli/install) and a profile from
  `nebius profile create`. The endpoint and the job run in the project of the profile, and in its region.
- git and [uv](https://docs.astral.sh/uv/).
- The id of a VPC subnet of the project. This command lists the subnets:

  ```bash
  nebius vpc subnet list --format json
  ```

- A policy image that Nebius can pull. For a private image, put the registry credential in a
  [MysteryBox](https://docs.nebius.com/mysterybox) secret with the keys `REGISTRY_USERNAME` and
  `REGISTRY_PASSWORD`, and add `--registry-secret=<secret name>` to the command.

### Run it

Check the two requests first. This creates nothing:

```bash
git clone https://github.com/Positronic-Robotics/positronic
cd positronic
uv run positronic/cli/examples/nebius_competition/serverless_eval.py \
  --policy-image=docker.io/<you>/<image>:v1 --subnet-id=<subnet id> --dry-run
```

Then run the same command without `--dry-run`. The last line that it prints is the score, as in step 4:

```
score 0.600 over 5 episodes, 0 unscored
```

- `--eval=molmo.franka_pick_mini` runs the 20 episodes. The default is `molmo.franka_pick_mini_smoke`.
- The command makes a MysteryBox secret for the bearer token of the endpoint, and deletes it at the end.
  The endpoint and the job read the token from the secret, so no command line holds it.
- Each job downloads the MolmoSpaces assets, about 13 GB, before the first episode. The image holds none.
- `--help` shows all the settings: the platform and the preset of each workload, the timeout, the project.

### What it costs

The endpoint and the job each run on one NVIDIA L40S (`gpu-l40s-a`, preset `1gpu-8vcpu-32gb`). The list price
of one is about $1.55 an hour, and Nebius bills by the second. The run costs two GPUs for its whole time:
the pull of both images, the download of the assets, and the episodes. See [Nebius prices](https://nebius.com/prices).

### The GPU is larger than the platform GPU

The L40S has 48 GB of GPU memory. The platform gives your image 24 GB
([Submit a policy image](submit-a-policy-image.md)). An image that needs more than 24 GB passes here and fails
on the platform.

### Clean up

The command deletes the endpoint, the job and the secret when the eval ends, fails or times out, and when you
press Ctrl-C. Then it looks up each of them by name, and says `No endpoint or job of this run is left`.

- The job stops at its timeout, 120 minutes by default. Set another one with `--timeout-min`.
- The endpoint has no timeout of its own. When the command stops before its end, for example because the
  laptop sleeps, the endpoint runs and costs money until you delete it. The command prints a delete command
  for each resource that it makes. Use it, or find what is left:

  ```bash
  nebius ai endpoint list --format json
  nebius ai job list --format json
  nebius ai endpoint delete --id <endpoint id>
  nebius ai job delete --id <job id>
  ```
