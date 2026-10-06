# Docker contexts and machines

| Context | GPU | Typical use |
|---------|-----|-------------|
| `desktop` | RTX 3060 (12GB) | LeRobot training/inference, GR00T inference |
| `notebook` | RTX 4060 (8GB) | GR00T inference, light tasks |
| `vm-train` / `vm-train2` / `vm-train3` | H100 (80GB) | OpenPI/GR00T training and inference |

## Images

| Image | Used For |
|-------|----------|
| `positro/positronic` | Dataset conversion, LeRobot training/inference |
| `positro/gr00t` | GR00T training and inference |
| `positro/openpi` | OpenPI training and inference |
| `positro/dreamzero` | DreamZero inference (1+ GPU, H100 80GB recommended) |
| `positro/robolab` | RoboLab (Isaac Lab) eval — runs `positronic eval run`, which spawns the Isaac sim subprocess in-container; needs an RTX-class GPU |
| `positro/galaxea` | G0.5-DROID inference, internal non-commercial evaluation only; isolated Galaxea and Positronic Python environments |

`Dockerfile.serve-<name>` builds `positro/<name>`, a policy image the platform runs: one
checkpoint, its weights inside, offline, serving on `:8000`. `pi05-droid` and `gr00t-n17-droid`
build on `positro/openpi-base` and `positro/gr00t-base`, and `molmoact2-droid` on a Python 3.13
image with uv. CI publishes each one; `make build-serve-<name>` builds it. `Dockerfile.flux3-action`
and `Dockerfile.cosmos3-nano` build the FLUX 3 Action and Cosmos3-Nano policy images, which CI
publishes as `positro/flux3-action` and `positro/cosmos3-nano`. See
[Submit a policy image](../docs/submit-a-policy-image.md).

Build and push all: `make push`

Galaxea is opt-in: `make build-galaxea`. Its evaluation-only image is excluded from
aggregate builds and pushes; see [the vendor README](../positronic/vendors/galaxea/README.md).

## References

- Service definitions and compose commands: `docker-compose.yml`
- Model-specific workflows: `positronic/vendors/{lerobot,gr00t,openpi}/README.md`

## Remote docker compose

When running `docker --context <remote> compose run ...`, volume paths in `docker-compose.yml` expand `${HOME}` locally (e.g. `/Users/<user>`), but the remote machine expects `/home/<user>`. Set `CACHE_ROOT` to the remote home:

```bash
CACHE_ROOT=/home/<user> docker --context vm-train compose run -d --service-ports openpi-server ...
```

## Restart policy

Start a container with `--restart on-failure:2`, not `--restart unless-stopped`. A container that crashes and
restarts reports `Up 1 second` to every `docker ps`, which reads as a slow start. `on-failure:2` stops the
container after the second crash, so `docker ps` shows it dead and `docker logs` gives the fault.

## VM management

Start: `../internal/scripts/start.sh train`
Check: `ssh -o ConnectTimeout=5 <user>@vm-train 'echo ok'`
