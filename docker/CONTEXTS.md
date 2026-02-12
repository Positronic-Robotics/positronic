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

Build and push all: `make push`

## References

- Service definitions and compose commands: `docker-compose.yml`
- Model-specific workflows: `positronic/vendors/{lerobot,gr00t,openpi}/README.md`

## Remote docker compose

When running `docker --context <remote> compose run ...`, volume paths in `docker-compose.yml` expand `${HOME}` locally (e.g. `/Users/vertix`), but the remote machine expects `/home/vertix`. Set `CACHE_ROOT` to the remote home:

```bash
CACHE_ROOT=/home/vertix docker --context vm-train compose run -d --service-ports openpi-server ...
```

## VM management

Start: `../internal/scripts/start.sh train`
Check: `ssh -o ConnectTimeout=5 vertix@vm-train 'echo ok'`
