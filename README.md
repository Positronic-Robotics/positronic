# Positronic

Main repository for the Positronic project.

## Installation

There are two supported ways to install and run Positronic:

1) Docker (recommended for reproducibility)
2) Native install using `uv` (fast and lockfile-driven)

### Option 1: Docker
```bash
git clone git@github.com:Positronic-Robotics/positronic.git
cd positronic

# Build the image (includes libfranka build and Python deps from uv.lock)
./docker/build.sh

# Run commands inside the container (mounts repo, sets PYTHONPATH)
./docker/run.sh pytest          # run tests
./docker/run.sh python -m pimm  # example: run a module
```

For real-time flags (rtprio/memlock), use `./docker/run-rt.sh` instead of `run.sh`.
For container details, see `docker/Dockerfile`.

### Option 2: Local via uv

Prerequisites: install `uv` (https://docs.astral.sh/uv/). Python 3.11 is required.

```bash
git clone git@github.com:Positronic-Robotics/positronic.git
cd positronic

# Install dev environment from the lockfile (recommended)
uv sync --frozen --extra dev

# Run tests (coverage is enabled via pyproject addopts)
uv run pytest

# If you need hardware extras as well
uv sync --frozen --extra dev --extra hardware

```

## Dependencies

- All dependencies are declared in `pyproject.toml`.
- Versions are locked via `uv.lock` and installed with `uv sync --frozen` for reproducibility.

### Adding or changing dependencies

Use `uv` to manage deps and the lockfile:

```bash
# Add/remove packages
uv add <package>
uv remove <package>

# Regenerate lock after changes
uv lock

# Install from the updated lock
uv sync --frozen
```

For optional features, use extras: `--extra dev` or `--extra hardware`.

## Development Installation

Set up a development environment with tests, linters, and coverage:

```bash
uv sync --frozen --extra dev

# Lint
uv run flake8

# Run tests with coverage (terminal)
uv run pytest

# Generate HTML coverage report
uv run pytest --cov-report=html
# Open htmlcov/index.html in your browser
```

## How to convert teleoperated dataset into LeRobot format (ready for training)
```python
python -m positronic.training.to_lerobot output_dir=_lerobot_ds/
```

By default, this reads data from `_dataset` directory. Use `input_dir=your_dir` to control inputs. Please refer to [configs/to_lerobot.yaml](../configs/to_lerobot.yaml) for more details.

## Train the model
DATA_DIR=/tmp/ python lerobot/scripts/train.py dataset_repo_id=lerobot_ds policy=positronic env=positronic hydra.run.dir=outputs/train/positronic hydra.job.name=positronic device=cuda wandb.enable=true resume=false wandb.disable_artifact=true env.state_dim=8
