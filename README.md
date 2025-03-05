# Positronic

Main repository for the Positronic project.

## Installation

### Using Docker
```bash
git clone git@github.com:Positronic-Robotics/positronic.git
cd positronic
./scripts/docker/build.sh
```

### Local

Check `scripts/docker/Dockerfile` for actual commands.


## Dependencies

Dependecies are specified in `pyproject.toml`.

`requirements*.txt` files are generated using `scripts/sync-reqs.sh` and represent locked python environments.

| file | description |
|----------|----------|
| `requirements.txt` | Basic dependencies to run tests |
| `requirements-hardware.txt` | Basic +  Hardware specific dependencies (cameras, etc) |
| `requirements-all.txt` | Hardware + Dev dependencies (like ipython, jupyter, etc) |


### Adding a new dependency

If you want to add a new dependency, you can do so by adding it to the `pyproject.toml` file and then running `scripts/sync-reqs.sh`.


## How to convert teleoperated dataset into LeRobot format (ready for training)
```python
python -m positronic.training.to_lerobot output_dir=_lerobot_ds/
```

By default, this reads data from `_dataset` directory. Use `input_dir=your_dir` to control inputs. Please refer to [configs/to_lerobot.yaml](../configs/to_lerobot.yaml) for more details.


## Train the model
DATA_DIR=/tmp/ python lerobot/scripts/train.py dataset_repo_id=lerobot_ds policy=positronic env=positronic hydra.run.dir=outputs/train/positronic hydra.job.name=positronic device=cuda wandb.enable=true resume=false wandb.disable_artifact=true env.state_dim=8