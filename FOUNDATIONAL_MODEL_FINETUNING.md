# Foundational Model Fine-tuning

This document provides guidelines for robotics foundational model fine-tuning and using them for inference with the Positronics framework.

Currently, only the fine-tuning of [Action Chunking with Transformers (ACT)](https://tonyzhaozh.github.io/aloha/) is described here. Other model fine-tuning procedures will be added in the future.

## Prerequisites

Install Positronics as described in the main [README.md](README.md) file.

## Fine-tune ACT Model

### Prepare Data

#### Dataset Conversion
The dataset in Positronics format should be converted to [LeRobot dataset format](https://github.com/huggingface/lerobot#the-lerobotdataset-format).

To convert the dataset to the target format, execute the following command:

```bash
PYTHONPATH="$PWD" uv run positronic/training/to_lerobot.py convert \
  --input-dir <DATASET_INPUT_PATH> \
  --output-dir <CONVERTED_DATASET_PATH>
```

### Install and Configure LeRobot

Install [LeRobot from source](https://github.com/huggingface/lerobot#from-source) following the official instructions.

Apply the necessary changes to LeRobot source code:
1. Copy the patch file located in `patches/positronic-lerobot.patch` to the root of the LeRobot source code repository
2. Apply the patch:

```bash
git apply positronic-lerobot.patch
```

### Train the Model

Run the following command from the LeRobot repository folder:

```bash
PYTHONPATH="$PWD/src/" python src/lerobot/scripts/train.py \
  --policy.type act \
  --dataset.root=<CONVERTED_DATASET_PATH> \
  --dataset.repo_id=local \
  --env.type positronic \
  --eval_freq 0 \
  --policy.push_to_hub=False
```

The trained model checkpoint is automatically saved to `outputs/train`.

### Inference

#### Install LeRobot in Positronics

```bash
uv pip install -U lerobot
```

#### Run Inference

Run the following command from the Positronics repository root folder:

```bash
PYTHONPATH="$PWD" python positronic/run_inference.py sim_act \
  --simulation-time 10 \
  --policy.n_action_steps=30 \
  --policy.checkpoint_path=<CHECKPOINT_PATH>
```

The `CHECKPOINT_PATH` should point to the model checkpoint, for example:
`<LEROBOT_SOURCE_PATH>/outputs/train/2025-09-11/14-16-10_positronic_act/checkpoints/last/pretrained_model/`

#### Headless Mode (Server)

When running in headless mode on a server, add `MUJOCO_GL=egl` to prevent the `AttributeError: 'Renderer' object has no attribute '_mjr_context'` error:

```bash
PYTHONPATH="$PWD" MUJOCO_GL=egl python positronic/run_inference.py sim_act \
  --simulation-time 10 \
  --policy.n_action_steps=30 \
  --policy.checkpoint_path=<CHECKPOINT_PATH>
```

#### Output Files

After the inference, the file `inference.rrd` containing the simulation run data is saved in the same directory. This file path can be specified by the `--rerun_path` parameter when running the inference.

### Visualizing Inference Results

Install the [ReRun](https://rerun.io/) tool and open the generated `.rrd` file in it to visualize the simulation results.