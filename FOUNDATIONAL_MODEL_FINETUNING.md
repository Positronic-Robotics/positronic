# Foundational Model Fine-tuning

This document provides guidelines for robotics foundational model fine-tuning and using them for inference with the Positronics framework. Make sure to install Positronics as described in the main [README.md](README.md) file before proceeding.

Currently, only the fine-tuning of [Action Chunking with Transformers (ACT)](https://tonyzhaozh.github.io/aloha/) is described here. Other model fine-tuning procedures will be added in the future.

## Fine-tune ACT Model

### Dataset Conversion
The dataset in Positronics format should be converted to [LeRobot dataset format](https://github.com/huggingface/lerobot#the-lerobotdataset-format).

To convert the dataset to the target format, execute the following command:

```bash
uv run --with-editable .  -s positronic/training/to_lerobot.py convert \
  --input-dir <DATASET_INPUT_PATH> \
  --output-dir <CONVERTED_DATASET_PATH>
```

### Install and Configure LeRobot

1. **Clone [LeRobot source code](https://github.com/huggingface/lerobot)** into a separate directory. 

**Note:** The recently introduced LeRobot Dataset v3.0 is not compatible with the Positronics Dataset format. Support for this version will be added to Positronics later. For now, please use LeRobot version 0.3.3 by checking out the corresponding tag:

```bash
git checkout v0.3.3
```

2. **Install LeRobot from source** following the [official instructions](https://github.com/huggingface/lerobot#from-source).


3. **Apply the Positronics patch:**

   This patch adds the `PositronicEnvConfig` class to LeRobot's environment system, registers "positronic" as a valid environment type for training, and defines observation shapes and data formats compatible with Positronics datasets.

  * Copy the patch file located at `patches/positronic-lerobot.patch` to the root of the LeRobot source code repository.
  * Apply the patch:
    ```bash
    git apply positronic-lerobot.patch
    ```

### Train the Model

Run the following command from the LeRobot repository folder:

```bash
python src/lerobot/scripts/train.py \
  --policy.type act \
  --dataset.root=<CONVERTED_DATASET_PATH> \
  --dataset.repo_id=local \
  --env.type positronic \
  --eval_freq 0 \
  --policy.push_to_hub=False
```

This command starts training with the default parameters. You can adjust them to suit your model-specific purposes. For example, the default training run uses 100,000 steps. 

The full list of training parameters and their descriptions can be found in the LeRobot source code, e.g., in the [TrainPipelineConfig class] (https://github.com/huggingface/lerobot/blob/55e752f0c2e7fab0d989c5ff999fbe3b6d8872ab/src/lerobot/configs/train.py#L36).

The trained model checkpoint is automatically saved to `outputs/train` by default. You can override this location using the `output_dir` parameter.

### Inference

#### Install LeRobot in Positronics Environment

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

**Note:** When running in headless mode on a server, add `MUJOCO_GL=egl` to prevent the `AttributeError: 'Renderer' object has no attribute '_mjr_context'` error.

#### Output Files

After the inference, the file `inference.rrd` containing the simulation run data is saved in the same directory. This file path can be specified by the `--rerun_path` parameter when running the inference.

### Visualizing Inference Results

Install the [ReRun](https://rerun.io/) tool and open the generated `.rrd` file in it to visualize the simulation results.
