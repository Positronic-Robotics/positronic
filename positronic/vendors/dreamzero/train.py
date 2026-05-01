"""DreamZero fine-tuning: wraps upstream Hydra training pipeline."""

import os
import subprocess
from pathlib import Path

import configuronic as cfn
import pos3

from positronic import utils

# Backbone-specific constants.
# wan2.1: 14B params, 176px height, frame_seqlen=880
# wan2.2: 5B params, 160px height, frame_seqlen from config (not CLI)
_BACKBONE_PARAMS = {
    'wan2.1': {
        'data_config': 'dreamzero/droid_relative',
        'action_head': 'wan_flow_matching_action_tf',
        'image_resolution_height': 176,
        'frame_seqlen': 880,
    },
    'wan2.2': {
        'data_config': 'dreamzero/droid_relative_wan22',
        'action_head': 'wan_flow_matching_action_tf_wan22',
        'image_resolution_height': 160,
        'frame_seqlen': None,  # set in config, not CLI
    },
}

# Architecture-specific constants.
_ARCH_PARAMS = {
    'lora': {
        'deepspeed_config': 'groot/vla/configs/deepspeed/zero2.json',
        'save_lora_only': 'true',
        'learning_rate': 1e-4,
    },
    'full': {
        'deepspeed_config': 'groot/vla/configs/deepspeed/zero2_offload.json',
        'save_lora_only': 'false',
        'learning_rate': 1e-5,
    },
}


def _dreamzero_root():
    return Path(__file__).parents[4] / 'dreamzero'


@cfn.config(
    dreamzero_venv='/.venv/',
    backbone='wan2.1',
    train_architecture='lora',
    num_gpus=1,
    max_steps=100,
    learning_rate=None,
    weight_decay=1e-5,
    save_steps=1000,
    batch_size=1,
    gradient_accumulation_steps=1,
    warmup_ratio=0.05,
    dataloader_num_workers=1,
    deepspeed=None,
    resume=False,
    save_total_limit=10,
    extra_args=[],
)
def main(
    input_path: str,
    output_path: str,
    exp_name: str,
    dreamzero_venv: str,
    backbone: str,
    train_architecture: str,
    num_gpus: int,
    max_steps: int,
    learning_rate: float | None,
    weight_decay: float,
    save_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    dataloader_num_workers: int,
    deepspeed: str | None,
    resume: bool,
    save_total_limit: int,
    extra_args: list[str],
):
    if backbone not in _BACKBONE_PARAMS:
        raise ValueError(f'Unknown backbone: {backbone!r}. Choose from {list(_BACKBONE_PARAMS)}')
    if train_architecture not in _ARCH_PARAMS:
        raise ValueError(f'Unknown train_architecture: {train_architecture!r}. Choose from {list(_ARCH_PARAMS)}')

    bb = _BACKBONE_PARAMS[backbone]
    arch = _ARCH_PARAMS[train_architecture]

    if learning_rate is None:
        learning_rate = arch['learning_rate']
    if deepspeed is None:
        deepspeed = arch['deepspeed_config']

    root = _dreamzero_root()
    venv = Path(dreamzero_venv)
    torchrun = str(venv / 'bin' / 'torchrun')

    exp_name = str(exp_name)
    dataset_local_path = pos3.download(input_path)
    output_path = output_path.rstrip('/')
    output_dir = pos3.sync(output_path + '/' + exp_name, delete_remote=not resume)
    utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])

    deepspeed_path = str(root / deepspeed) if not Path(deepspeed).is_absolute() else deepspeed

    command = [
        torchrun,
        f'--nproc_per_node={num_gpus}',
        'groot/vla/experiment/experiment.py',
        f'data={bb["data_config"]}',
        f'droid_data_root={dataset_local_path}',
        'model=dreamzero/vla',
        f'model/dreamzero/action_head={bb["action_head"]}',
        'model/dreamzero/transform=dreamzero_cotrain',
        f'train_architecture={train_architecture}',
        f'report_to={"wandb" if os.environ.get("WANDB_API_KEY") else "none"}',
        'wandb_project=dreamzero',
        'num_frames=33',
        'action_horizon=24',
        'num_views=3',
        'num_frame_per_block=2',
        'num_action_per_block=24',
        'num_state_per_block=1',
        'image_resolution_width=320',
        f'image_resolution_height={bb["image_resolution_height"]}',
        'max_chunk_size=4',
        f'per_device_train_batch_size={batch_size}',
        f'gradient_accumulation_steps={gradient_accumulation_steps}',
        f'dataloader_num_workers={dataloader_num_workers}',
        'dataloader_pin_memory=false',
        'bf16=true',
        'tf32=true',
        f'training_args.learning_rate={learning_rate}',
        f'training_args.warmup_ratio={warmup_ratio}',
        f'training_args.deepspeed={deepspeed_path}',
        f'weight_decay={weight_decay}',
        f'max_steps={max_steps}',
        f'save_steps={save_steps}',
        f'output_dir={output_dir}',
        f'save_lora_only={arch["save_lora_only"]}',
        f'save_total_limit={save_total_limit}',
    ]
    if bb['frame_seqlen'] is not None:
        command.append(f'frame_seqlen={bb["frame_seqlen"]}')
    command.extend(extra_args)

    env = os.environ.copy()
    env['VIRTUAL_ENV'] = str(venv)
    env['PATH'] = f'{venv / "bin"}:{env.get("PATH", "")}'
    env['HYDRA_FULL_ERROR'] = '1'

    print(f'Running command: `{" ".join(command)}`')
    subprocess.run(command, check=True, cwd=str(root), env=env)


# h100x1: DeepSpeed ZeRO-2 is required — the 14B model doesn't fit on a single H100 without it.
# However, on 1 GPU ZeRO-2 can't shard optimizer state, so DeepSpeed's native checkpoint save
# writes the full frozen model (~91GB) in a single torch.save() call that hits PyTorch's ZIP64
# bug. We tried exclude_frozen_parameters=True (reduces to ~1.7GB) but DeepSpeed's
# load_checkpoint doesn't support loading checkpoints saved with exclude_frozen_parameters,
# making resume fail with missing keys.
#
# Workaround: training_args.save_only_model=true skips DeepSpeed's checkpoint entirely. The
# LoRA adapter (217MB model.safetensors) is still saved by save_lora_only. Resume is NOT
# supported with h100x1 — DeepSpeed's load_checkpoint fails because there's no DeepSpeed
# state to load. To train longer, start a fresh run (the LoRA adapters train from zero anyway).
#
# h100x8: ZeRO-2 shards across 8 GPUs. LoRA checkpoints are small enough for full
# DeepSpeed saves (resume restores optimizer state). Full finetune checkpoints are ~200GB
# each (all params trainable + optimizer state), so save_only_model=true is used to avoid
# filling the disk. Resume won't restore optimizer state for full finetune h100x8.

# --- wan2.1 (14B) presets ---
wan21_lora_h100x1 = main.override(
    backbone='wan2.1',
    train_architecture='lora',
    num_gpus=1,
    batch_size=1,
    gradient_accumulation_steps=8,
    extra_args=['+training_args.save_only_model=true'],
)
wan21_lora_h100x8 = main.override(
    backbone='wan2.1', train_architecture='lora', num_gpus=8, batch_size=1, gradient_accumulation_steps=1
)
wan21_full_h100x1 = main.override(
    backbone='wan2.1',
    train_architecture='full',
    num_gpus=1,
    batch_size=1,
    gradient_accumulation_steps=8,
    extra_args=['+training_args.save_only_model=true'],
)
wan21_full_h100x8 = main.override(
    backbone='wan2.1',
    train_architecture='full',
    num_gpus=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    extra_args=['+training_args.save_only_model=true'],
)

# --- wan2.2 (5B) presets ---
wan22_lora_h100x1 = main.override(
    backbone='wan2.2',
    train_architecture='lora',
    num_gpus=1,
    batch_size=1,
    gradient_accumulation_steps=8,
    extra_args=['+training_args.save_only_model=true'],
)
wan22_lora_h100x8 = main.override(
    backbone='wan2.2', train_architecture='lora', num_gpus=8, batch_size=1, gradient_accumulation_steps=1
)
wan22_full_h100x1 = main.override(
    backbone='wan2.2',
    train_architecture='full',
    num_gpus=1,
    batch_size=1,
    gradient_accumulation_steps=8,
    extra_args=['+training_args.save_only_model=true'],
)
wan22_full_h100x8 = main.override(
    backbone='wan2.2',
    train_architecture='full',
    num_gpus=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    extra_args=['+training_args.save_only_model=true'],
)

# Legacy aliases
h100x1 = wan21_lora_h100x1
h100x8 = wan21_lora_h100x8

_ALL_PRESETS = {
    'wan21_lora_h100x1': wan21_lora_h100x1,
    'wan21_lora_h100x8': wan21_lora_h100x8,
    'wan21_full_h100x1': wan21_full_h100x1,
    'wan21_full_h100x8': wan21_full_h100x8,
    'wan22_lora_h100x1': wan22_lora_h100x1,
    'wan22_lora_h100x8': wan22_lora_h100x8,
    'wan22_full_h100x1': wan22_full_h100x1,
    'wan22_full_h100x8': wan22_full_h100x8,
    # Legacy
    'h100x1': h100x1,
    'h100x8': h100x8,
}


if __name__ == '__main__':
    with pos3.mirror():
        cfn.cli(_ALL_PRESETS)
