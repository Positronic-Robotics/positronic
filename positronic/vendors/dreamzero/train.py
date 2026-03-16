"""DreamZero LoRA fine-tuning: wraps upstream Hydra training pipeline."""

import os
import subprocess
from pathlib import Path

import configuronic as cfn
import pos3

from positronic import utils


def _dreamzero_root():
    return Path(__file__).parents[4] / 'dreamzero'


@cfn.config(
    dreamzero_venv='/.venv/',
    num_gpus=1,
    max_steps=100,
    learning_rate=1e-5,
    weight_decay=1e-5,
    save_steps=1000,
    batch_size=1,
    gradient_accumulation_steps=1,
    warmup_ratio=0.05,
    dataloader_num_workers=1,
    deepspeed=None,
    resume=False,
    extra_args=[],
)
def main(
    input_path: str,
    output_path: str,
    exp_name: str,
    dreamzero_venv: str,
    num_gpus: int,
    max_steps: int,
    learning_rate: float,
    weight_decay: float,
    save_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    dataloader_num_workers: int,
    deepspeed: str | None,
    resume: bool,
    extra_args: list[str],
):
    root = _dreamzero_root()
    venv = Path(dreamzero_venv)
    torchrun = str(venv / 'bin' / 'torchrun')

    exp_name = str(exp_name)
    dataset_local_path = pos3.download(input_path)
    output_path = output_path.rstrip('/')
    output_dir = pos3.sync(output_path + '/' + exp_name, delete_remote=not resume)
    utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])

    # Architecture constants from upstream droid_training.sh.
    # These define the DreamZero DROID model geometry and must match the pretrained weights.
    command = [
        torchrun,
        f'--nproc_per_node={num_gpus}',
        'groot/vla/experiment/experiment.py',
        'data=dreamzero/droid_relative',
        f'droid_data_root={dataset_local_path}',
        'model=dreamzero/vla',
        'model/dreamzero/action_head=wan_flow_matching_action_tf',
        'model/dreamzero/transform=dreamzero_cotrain',
        'train_architecture=lora',
        f'report_to={"wandb" if os.environ.get("WANDB_API_KEY") else "none"}',
        'wandb_project=dreamzero',
        'num_frames=33',
        'action_horizon=24',
        'num_views=3',
        'num_frame_per_block=2',
        'num_action_per_block=24',
        'num_state_per_block=1',
        'image_resolution_width=320',
        'image_resolution_height=176',
        'frame_seqlen=880',
        'max_chunk_size=4',
        f'per_device_train_batch_size={batch_size}',
        f'gradient_accumulation_steps={gradient_accumulation_steps}',
        f'dataloader_num_workers={dataloader_num_workers}',
        'dataloader_pin_memory=false',
        'bf16=true',
        'tf32=true',
        f'training_args.learning_rate={learning_rate}',
        f'training_args.warmup_ratio={warmup_ratio}',
        f'weight_decay={weight_decay}',
        f'max_steps={max_steps}',
        f'save_steps={save_steps}',
        f'output_dir={output_dir}',
        'save_lora_only=true',
        'save_total_limit=10',
    ]
    if deepspeed is not None:
        deepspeed_path = str(root / deepspeed) if not Path(deepspeed).is_absolute() else deepspeed
        command.append(f'training_args.deepspeed={deepspeed_path}')
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
# h100x8: ZeRO-2 shards across 8 GPUs (~12GB per shard), no ZIP64 issue. Full DeepSpeed
# checkpoints are saved, so resume restores optimizer state exactly.
h100x1 = main.override(
    num_gpus=1,
    batch_size=1,
    gradient_accumulation_steps=8,
    deepspeed='groot/vla/configs/deepspeed/zero2.json',
    extra_args=['+training_args.save_only_model=true'],
)
h100x8 = main.override(
    num_gpus=8, batch_size=1, gradient_accumulation_steps=1, deepspeed='groot/vla/configs/deepspeed/zero2.json'
)

# h200x1: Same constraints as h100x1 (ZeRO-2 on 1 GPU can't shard optimizer state), but
# 141GB HBM3e allows batch_size=2. We halve gradient_accumulation_steps to keep the same
# effective batch size of 8.
h200x1 = main.override(
    num_gpus=1,
    batch_size=2,
    gradient_accumulation_steps=4,
    deepspeed='groot/vla/configs/deepspeed/zero2.json',
    extra_args=['+training_args.save_only_model=true'],
)


if __name__ == '__main__':
    with pos3.mirror():
        cfn.cli({'h100x1': h100x1, 'h100x8': h100x8, 'h200x1': h200x1})
