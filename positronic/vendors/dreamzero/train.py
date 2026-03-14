"""DreamZero LoRA fine-tuning: wraps upstream Hydra training pipeline."""

import os
import subprocess
from pathlib import Path

import configuronic as cfn
import pos3

from positronic import utils

CACHE_DIR = Path.home() / '.cache' / 'dreamzero'
WAN_REPO = 'Wan-AI/Wan2.1-I2V-14B-480P'
UMT5_REPO = 'google/umt5-xxl'


def _dreamzero_root():
    return Path(__file__).parents[4] / 'dreamzero'


def _download_base_weights(python_bin: str):
    """Download Wan2.1-I2V-14B-480P and umt5-xxl to persistent cache.

    Training passes explicit paths to these weights as Hydra overrides
    (dit_version, text_encoder_pretrained_path, etc.), so we need them
    at known locations rather than relying on HF auto-caching.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    wan_dir = CACHE_DIR / 'Wan2.1-I2V-14B-480P'
    umt5_dir = CACHE_DIR / 'umt5-xxl'

    if not wan_dir.exists():
        print(f'Downloading {WAN_REPO} to {wan_dir}...')
        subprocess.run(
            [
                python_bin,
                '-c',
                f'from huggingface_hub import snapshot_download; '
                f'snapshot_download("{WAN_REPO}", local_dir="{wan_dir}")',
            ],
            check=True,
        )

    if not umt5_dir.exists():
        print(f'Downloading {UMT5_REPO} to {umt5_dir}...')
        subprocess.run(
            [
                python_bin,
                '-c',
                f'from huggingface_hub import snapshot_download; '
                f'snapshot_download("{UMT5_REPO}", local_dir="{umt5_dir}")',
            ],
            check=True,
        )

    return wan_dir, umt5_dir


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
    python_bin = str(venv / 'bin' / 'python')
    torchrun = str(venv / 'bin' / 'torchrun')

    wan_dir, umt5_dir = _download_base_weights(python_bin)

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
        f'dit_version={wan_dir}',
        f'text_encoder_pretrained_path={wan_dir}/models_t5_umt5-xxl-enc-bf16.pth',
        f'image_encoder_pretrained_path={wan_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth',
        f'vae_pretrained_path={wan_dir}/Wan2.1_VAE.pth',
        f'tokenizer_path={umt5_dir}',
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


if __name__ == '__main__':
    with pos3.mirror():
        cfn.cli({'h100x1': h100x1, 'h100x8': h100x8})
