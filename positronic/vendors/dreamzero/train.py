"""DreamZero LoRA fine-tuning: wraps upstream Hydra training pipeline."""

import os
import subprocess

import configuronic as cfn
import pos3

from positronic import utils
from positronic.vendors.dreamzero.server import DREAMZERO_ROOT, DREAMZERO_VENV, _download_base_weights


@cfn.config(
    num_gpus=1,
    max_steps=100,
    learning_rate=1e-5,
    save_steps=1000,
    batch_size=1,
    gradient_accumulation_steps=1,
    warmup_ratio=0.05,
    dataloader_num_workers=10,
    resume=False,
    extra_args=[],
)
def main(
    input_path: str,
    output_path: str,
    exp_name: str,
    num_gpus: int,
    max_steps: int,
    learning_rate: float,
    save_steps: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
    dataloader_num_workers: int,
    resume: bool,
    extra_args: list[str],
):
    exp_name = str(exp_name)
    torchrun = str(DREAMZERO_VENV / 'bin' / 'torchrun')

    wan_dir, umt5_dir = _download_base_weights()

    with pos3.mirror():
        dataset_local_path = pos3.download(input_path)
        output_path = output_path.rstrip('/')
        output_dir = pos3.sync(output_path + '/' + exp_name, delete_remote=not resume)
        prefix = 'resume_metadata' if resume else 'run_metadata'
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'], prefix=prefix)

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
            'bf16=true',
            'tf32=true',
            f'training_args.deepspeed={DREAMZERO_ROOT}/groot/vla/configs/deepspeed/zero2.json',
            f'training_args.learning_rate={learning_rate}',
            f'training_args.warmup_ratio={warmup_ratio}',
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
        command.extend(extra_args)

        env = os.environ.copy()
        env['VIRTUAL_ENV'] = str(DREAMZERO_VENV)
        env['PATH'] = f'{DREAMZERO_VENV / "bin"}:{env.get("PATH", "")}'
        env['HYDRA_FULL_ERROR'] = '1'

        print(f'Running command: `{" ".join(command)}`')
        subprocess.run(command, check=True, cwd=str(DREAMZERO_ROOT), env=env)


if __name__ == '__main__':
    cfn.cli(main)
