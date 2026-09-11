import json
import os
import subprocess
from pathlib import Path

import configuronic as cfn
import pos3

from positronic import utils
from positronic.policy.codec import GR00T_MODALITY_PATH
from positronic.vendors import gr00t


def cleanup_old_optimizers(output_dir: Path, keep_last_n: int = 2):
    """Delete optimizer.pt from all but the last N checkpoints to save space."""
    checkpoints = sorted(output_dir.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
    for ckpt in checkpoints[:-keep_last_n] if keep_last_n > 0 else checkpoints:
        opt_file = ckpt / 'optimizer.pt'
        if opt_file.exists():
            opt_file.unlink()
            print(f'Deleted {opt_file}')


@cfn.config(num_train_steps=None, groot_venv_path=gr00t.VENV, base_model=gr00t.BASE_MODEL, batch_size=64)
@pos3.mirror()
def main(
    input_path: str,
    output_path: str,
    exp_name: str,
    base_model: str,
    batch_size: int,
    num_train_steps,
    groot_venv_path: str,
    learning_rate: float | None = None,
    save_steps: int | None = None,
    resume: bool = False,
    num_workers: int | None = None,
    keep_optimizers_for_last_n: int = 2,
):
    exp_name = str(exp_name)
    groot_root = Path(__file__).parents[4] / 'gr00t'
    python_bin = str(Path(groot_venv_path).expanduser() / 'bin' / 'python')

    dataset_local_path = pos3.download(input_path)
    with (Path(dataset_local_path) / GR00T_MODALITY_PATH).open() as f:
        video_keys = list(json.load(f)[gr00t.VIDEO])
    output_path = output_path.rstrip('/')
    output_dir = pos3.sync(output_path + '/' + exp_name, delete_remote=not resume)
    prefix = 'resume_metadata' if resume else 'run_metadata'
    utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'], prefix=prefix)

    command = [python_bin, 'gr00t/experiment/launch_finetune.py']
    command.extend(['--base-model-path', base_model])
    command.extend(['--dataset_path', str(dataset_local_path)])
    command.extend(['--video-keys', *video_keys])
    command.extend(['--embodiment-tag', gr00t.EMBODIMENT])
    command.extend(['--global-batch-size', str(batch_size)])
    if resume:
        command.append('--resume-from-checkpoint')
    command.extend(['--output_dir', str(output_dir)])
    command.extend(['--num_gpus', '1'])
    command.extend(['--save_total_limit', '9999'])  # Keep all checkpoints
    if num_train_steps is not None:
        command.extend(['--max_steps', str(num_train_steps)])
    if learning_rate is not None:
        command.extend(['--learning_rate', str(learning_rate)])
    if save_steps is not None:
        command.extend(['--save_steps', str(save_steps)])
    if num_workers is not None:
        command.extend(['--dataloader_num_workers', str(num_workers)])
    command.append('--use-wandb')

    env = os.environ.copy()
    print(f'Running command: {command}')
    subprocess.run(command, check=True, cwd=str(groot_root), env=env)

    cleanup_old_optimizers(Path(output_dir), keep_last_n=keep_optimizers_for_last_n)


if __name__ == '__main__':
    cfn.cli(main)
