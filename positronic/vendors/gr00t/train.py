import os
import subprocess
from pathlib import Path

import configuronic as cfn
import pos3

from positronic import utils
from positronic.vendors.gr00t import MODALITY_CONFIGS


def cleanup_old_optimizers(output_dir: str, keep_last_n: int = 2):
    """Delete optimizer.pt from all but the last N checkpoints to save space."""
    checkpoints = sorted(Path(output_dir).glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[1]))
    for ckpt in checkpoints[:-keep_last_n] if keep_last_n > 0 else checkpoints:
        opt_file = ckpt / 'optimizer.pt'
        if opt_file.exists():
            opt_file.unlink()
            print(f'Deleted {opt_file}')


@cfn.config(num_train_steps=None, groot_venv_path='/.venv/', modality_config='ee')
def main(
    input_path: str,
    output_path: str,
    exp_name: str,
    modality_config: str,
    num_train_steps,
    groot_venv_path: str,
    learning_rate: float = None,
    save_steps: int = None,
    resume: bool = False,
    num_workers: int = None,
    keep_optimizers_for_last_n: int = 2,
):
    exp_name = str(exp_name)
    groot_root = Path(__file__).parents[4] / 'gr00t'
    python_bin = str(Path(groot_venv_path) / 'bin' / 'python')
    modality_config_path = MODALITY_CONFIGS.get(modality_config, modality_config)

    with pos3.mirror():
        dataset_local_path = pos3.download(input_path)
        output_path = output_path.rstrip('/')
        # When resuming, don't delete existing checkpoint files
        output_dir = pos3.sync(output_path + '/' + exp_name, delete_remote=not resume)
        prefix = 'resume_metadata' if resume else 'run_metadata'
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'], prefix=prefix)

        # Calculate save_steps: 20 checkpoints per run, but at least every 2000 steps
        if save_steps is None and num_train_steps is not None:
            save_steps = max(num_train_steps // 20, 2000)

        # N1.6 uses launch_finetune.py with new CLI format (auto-resumes if checkpoint exists)
        command = [python_bin, 'gr00t/experiment/launch_finetune.py']
        command.extend(['--base_model_path', 'nvidia/GR00T-N1.6-3B'])
        command.extend(['--dataset_path', str(dataset_local_path)])
        command.extend(['--modality_config_path', modality_config_path])
        command.extend(['--embodiment_tag', 'NEW_EMBODIMENT'])
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
        print(f'Running command: `{" ".join(command)}`\n with env: {env}')
        subprocess.run(command, check=True, cwd=str(groot_root), env=env)

        # Clean up optimizer state from old checkpoints to save space
        cleanup_old_optimizers(output_dir, keep_last_n=keep_optimizers_for_last_n)


if __name__ == '__main__':
    cfn.cli(main)
