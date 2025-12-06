import os
import subprocess
from pathlib import Path

import configuronic as cfn

import positronic.utils.s3 as pos3
from positronic import utils


@cfn.config(
    num_train_steps=None, groot_venv_path='/.venv/', data_config='ee_absolute', resume=False, learning_rate=None
)
def main(
    input_path: str,
    output_path: str,
    exp_name: str,
    data_config: str,
    num_train_steps,
    resume: bool,
    groot_venv_path: str,
    learning_rate: float,
):
    exp_name = str(exp_name)
    groot_root = Path(__file__).parents[4] / 'gr00t'
    python_bin = str(Path(groot_venv_path) / 'bin' / 'python')

    with pos3.mirror():
        dataset_local_path = pos3.download(input_path)
        output_path = output_path.rstrip('/')
        output_dir = pos3.sync(output_path + '/' + exp_name, delete_remote=not resume)
        prefix = 'resume_metadata' if resume else 'run_metadata'
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'], prefix=prefix)

        command = [python_bin, 'scripts/gr00t_finetune.py']
        command.extend(['--video_backend', 'torchvision_av'])
        command.extend(['--num_gpus', '1'])
        command.extend(['--data-config', data_config])
        command.extend(['--output-dir', str(output_dir)])
        command.extend(['--dataset-path', str(dataset_local_path)])
        if num_train_steps is not None:
            command.extend(['--max_steps', str(num_train_steps)])
        if resume:
            command.extend(['--resume'])
        if learning_rate is not None:
            command.extend(['--learning_rate', str(learning_rate)])

        env = os.environ.copy()
        env['WANDB_PROJECT'] = 'groot'
        print(f'Running command: `{" ".join(command)}`\n with env: {env}')
        subprocess.run(command, check=True, cwd=str(groot_root), env=env)


if __name__ == '__main__':
    cfn.cli(main)
