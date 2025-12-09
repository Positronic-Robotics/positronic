import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import configuronic as cfn
import pos3

from positronic import utils


@cfn.config(config_name='pi05_positronic_lowmem', resume=False, num_train_steps=None, extra_args=[])
def main(
    input_path: str,
    config_name: str,
    stats_path: str,
    output_path: str,
    exp_name: str,
    resume: bool,
    num_train_steps: int | None,
    extra_args: list[str],
):
    uv_path = shutil.which('uv')
    openpi_root = Path(__file__).parents[4] / 'openpi'
    exp_name = str(exp_name)  # Just in case, as sometimes exp name can look like numbers

    with pos3.mirror():
        input_dir = pos3.download(input_path).resolve()
        stats_dir = pos3.download(stats_path).resolve()
        output_path = output_path.rstrip('/')
        output_dir = pos3.sync(output_path + '/' + config_name + '/' + exp_name, delete_remote=not resume)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])

        if input_dir.name != 'lerobot':
            temp_dir = tempfile.mkdtemp()
            local_input_dir = Path(temp_dir) / 'lerobot'
            os.symlink(input_dir, local_input_dir)
        else:
            local_input_dir = input_dir

        command = [uv_path, 'run', '--frozen', '--project', str(openpi_root), '--']
        command.extend(['python', 'scripts/train.py'])
        command.extend([config_name, '--exp-name', exp_name])
        command.append('--resume' if resume else '--overwrite')
        if num_train_steps is not None:
            command.append(f'--num-train-steps={num_train_steps}')
        command.extend(['--assets-base-dir', stats_dir.as_posix()])
        command.extend(['--checkpoint-base-dir', output_dir.parent.parent.as_posix()])
        command.extend(extra_args)

        env = os.environ.copy()
        env.update({'HF_LEROBOT_HOME': local_input_dir.parent.as_posix(), 'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.995'})
        print(f'Running command: `{" ".join(command)}`\n with env: {env}')
        subprocess.run(command, env=env, check=True, cwd=openpi_root)


if __name__ == '__main__':
    cfn.cli(main)
