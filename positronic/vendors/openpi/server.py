import os
import shutil
import subprocess
from pathlib import Path

import configuronic as cfn
import pos3

from positronic.utils import get_latest_checkpoint


@cfn.config(config_name='pi05_positronic_lowmem', checkpoint=None, port=None, extra_args=[])
def main(config_name: str, checkpoints_dir: str, checkpoint, port: int, extra_args: list[str]):
    uv_path = shutil.which('uv')
    openpi_root = Path(__file__).parents[4] / 'openpi'

    with pos3.mirror():
        if checkpoint is None:
            checkpoint = get_latest_checkpoint(checkpoints_dir)
        else:
            checkpoint = str(checkpoint)

        checkpoint_dir = pos3.download(checkpoints_dir.rstrip('/') + '/' + checkpoint)
        print(f'Checkpoint directory: {checkpoint_dir}')

        command = [uv_path, 'run', '--frozen', '--project', str(openpi_root), '--']
        command.extend(['python', 'scripts/serve_policy.py'])
        command.extend(['policy:checkpoint'])
        command.extend(['--policy.config', config_name])
        command.extend(['--policy.dir', str(checkpoint_dir)])
        if port is not None:
            command.extend(['--port', str(port)])
        command.extend(extra_args)

        env = os.environ.copy()
        print(f'Running command: `{" ".join(command)}`\n with env: {env}')
        subprocess.run(command, env=env, check=True, cwd=openpi_root)


if __name__ == '__main__':
    cfn.cli(main)
