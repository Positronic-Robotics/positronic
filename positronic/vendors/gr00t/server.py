import os
import subprocess
from pathlib import Path

import configuronic as cfn
import pos3

from positronic.utils import get_latest_checkpoint


@cfn.config(checkpoint=None, groot_venv_path='/.venv/', port=9000, data_config='oxe_droid')
def main(checkpoints_dir: str, checkpoint: str | None, port: int, groot_venv_path: str, data_config: str):
    groot_root = Path(__file__).parents[4] / 'gr00t'
    python_bin = str(Path(groot_venv_path) / 'bin' / 'python')

    with pos3.mirror():
        if checkpoint is None:
            checkpoint = get_latest_checkpoint(checkpoints_dir, 'checkpoint-')
        else:
            checkpoint = 'checkpoint-' + str(checkpoint)

        checkpoint_dir = pos3.download(checkpoints_dir.rstrip('/') + '/' + checkpoint, exclude=['optimizer.pt'])

        command = [python_bin, 'scripts/inference_service.py']
        command.extend(['--server'])
        command.extend(['--model-path', str(checkpoint_dir)])
        command.extend(['--embodiment-tag', 'new_embodiment'])
        command.extend(['--data-config', data_config])
        command.extend(['--port', str(port)])

        env = os.environ.copy()
        print(f'Running command: `{" ".join(command)}`\n with env: {env}')
        subprocess.run(command, env=env, check=True, cwd=str(groot_root))


if __name__ == '__main__':
    cfn.cli(main)
