import os
import subprocess
from pathlib import Path

import configuronic as cfn
import pos3

from positronic.utils import get_latest_checkpoint
from positronic.vendors.gr00t import MODALITY_CONFIGS


@cfn.config(checkpoint=None, groot_venv_path='/.venv/', port=9000, modality_config='ee')
def main(checkpoints_dir: str, checkpoint: str | None, port: int, groot_venv_path: str, modality_config: str):
    groot_root = Path(__file__).parents[4] / 'gr00t'
    python_bin = str(Path(groot_venv_path) / 'bin' / 'python')
    modality_config_path = MODALITY_CONFIGS.get(modality_config, modality_config)

    with pos3.mirror():
        if checkpoint is None:
            checkpoint = get_latest_checkpoint(checkpoints_dir, 'checkpoint-')
        else:
            checkpoint = 'checkpoint-' + str(checkpoint)

        checkpoint_dir = pos3.download(checkpoints_dir.rstrip('/') + '/' + checkpoint, exclude=['optimizer.pt'])

        # N1.6 uses run_gr00t_server.py with new CLI format
        command = [python_bin, 'gr00t/eval/run_gr00t_server.py']
        command.extend(['--model_path', str(checkpoint_dir)])
        command.extend(['--embodiment_tag', 'NEW_EMBODIMENT'])
        command.extend(['--modality_config_path', modality_config_path])
        command.extend(['--host', '0.0.0.0'])
        command.extend(['--port', str(port)])

        env = os.environ.copy()
        print(f'Running command: `{" ".join(command)}`\n with env: {env}')
        subprocess.run(command, env=env, check=True, cwd=str(groot_root))


if __name__ == '__main__':
    cfn.cli(main)
