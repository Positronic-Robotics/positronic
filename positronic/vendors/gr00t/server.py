import os
import subprocess
from pathlib import Path

import configuronic as cfn

import positronic.utils.s3 as pos3


@cfn.config(checkpoint=None, groot_venv_path='/.venv/', port=9000, data_config='oxe_droid')
def main(checkpoints_dir: str, checkpoint: str | None, port: int, groot_venv_path: str, data_config: str):
    groot_root = Path(__file__).parents[4] / 'gr00t'
    python_bin = str(Path(groot_venv_path) / 'bin' / 'python')

    with pos3.mirror():
        if checkpoint is None:
            # Find the directories whose names are numeric and pick the maximum
            checkpoint_nums = []
            children = []
            for child in pos3.ls(checkpoints_dir, recursive=False):
                name = child.rstrip('/').split('/')[-1]
                children.append(name)
                if name.startswith('checkpoint-'):
                    checkpoint_nums.append(int(name[len('checkpoint-') :]))
            if checkpoint_nums:
                checkpoint = 'checkpoint-' + str(max(checkpoint_nums))
            else:
                raise ValueError(f'No checkpoint found in {checkpoints_dir}. Available files: {children}')
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
