import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import configuronic as cfn

import positronic.utils.s3 as pos3
from positronic import utils


@cfn.config(config_name='pi05_positronic_lowmem')
def main(input_path: str, config_name: str, output_path: str):
    uv_path = shutil.which('uv')
    openpi_root = Path(__file__).parents[4] / 'openpi'

    with pos3.mirror():
        output_dir = pos3.upload(output_path, sync_on_error=False)
        utils.save_run_metadata(output_dir, patterns=['*.py', '*.toml'])

        input_dir = pos3.download(input_path).resolve()

        if input_dir.name != 'lerobot':
            temp_dir = tempfile.mkdtemp()
            local_input_dir = Path(temp_dir) / 'lerobot'
            os.symlink(input_dir, local_input_dir)
        else:
            local_input_dir = input_dir

        command = [uv_path, 'run', '--frozen', '--project', str(openpi_root), '--']
        command.extend(['python', 'scripts/compute_norm_stats.py'])
        command.extend(['--config-name', config_name])

        src_assets_dir = openpi_root / 'assets'
        if src_assets_dir.exists():  # Clean up the assets directory before run.
            shutil.rmtree(src_assets_dir, ignore_errors=True)

        env = os.environ.copy()
        env['HF_LEROBOT_HOME'] = local_input_dir.parent.as_posix()
        print(f'Running command: `{" ".join(command)}`\n with env: {env}')
        subprocess.run(command, env=env, check=True, cwd=openpi_root)

        # Move the 'assets' directory and its contents from openpi_root to output_dir
        dst_assets_dir = output_dir / 'assets'
        if dst_assets_dir.exists():  # Remove if destination already exists to ensure consistent copy
            shutil.rmtree(dst_assets_dir)
        shutil.move(str(src_assets_dir), str(dst_assets_dir))


if __name__ == '__main__':
    cfn.cli(main)
