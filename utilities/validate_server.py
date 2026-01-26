import shlex
import shutil
import subprocess
from pathlib import Path

import configuronic as cfn

from positronic.offboard.client import InferenceClient


def _shell_join(command: list[str]) -> str:
    return ' '.join(shlex.quote(part) for part in command)


def _infer_repo_root() -> Path:
    # utilities/validate_server.py -> repo root is parent of utilities/
    return Path(__file__).resolve().parents[1]


def _build_inference_command(
    *, uv_path: str, mode: str, host: str, port: int, model_id: str, output_dir: str, extra_args: list[str]
) -> list[str]:
    cmd = [
        uv_path,
        'run',
        '--frozen',
        'positronic-inference',
        mode,
        '--policy=.remote',
        f'--policy.host={host}',
        f'--policy.port={port}',
        f'--policy.model_id={model_id}',
        f'--output_dir={output_dir}',
        *extra_args,
    ]
    return cmd


@cfn.config(
    mode='sim', output_dir='', extra_args=[], dry_run=False, continue_on_error=False, host='localhost', port=8000
)
def main(
    mode: str, output_dir: str, extra_args: list[str], dry_run: bool, continue_on_error: bool, host: str, port: int
):
    """Validate an inference server by iterating all available models and running inference for each.

    Example:

        uv run python utilities/validate_server.py \\
            --host=notebook --port=8000 \\
            --output_dir=s3://runs/server_validation/021225/

    This will execute commands like:

        uv run positronic-inference sim --policy=.remote \\
            --policy.host=notebook --policy.port=8000 \\
            --policy.model_id=checkpoint-123 \\
            --output_dir=s3://runs/server_validation/021225/checkpoint-123/
    """
    uv_path = shutil.which('uv')
    if uv_path is None:
        raise RuntimeError('Could not find `uv` on PATH.')

    if not output_dir:
        raise ValueError('`output_dir` must be provided.')

    repo_root = _infer_repo_root()

    print(f'Connecting to {host}:{port}...')
    client = InferenceClient(host, port)
    try:
        models = client.list_models()
    except Exception as e:
        raise RuntimeError(f'Failed to list models from {host}:{port}: {e}') from e

    print(f'Found {len(models)} models:')
    print('  ' + ', '.join(models))
    print()

    for idx, model_id in enumerate(models):
        cmd = _build_inference_command(
            uv_path=uv_path,
            mode=mode,
            host=host,
            port=port,
            model_id=model_id,
            output_dir=output_dir.rstrip('/'),
            extra_args=extra_args,
        )
        print(f'[{idx + 1}/{len(models)}] Running for {model_id}: `{_shell_join(cmd)}`')
        if dry_run:
            continue

        try:
            subprocess.run(cmd, check=True, cwd=repo_root)
        except subprocess.CalledProcessError as e:
            print(f'Command failed (exit {e.returncode}): `{_shell_join(cmd)}`')
            if not continue_on_error:
                raise


if __name__ == '__main__':
    cfn.cli(main)
