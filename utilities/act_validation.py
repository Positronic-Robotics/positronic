import shlex
import shutil
import subprocess
from pathlib import Path

import configuronic as cfn
import pos3

from positronic.utils.checkpoints import list_checkpoints


def _shell_join(command: list[str]) -> str:
    return ' '.join(shlex.quote(part) for part in command)


def _infer_repo_root() -> Path:
    # utilities/act_validation.py -> repo root is parent of utilities/
    return Path(__file__).resolve().parents[1]


def _build_inference_command(
    *,
    uv_path: str,
    mode: str,
    policy: str,
    checkpoints_dir: str,
    checkpoint: str,
    output_dir: str,
    extra_args: list[str],
) -> list[str]:
    cmd = [
        uv_path,
        'run',
        '--frozen',
        'positronic-inference',
        mode,
        f'--policy={policy}',
        f'--policy.base.checkpoints_dir={checkpoints_dir}',
        f'--policy.base.checkpoint={checkpoint}',
        f'--output_dir={output_dir}',
        *extra_args,
    ]
    return cmd


@cfn.config(
    mode='sim',
    policy='.act_absolute',
    checkpoint_prefix='',
    output_dir='',
    extra_args=[],
    dry_run=False,
    continue_on_error=False,
)
def main(
    checkpoints_dir: str,
    mode: str,
    policy: str,
    output_dir: str,
    checkpoint_prefix: str,
    extra_args: list[str],
    dry_run: bool,
    continue_on_error: bool,
):
    """Validate an ACT run by iterating all checkpoints and running inference for each.

    Example:

        uv run python utilities/act_validation.py \\
            --checkpoints_dir=s3://checkpoints/full_ft/act/021225/ \\
            --output_dir=s3://runs/act_validation/021225/

    This will execute commands like:

        uv run positronic-inference sim --policy=.act_absolute \\
            --policy.base.checkpoints_dir=s3://checkpoints/full_ft/act/021225/ \\
            --policy.base.checkpoint=123 \\
            --output_dir=s3://runs/act_validation/021225/
    """
    uv_path = shutil.which('uv')
    if uv_path is None:
        raise RuntimeError('Could not find `uv` on PATH.')

    if not output_dir:
        raise ValueError('`output_dir` must be provided (all checkpoint runs will write into the same directory).')

    repo_root = _infer_repo_root()
    checkpoints_root = checkpoints_dir.rstrip('/') + '/checkpoints/'

    with pos3.mirror():
        checkpoints = list_checkpoints(checkpoints_root, prefix=checkpoint_prefix)

    print(f'Found {len(checkpoints)} checkpoints in {checkpoints_root}:')
    print('  ' + ', '.join(checkpoints))
    print()

    for idx, checkpoint in enumerate(checkpoints):
        cmd = _build_inference_command(
            uv_path=uv_path,
            mode=mode,
            policy=policy,
            checkpoints_dir=checkpoints_dir,
            checkpoint=checkpoint,
            output_dir=output_dir,
            extra_args=extra_args,
        )
        print(f'[{idx + 1}/{len(checkpoints)}] Running: `{_shell_join(cmd)}`')
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
