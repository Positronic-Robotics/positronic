import os
import shlex
import shutil
import subprocess
import urllib.parse
from pathlib import Path

import configuronic as cfn

from positronic.cfg.policy import bearer_headers
from positronic.offboard.client import InferenceClient
from positronic.offboard.server import AUTH_TOKEN_ENV


def _shell_join(command: list[str]) -> str:
    return ' '.join(shlex.quote(part) for part in command)


def _infer_repo_root() -> Path:
    # utilities/validate_server.py -> repo root is parent of utilities/
    return Path(__file__).resolve().parents[1]


def _model_url(url: str, model_id: str) -> str:
    """``url``'s server and session params, addressing ``model_id``.

    Only the origin and the query survive: ``InferenceClient`` also accepts a URL that already names a
    session path, and appending to one of those would address a path inside it instead of a model.
    """
    # ``safe='/'`` keeps a path-shaped id's separators as path segments, matching how the server routes them.
    quoted = urllib.parse.quote(model_id, safe='/')
    split = urllib.parse.urlsplit(url if '://' in url else f'//{url}')
    scheme = f'{split.scheme}://' if split.scheme else ''
    query = f'?{split.query}' if split.query else ''
    return f'{scheme}{split.netloc}/api/v1/session/{quoted}{query}'


def _build_inference_command(
    *, uv_path: str, eval_ref: str, url: str, policy_ref: str, model_id: str, output_dir: str, extra_args: list[str]
) -> list[str]:
    return [
        uv_path,
        'run',
        '--locked',
        'positronic',
        'eval',
        'run',
        f'--eval={eval_ref}',
        f'--policy={policy_ref}',
        f'--policy.url={_model_url(url, model_id)}',
        f'--output_dir={output_dir}',
        *extra_args,
    ]


@cfn.config(
    eval='.sim.positronic.stack_cubes',
    output_dir='',
    extra_args=[],
    dry_run=False,
    continue_on_error=False,
    url='localhost:8000',
)
def main(
    eval: str,  # noqa: A002 — the CLI flag is `--eval`, mirroring `positronic eval run --eval=...`
    output_dir: str,
    extra_args: list[str],
    dry_run: bool,
    continue_on_error: bool,
    url: str,
):
    """Validate an inference server by iterating all available models and running inference for each.

    ``url`` names the server, in any form ``InferenceClient`` takes; a gated one also needs its bearer
    token exported as ``AUTH_TOKEN``.

    Example:

        AUTH_TOKEN=<endpoint token> uv run --locked python utilities/validate_server.py \\
            --url=https://<endpoint-managed-url> \\
            --output_dir=s3://runs/server_validation/021225/

    This will execute commands like:

        uv run --locked positronic eval run --eval=.sim.positronic.stack_cubes --policy=.authed_remote \\
            --policy.url=https://<endpoint-managed-url>/api/v1/session/checkpoint-123 \\
            --output_dir=s3://runs/server_validation/021225/checkpoint-123/
    """
    uv_path = shutil.which('uv')
    if uv_path is None:
        raise RuntimeError('Could not find `uv` on PATH.')

    if not output_dir:
        raise ValueError('`output_dir` must be provided.')

    repo_root = _infer_repo_root()

    # A served endpoint is gated on a bearer token, a server someone started by hand need not be. The token
    # picks both the header sent from here and the policy config the eval subprocess reads it back through.
    token = os.environ.get(AUTH_TOKEN_ENV)
    policy_ref = '.authed_remote' if token else '.remote'

    print(f'Connecting to {url}...')
    client = InferenceClient(url, headers=bearer_headers.instantiate() if token else None)
    try:
        models = client.list_models()
    except Exception as e:
        raise RuntimeError(f'Failed to list models from {url}: {e}') from e

    print(f'Found {len(models)} models:')
    print('  ' + ', '.join(models))
    print()

    for idx, model_id in enumerate(models):
        cmd = _build_inference_command(
            uv_path=uv_path,
            eval_ref=eval,
            url=url,
            policy_ref=policy_ref,
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
