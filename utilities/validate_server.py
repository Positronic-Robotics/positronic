import os
import shlex
import shutil
import subprocess
from pathlib import Path

import configuronic as cfn
from positronic_wire import registry
from positronic_wire import wire as wire_module

from positronic.cfg import policy as policy_cfg
from positronic.cfg.policy import bearer_headers
from positronic.offboard.client import InferenceClient
from positronic.offboard.server import AUTH_TOKEN_ENV

# The address config the socket wire's own flags hang off, named as the CLI names it.
_SOCKET_ADDRESS = 'positronic.cfg.policy.socket_address'


def _shell_join(command: list[str]) -> str:
    return ' '.join(shlex.quote(part) for part in command)


def _infer_repo_root() -> Path:
    # utilities/validate_server.py -> repo root is parent of utilities/
    return Path(__file__).resolve().parents[1]


def _build_inference_command(
    *,
    uv_path: str,
    eval_ref: str,
    wire_name: str,
    address_args: list[str],
    query: str,
    policy_ref: str,
    model_id: str,
    output_dir: str,
    extra_args: list[str],
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
        f'--policy.wire={wire_name}',
        *address_args,
        f'--policy.address.model={model_id}',
        *([f'--policy.address.query={query}'] if query else []),
        f'--output_dir={output_dir}',
        *extra_args,
    ]


def _policy_address_args(address: wire_module.SessionAddress) -> list[str]:
    """The flags that rebuild ``address`` in the eval subprocess, spelled as its own wire's."""
    if isinstance(address, wire_module.UnixSocketAddress):
        return [f'--policy.address=@{_SOCKET_ADDRESS}', f'--policy.address.uds={address.uds}']
    assert isinstance(address, wire_module.HostPortAddress), f'{type(address).__name__} spells no policy flags'
    return [f'--policy.address.host={address.host}', f'--policy.address.port={address.port}']


@cfn.config(
    eval='.sim.positronic.stack_cubes',
    output_dir='',
    extra_args=[],
    dry_run=False,
    continue_on_error=False,
    wire='websocket',
    address=policy_cfg.network_address,
)
def main(
    eval: str,  # noqa: A002 — the CLI flag is `--eval`, mirroring `positronic eval run --eval=...`
    output_dir: str,
    extra_args: list[str],
    dry_run: bool,
    continue_on_error: bool,
    wire: str,
    address: wire_module.SessionAddress,
):
    """Validate an inference server by iterating all available models and running inference for each.

    ``wire`` names the transport and the rest fill that wire's address: ``host`` and ``port`` for a
    network wire, ``uds`` for ``websocket_unix``, which reaches a server on this machine. ``wire`` is a
    websocket one, since this lists the models first and the gRPC port carries sessions alone. A gated
    server also needs its bearer token exported as ``AUTH_TOKEN``.

    Example:

        AUTH_TOKEN=<endpoint token> uv run --locked python utilities/validate_server.py \\
            --wire=websocket_tls --address.host=<endpoint-managed-host> --address.port=443 \\
            --output_dir=s3://runs/server_validation/021225/

    A server on this machine answers on a socket, which is a different address:

        uv run --locked python utilities/validate_server.py --wire=websocket_unix \\
            --address=@positronic.cfg.policy.socket_address --address.uds=/run/policy.sock \\
            --output_dir=s3://runs/server_validation/021225/

    This will execute commands like:

        uv run --locked positronic eval run --eval=.sim.positronic.stack_cubes --policy=.authed_remote \\
            --policy.wire=websocket_tls --policy.address.host=<endpoint-managed-host> --policy.address.port=443 \\
            --policy.address.model=checkpoint-123 --output_dir=s3://runs/server_validation/021225/checkpoint-123/
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

    # ``InferenceClient`` refuses an address the named wire does not dial.
    client_wire = registry.client_wire(wire)
    client = InferenceClient(client_wire, address, headers=bearer_headers.instantiate() if token else None)
    print(f'Connecting to {client.session_url}...')
    try:
        models = client.list_models()
    except Exception as e:
        raise RuntimeError(f'Failed to list models from {client.session_url}: {e}') from e

    print(f'Found {len(models)} models:')
    print('  ' + ', '.join(models))
    print()

    for idx, model_id in enumerate(models):
        cmd = _build_inference_command(
            uv_path=uv_path,
            eval_ref=eval,
            wire_name=wire,
            address_args=_policy_address_args(address),
            query=address.query,
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
