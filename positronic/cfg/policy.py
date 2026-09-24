import json
import os
from pathlib import Path

import configuronic as cfn
from positronic_wire.wire import HostPortAddress, UnixSocketAddress, session_path

from positronic.offboard.protocol import AUTH_HEADER, AUTH_TOKEN_ENV, bearer
from positronic.policy import RemotePolicy
from positronic.utils import nebius


@cfn.config()
def unset():
    """No policy. It lives in this package so a relative `--policy=.act` still resolves against it,
    and instantiates to None rather than raising, so the absence is a value a caller can act on."""
    return None


@cfn.config()
def placeholder():
    raise RuntimeError(
        'This config is not supposed to be instantiated, '
        'and is used only to simplify relative imports of other policy configs.'
    )


@cfn.config(host='localhost', port=8000, model='', query='')
def network_address(host: str, port: int, model: str, query: str) -> HostPortAddress:
    """A session on a server reached over the network, as the network wires dial one."""
    return HostPortAddress(host, port, session_path(model), query)


@cfn.config(model='', query='')
def socket_address(uds: str, model: str, query: str) -> UnixSocketAddress:
    """A session on a server on this machine, as `websocket_unix` dials one."""
    return UnixSocketAddress(Path(uds), session_path(model), query)


remote = cfn.Config(RemotePolicy, wire='websocket', address=network_address)


@cfn.config()
def bearer_headers():
    token = os.environ.get(AUTH_TOKEN_ENV)
    if not token:
        raise ValueError(f'{AUTH_TOKEN_ENV} is not set; export the endpoint token before running inference')
    return {AUTH_HEADER: bearer(token)}


@cfn.config()
def nebius_bearer_headers():
    return {AUTH_HEADER: bearer(nebius.auth_token())}


@cfn.config()
def file_headers(path: str) -> dict[str, str]:
    """The header set at ``path``: a JSON object of header name to value.

    A path, so no credential reaches this process's command line.
    """
    file = Path(path).expanduser()
    try:
        parsed = json.loads(file.read_text())
    except Exception as exc:
        problem = f'could not be read as a header set ({type(exc).__name__})'
    else:
        if isinstance(parsed, dict) and parsed and all(isinstance(s, str) for kv in parsed.items() for s in kv):
            return parsed
        problem = 'must hold a JSON object of at least one header, every name and value a string'
        del parsed  # A frame's locals reach a traceback that renders them, and this frame raises below.
    # `from None`: a failed read or parse keeps what it choked on, a `UnicodeDecodeError` its whole
    # byte string, and a chained cause carries that into the traceback.
    raise ValueError(f'{file}: {problem}') from None


# The caller names the wire and the endpoint. `network_address` holds localhost, so a run that names no host
# sends the credential to this machine and not to a stranger.
authed_remote = cfn.Config(RemotePolicy, address=network_address, headers=bearer_headers)
nebius_remote = cfn.Config(RemotePolicy, address=network_address, headers=nebius_bearer_headers)
file_authed_remote = cfn.Config(RemotePolicy, address=network_address, headers=file_headers)
