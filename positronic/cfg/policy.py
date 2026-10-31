import json
import os
from pathlib import Path

import configuronic as cfn
from platform_client.policy_container import AUTH_TOKEN_ENV
from positronic_model_server.protocol import AUTH_HEADER, bearer
from positronic_wire.wire import SESSION_PATH, HostPortAddress, UnixSocketAddress

from positronic import keys
from positronic.offboard.client import DEFAULT_CONNECT_DEADLINE
from positronic.policy import RemotePolicy
from positronic.teleop.clutch import OperatorPosition
from positronic.teleop.protocol import DEFAULT_PORT, default_socket_path, session_query
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


@cfn.config(host='localhost', port=8000, query='')
def network_address(host: str, port: int, query: str) -> HostPortAddress:
    """A session on a server reached over the network, as the network wires dial one."""
    return HostPortAddress(host, port, SESSION_PATH, query)


@cfn.config(query='')
def socket_address(uds: str, query: str) -> UnixSocketAddress:
    """A session on a server on this machine, as `websocket_unix` dials one."""
    return UnixSocketAddress(Path(uds), SESSION_PATH, query)


remote = cfn.Config(RemotePolicy, wire='websocket', address=network_address)

# A person, through the teleop service on this machine: an ordinary remote policy on the socket of the service. A frame
# waits at most the silence limit of the service, so a short infer timeout bounds it.
teleop_query = cfn.Config(session_query)
teleop = remote.override(
    wire='websocket_unix',
    address=socket_address.override(uds=str(default_socket_path()), query=teleop_query),
    infer_timeout=10.0,
)

# The views of `cfg.eval.sim.positronic.stack_cubes_collect` that a person teleoperates with, the most useful first.
TELEOP_COLLECT_CAMERAS = (keys.EXTERIOR_IMAGE, keys.WRIST_UPRIGHT_IMAGE, keys.HANDCAM_RIGHT_IMAGE, keys.WRIST_IMAGE)
teleop_collect_query = teleop_query.override(
    operator_position=OperatorPosition.BACK.name, cameras=list(TELEOP_COLLECT_CAMERAS)
)
teleop_collect = teleop.override(**{'address.query': teleop_collect_query})

# A person on MolmoSpaces. The eval puts each shoulder camera in one exterior slot at random, so the person sees both
# exteriors.
TELEOP_MOLMO_CAMERAS = (keys.EXTERIOR_IMAGE, keys.EXTERIOR_IMAGE_2, keys.WRIST_IMAGE)
teleop_molmo_query = teleop_query.override(
    operator_position=OperatorPosition.BACK.name, cameras=list(TELEOP_MOLMO_CAMERAS)
)
teleop_molmo = teleop.override(**{'address.query': teleop_molmo_query})


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

# A Modal Server with sticky sessions: each session starts a Modal session, which waits for its container, and
# dials with the session token. `modal_remote_legacy` is for a Modal web endpoint without sessions. Its edge queues a
# websocket upgrade while it starts a container, and starts a container for every upgrade the client gave up on,
# so one dial waits as long as the connect deadline.
modal_remote_legacy = file_authed_remote.override(
    wire='websocket_tls',
    address=network_address.override(port=443),
    open_timeout=DEFAULT_CONNECT_DEADLINE,
    connect_deadline=DEFAULT_CONNECT_DEADLINE,
)
modal_remote = modal_remote_legacy.override(modal_sessions=True)

# A person, through a teleop service on another machine: `--policy.address.host` names it. The token in `AUTH_TOKEN_ENV`
# is the token of the service, and `--policy.headers=None` dials a service with no token.
# `--policy.address.query=@positronic.cfg.policy.teleop_molmo_query` sets the cameras.
teleop_network = teleop.override(
    wire='websocket', address=network_address.override(port=DEFAULT_PORT, query=teleop_query), headers=bearer_headers
)
