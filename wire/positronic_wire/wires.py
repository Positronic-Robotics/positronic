"""Every wire a client can open a session on, and the one that a URL selects."""

import urllib.parse

from positronic_wire import grpc, websocket, wire

CLIENT_WIRES: tuple[wire.ClientWire, ...] = (websocket.WebsocketClientWire(), grpc.GrpcClientWire())

# The wire a URL scheme selects, and whether that scheme names TLS.
BY_SCHEME: dict[str, tuple[wire.ClientWire, wire.Scheme]] = {
    scheme.text: (client_wire, scheme) for client_wire in CLIENT_WIRES for scheme in client_wire.schemes()
}


def _session_path(path: str, url: str) -> str:
    """The session path a URL names: ``/api/v1/session``, plus the model id it addresses, if any.

    A URL naming no model — a bare host, or the endpoint with or without a trailing slash — addresses the
    endpoint itself, which serves whatever the server pinned.
    """
    if path.rstrip('/') in ('', wire.SESSION_PATH):
        return wire.SESSION_PATH
    if not path.startswith(f'{wire.SESSION_PATH}/'):
        raise ValueError(f'Unexpected path {path!r} in {url!r}; expected {wire.SESSION_PATH}[/<model_id>]')
    # Kept as written, percent-encoding included: a trailing slash is part of the id, and an id that is
    # itself a path (a HuggingFace repo) keeps its slashes as separators.
    return path


def from_url(url: str) -> tuple[wire.ClientWire, wire.SessionAddress]:
    """The wire one URL selects, and the session address it names.

    The URL is ``host``, ``host:port`` or ``scheme://host[:port][/api/v1/session[/<model_id>]]``, each with
    an optional ``?query``. The scheme selects the wire and whether the session runs over TLS
    (``BY_SCHEME`` lists them); a URL with no scheme takes the wire that lists the empty scheme, without
    TLS. The port defaults to 443 with TLS and to 80 without. The model id and the query reach the server
    as written. Raises ``ValueError`` where no wire lists the scheme, or the URL names no host.
    """
    split = urllib.parse.urlsplit(url if '://' in url else f'//{url}')
    selected = BY_SCHEME.get(split.scheme)
    if selected is None:
        raise ValueError(f'Unsupported scheme {split.scheme!r} in {url!r}')
    if not split.hostname:
        raise ValueError(f'No host in {url!r}')
    client_wire, scheme = selected
    address = wire.SessionAddress(
        host=split.hostname,
        port=wire.default_port(scheme.secure) if split.port is None else split.port,
        path=_session_path(split.path, url),
        # Forwarded verbatim: the server reads each param value as a JSON literal, and only whoever
        # wrote the URL knows whether `true` means the bool or the string.
        query=split.query,
        secure=scheme.secure,
    )
    return client_wire, address
