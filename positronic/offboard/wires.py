"""Every wire a client can open a session on, and the one that a URL scheme selects."""

from . import grpc_wire, websocket_wire, wire

CLIENT_WIRES: tuple[wire.ClientWire, ...] = (websocket_wire.WebsocketClientWire(), grpc_wire.GrpcClientWire())

# The wire a URL scheme selects, and whether that scheme names TLS.
BY_SCHEME: dict[str, tuple[wire.ClientWire, wire.Scheme]] = {
    scheme.text: (client_wire, scheme) for client_wire in CLIENT_WIRES for scheme in client_wire.schemes()
}
