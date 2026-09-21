"""Every wire a client can open a session on, by the name a caller selects it with."""

from positronic_wire import grpc, websocket, wire

CLIENT_WIRES: dict[str, wire.ClientWire] = {
    client_wire.NAME: client_wire
    for client_wire in (
        websocket.WebsocketClientWire(),
        websocket.WebsocketTlsClientWire(),
        grpc.GrpcClientWire(),
        grpc.GrpcTlsClientWire(),
    )
}


def client_wire(name: str) -> wire.ClientWire:
    """The wire ``name`` selects. Raises ``ValueError`` naming every wire where none is called that."""
    try:
        return CLIENT_WIRES[name]
    except KeyError:
        raise ValueError(f'No wire is called {name!r}; the wires are {", ".join(CLIENT_WIRES)}') from None
