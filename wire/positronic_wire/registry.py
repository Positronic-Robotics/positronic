"""Every wire a client can open a session on, by the name a caller selects it with."""

from typing import Any

from positronic_wire import grpc, roboarena, websocket, wire

# One stateless wire per name. A wire holds no address: the caller builds the wire's own and hands it in.
CLIENT_WIRES: dict[str, wire.ClientWire[Any]] = {
    client_wire.NAME: client_wire
    for client_wire in (
        websocket.WebsocketClientWire(),
        websocket.WebsocketTlsClientWire(),
        websocket.WebsocketUnixClientWire(),
        grpc.GrpcClientWire(),
        grpc.GrpcTlsClientWire(),
        roboarena.RoboarenaClientWire(),
    )
}


def client_wire(name: str) -> wire.ClientWire[Any]:
    """The wire ``name`` selects. Raises ``ValueError`` naming every wire where none is called that."""
    try:
        return CLIENT_WIRES[name]
    except KeyError:
        raise ValueError(f'No wire is called {name!r}; the wires are {", ".join(CLIENT_WIRES)}') from None
