"""Installed client transports, selected by name."""

from importlib.util import find_spec
from typing import Any

from positronic_wire import wire

# One wire per name. A wire takes the address on each call: the caller builds the wire's own and hands it in. Only
# `modal_websocket_tls` holds state, the Modal sessions of this process on each address.
CLIENT_WIRES: dict[str, wire.ClientWire[Any]] = {}

if find_spec('websockets') is not None:
    from positronic_wire import modal, roboarena, websocket

    for _member in (
        websocket.WebsocketClientWire(),
        websocket.WebsocketTlsClientWire(),
        websocket.WebsocketUnixClientWire(),
        roboarena.RoboarenaClientWire(),
        modal.ModalClientWire(websocket.WebsocketTlsClientWire()),
    ):
        CLIENT_WIRES[_member.NAME] = _member

if find_spec('grpc') is not None:
    from positronic_wire import grpc

    for _member in (grpc.GrpcClientWire(), grpc.GrpcTlsClientWire()):
        CLIENT_WIRES[_member.NAME] = _member


def client_wire(name: str) -> wire.ClientWire[Any]:
    """Select an installed wire, or raise with the available names and installation extras."""
    try:
        return CLIENT_WIRES[name]
    except KeyError:
        raise ValueError(
            f'No wire is called {name!r}; the installed wires are {", ".join(CLIENT_WIRES)}. '
            'Install positronic-wire[websocket] or positronic-wire[grpc] for the required transport.'
        ) from None
