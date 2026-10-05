"""Check a wheel installation without the repository or robotics packages on its import path."""

import argparse
from importlib.util import find_spec

from positronic_model_server import protocol, serialization, server, spec
from positronic_wire import registry, wire

if find_spec('uvicorn') is not None:
    from positronic_model_server import websocket_wire

if find_spec('grpc') is not None:
    from positronic_model_server import grpc_wire


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wire', nargs='*', default=[], help='exact set of transport names expected in this install')
    args = parser.parse_args()
    for package in ('positronic', 'torch', 'jax', 'fastapi', 'starlette', 'anyio', 'scipy', 'pydantic'):
        assert find_spec(package) is None, f'{package} must not be installed'
    assert set(registry.CLIENT_WIRES) == set(args.wire), registry.CLIENT_WIRES
    for name in args.wire:
        assert registry.client_wire(name).NAME == name
    assert server.ModelServer
    if 'websocket' in args.wire:
        assert websocket_wire.WebsocketWire
    else:
        assert find_spec('uvicorn') is None
    if 'grpc' in args.wire:
        assert grpc_wire.GrpcWire
    description = spec.sequence(spec.component('vendor-component', version=2, width=320))
    message = {protocol.META: description}
    assert serialization.deserialise(serialization.serialise(message)) == message
    assert wire.SESSION_PATH
    print(f'Isolated package imports and serialization passed; transports: {args.wire}')


if __name__ == '__main__':
    main()
