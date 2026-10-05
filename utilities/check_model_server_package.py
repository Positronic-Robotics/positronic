"""Check a wheel installation without the repository or robotics packages on its import path."""

import argparse
from importlib.util import find_spec

from positronic_model_server import protocol, serialization, spec
from positronic_wire import registry, wire


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wire', nargs='*', default=[], help='exact set of transport names expected in this install')
    args = parser.parse_args()
    for package in ('positronic', 'torch', 'jax', 'fastapi', 'starlette', 'uvicorn', 'scipy', 'pydantic'):
        assert find_spec(package) is None, f'{package} must not be installed'
    assert set(registry.CLIENT_WIRES) == set(args.wire), registry.CLIENT_WIRES
    for name in args.wire:
        assert registry.client_wire(name).NAME == name
    description = spec.sequence(spec.component('vendor-component', version=2, width=320))
    message = {protocol.META: description}
    assert serialization.deserialise(serialization.serialise(message)) == message
    assert wire.SESSION_PATH
    print(f'Isolated package imports and serialization passed; transports: {args.wire}')


if __name__ == '__main__':
    main()
