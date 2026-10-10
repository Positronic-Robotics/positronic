"""Check a wheel installation without the repository or robotics packages on its import path."""

import argparse
import ast
import re
import sys
from collections.abc import Sequence
from importlib import import_module
from importlib.metadata import distributions
from importlib.util import find_spec, resolve_name
from pathlib import Path

from positronic_model_server import protocol, serialization, server, spec
from positronic_wire import registry, wire

if find_spec('uvicorn') is not None:
    from positronic_model_server import websocket_wire

if find_spec('grpc') is not None:
    from positronic_model_server import grpc_wire


CORE_PACKAGES = {'positronic-model-server', 'positronic-wire', 'msgpack', 'numpy', 'pillow'}
TRANSPORT_PACKAGES = {'websocket': {'websockets', 'uvicorn', 'click', 'h11'}, 'grpc': {'grpcio', 'typing-extensions'}}


def check_runtime_dependencies(wires: Sequence[str]) -> None:
    """Reject unapproved distributions in a runtime-only installation, including indirect dependencies."""
    allowed = CORE_PACKAGES.copy()
    for transport, packages in TRANSPORT_PACKAGES.items():
        if transport in wires:
            allowed.update(packages)
    installed = {re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower() for dist in distributions()}
    unexpected = installed - allowed
    if unexpected:
        raise SystemExit(
            f'Unapproved model-server runtime packages: {", ".join(sorted(unexpected))}. '
            'The lightweight wrapper restricts direct and indirect dependencies. '
            'Dependency additions require explicit design review; do not expand the allowlist just to pass CI. '
            'Run this check before installing test tools.'
        )


def _constant_references(module: str, namespace: str, tree: ast.Module) -> set[str]:
    imports = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split('.')[0]] = (
                    alias.name if alias.asname else alias.name.split('.')[0]
                )
        elif isinstance(node, ast.ImportFrom):
            origin = resolve_name('.' * node.level + (node.module or ''), namespace)
            for alias in node.names:
                imports[alias.asname or alias.name] = f'{origin}.{alias.name}'

    def qualified(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return imports.get(node.id, f'{module}.{node.id}')
        if isinstance(node, ast.Attribute):
            return f'{qualified(node.value)}.{node.attr}'
        return ''

    references = set()
    for statement in tree.body:
        if isinstance(statement, ast.Assign | ast.AnnAssign):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            # A bare alias exports a name; it does not use the constant's meaning.
            if isinstance(statement.value, ast.Name | ast.Attribute) or any(
                isinstance(target, ast.Name) and target.id == '__all__' for target in targets
            ):
                continue
        for node in ast.walk(statement):
            if isinstance(node, ast.Name | ast.Attribute) and isinstance(node.ctx, ast.Load):
                references.add(qualified(node))
    return references


def check_internal_constants(package: Path) -> None:
    """Every module-level constant must be read by production code inside its defining package."""
    definitions = set()
    references = set()
    for path in package.rglob('*.py'):
        relative = path.relative_to(package)
        if (
            any(part in ('tests', 'examples') for part in relative.parts)
            or path.stem.startswith('test_')
            or path.stem.endswith('_test')
        ):
            continue
        namespace = '.'.join((package.name, *relative.parts[:-1]))
        module = namespace if path.stem == '__init__' else f'{namespace}.{path.stem}'
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.Assign | ast.AnnAssign):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                definitions.update(
                    f'{module}.{target.id}'
                    for target in targets
                    if isinstance(target, ast.Name) and target.id.isupper()
                )
        references.update(_constant_references(module, namespace, tree))
    unused = definitions - references
    if unused:
        raise SystemExit(
            f'Model-server constants without internal production use: {", ".join(sorted(unused))}. '
            'Move vendor settings, client component names and legacy-only fields to their owning modules. '
            'Tests, examples and re-exports do not justify a wrapper constant.'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wire', nargs='*', default=[], help='exact set of transport names expected in this install')
    args = parser.parse_args()
    check_runtime_dependencies(args.wire)
    check_internal_constants(Path(spec.__file__).parent)
    for package in ('positronic', 'torch', 'jax', 'fastapi', 'starlette', 'anyio', 'scipy', 'pydantic'):
        assert find_spec(package) is None, f'{package} must not be installed'
    assert set(registry.CLIENT_WIRES) == set(args.wire), registry.CLIENT_WIRES
    for name in args.wire:
        assert registry.client_wire(name).NAME == name
    assert server.ModelServer
    # Only the adapter subtree is available, as in a vendor model image.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'positronic' / 'vendors' / 'gr00t'))
    settings = import_module('serving.settings')
    recipe = import_module('serving.recipe')
    description = recipe.inference(settings.droid())
    spec.validate(description)
    assert serialization.deserialise(serialization.serialise(description)) == description
    assert 'positronic' not in sys.modules
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
    print(f'Isolated runtime dependencies, imports and serialization passed; transports: {args.wire}')


if __name__ == '__main__':
    main()
