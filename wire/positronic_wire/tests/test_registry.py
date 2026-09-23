"""The wires a caller selects by name."""

import ast
import importlib
import inspect
import pkgutil
import re
from pathlib import Path

import positronic_wire
import pytest
from positronic_wire import grpc, registry, roboarena, websocket, wire


# rules-allow: hardcoded-keys — the names are spelled as a caller types them, so the test pins them; reading
# each member's NAME would make the test and the registry agree whatever the names became.
def test_every_member_is_registered_under_its_own_name():
    assert registry.CLIENT_WIRES == {
        'websocket': registry.CLIENT_WIRES['websocket'],
        'websocket_tls': registry.CLIENT_WIRES['websocket_tls'],
        'websocket_unix': registry.CLIENT_WIRES['websocket_unix'],
        'grpc': registry.CLIENT_WIRES['grpc'],
        'grpc_tls': registry.CLIENT_WIRES['grpc_tls'],
        'roboarena': registry.CLIENT_WIRES['roboarena'],
    }
    for name, client_wire in registry.CLIENT_WIRES.items():
        assert client_wire.NAME == name
        assert issubclass(client_wire.ADDRESS, wire.SessionAddress)


@pytest.mark.parametrize(
    ('name', 'kind'),
    [
        ('websocket', websocket.WebsocketClientWire),
        ('websocket_tls', websocket.WebsocketTlsClientWire),
        ('websocket_unix', websocket.WebsocketUnixClientWire),
        ('grpc', grpc.GrpcClientWire),
        ('grpc_tls', grpc.GrpcTlsClientWire),
        ('roboarena', roboarena.RoboarenaClientWire),
    ],
)
def test_a_name_selects_its_member(name, kind):
    assert type(registry.client_wire(name)) is kind


def test_a_name_no_wire_carries_is_refused_naming_every_wire():
    with pytest.raises(ValueError, match='websocket, websocket_tls, websocket_unix, grpc, grpc_tls, roboarena'):
        registry.client_wire('ws')


@pytest.mark.parametrize(
    ('model', 'path'),
    [
        ('', wire.SESSION_PATH),
        ('10000', '/api/v1/session/10000'),
        ('GEAR-Dreams/DreamZero-DROID', '/api/v1/session/GEAR-Dreams/DreamZero-DROID'),
        ('s3://bucket/ckpt#1', '/api/v1/session/s3%3A//bucket/ckpt%231'),
        ('checkpoint-500/', '/api/v1/session/checkpoint-500/'),
    ],
)
def test_the_session_path_keeps_a_models_slashes_and_encodes_the_rest(model, path):
    assert wire.session_path(model) == path


def test_the_readme_package_table_lists_every_public_name():
    readme = (Path(__file__).parents[2] / 'README.md').read_text()
    rows = dict(re.findall(r'^\| `positronic_wire\.(\w+)` \| (.*) \|$', readme, re.MULTILINE))
    modules = {m.name for m in pkgutil.iter_modules(positronic_wire.__path__) if not m.ispkg}
    assert rows.keys() == modules
    for module in modules:
        tree = ast.parse(inspect.getsource(importlib.import_module(f'positronic_wire.{module}')))
        names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef | ast.ClassDef)]
        names += [
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
        names += [
            node.target.id
            for node in tree.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        ]
        unlisted = [name for name in names if not name.startswith('_') and not re.search(rf'`{name}\b', rows[module])]
        assert unlisted == [], module
