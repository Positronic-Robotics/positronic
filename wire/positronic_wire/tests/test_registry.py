"""The wires a caller selects by name."""

import pytest
from positronic_wire import grpc, registry, websocket, wire


# rules-allow: hardcoded-keys — the names are spelled as a caller types them, so the test pins them; reading
# each member's NAME would make the test and the registry agree whatever the names became.
def test_every_member_is_registered_under_its_own_name():
    assert registry.CLIENT_WIRES == {
        'websocket': registry.CLIENT_WIRES['websocket'],
        'websocket_tls': registry.CLIENT_WIRES['websocket_tls'],
        'grpc': registry.CLIENT_WIRES['grpc'],
        'grpc_tls': registry.CLIENT_WIRES['grpc_tls'],
    }
    for name, client_wire in registry.CLIENT_WIRES.items():
        assert client_wire.NAME == name
        assert isinstance(client_wire.DEFAULT_PORT, int)


@pytest.mark.parametrize(
    ('name', 'kind'),
    [
        ('websocket', websocket.WebsocketClientWire),
        ('websocket_tls', websocket.WebsocketTlsClientWire),
        ('grpc', grpc.GrpcClientWire),
        ('grpc_tls', grpc.GrpcTlsClientWire),
    ],
)
def test_a_name_selects_its_member(name, kind):
    assert type(registry.client_wire(name)) is kind


def test_a_name_no_wire_carries_is_refused_naming_every_wire():
    with pytest.raises(ValueError, match='websocket, websocket_tls, grpc, grpc_tls'):
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
