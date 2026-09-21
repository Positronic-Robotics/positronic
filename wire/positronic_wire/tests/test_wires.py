"""One URL selects a wire and names a session address."""

import pytest
from positronic_wire import grpc, websocket, wire, wires


def test_every_scheme_selects_one_wire():
    assert set(wires.BY_SCHEME) == {'', 'http', 'https', 'ws', 'wss', 'grpc', 'grpcs'}
    for scheme, (client_wire, selected) in wires.BY_SCHEME.items():
        assert selected.text == scheme
        assert selected in client_wire.schemes()


@pytest.mark.parametrize(
    ('url', 'kind', 'host', 'port', 'secure'),
    [
        ('gpu-host', websocket.WebsocketClientWire, 'gpu-host', 80, False),
        ('gpu-host:8000', websocket.WebsocketClientWire, 'gpu-host', 8000, False),
        ('https://example.com', websocket.WebsocketClientWire, 'example.com', 443, True),
        ('wss://example.com', websocket.WebsocketClientWire, 'example.com', 443, True),
        ('grpc://gpu-host', grpc.GrpcClientWire, 'gpu-host', 80, False),
        ('grpcs://gpu-host', grpc.GrpcClientWire, 'gpu-host', 443, True),
        ('grpcs://gpu-host:9000', grpc.GrpcClientWire, 'gpu-host', 9000, True),
        ('ws://[::1]:8000', websocket.WebsocketClientWire, '::1', 8000, False),
    ],
)
def test_the_scheme_selects_the_wire_and_fixes_the_port_and_the_tls(url, kind, host, port, secure):
    client_wire, address = wires.from_url(url)
    assert isinstance(client_wire, kind)
    assert (address.host, address.port, address.secure) == (host, port, secure)


@pytest.mark.parametrize('url', ['gpu-host/', 'http://gpu-host/api/v1/session', 'http://gpu-host/api/v1/session/'])
def test_a_url_naming_no_model_is_the_bare_endpoint(url):
    assert wires.from_url(url)[1].path == wire.SESSION_PATH


def test_the_model_id_and_the_query_reach_the_address_as_written():
    _, address = wires.from_url('gpu-host:8000/api/v1/session/s3%3A//bucket/ckpt%231/?fps=2.5&pad=false')
    assert address.path == '/api/v1/session/s3%3A//bucket/ckpt%231/'
    assert address.query == 'fps=2.5&pad=false'


@pytest.mark.parametrize('url', ['gpu-host:8000/api/v2/other', 'gpu-host:8000/api/v1/sessions/10000'])
def test_a_path_outside_the_session_route_is_refused(url):
    with pytest.raises(ValueError, match='/api/v1/session'):
        wires.from_url(url)


def test_an_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match='Unsupported scheme'):
        wires.from_url('ftp://gpu-host:8000')


def test_a_url_naming_no_host_is_refused():
    with pytest.raises(ValueError, match='No host'):
        wires.from_url('ws://')
