import ipaddress
import socket
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from cryptography import x509
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

from positronic.server import positronic_server
from positronic.server.positronic_server import (
    _access_url,
    _generate_self_signed_cert,
    _get_rrd_cache_path,
    _is_loopback,
    _served_addresses,
    app,
    app_state,
)

_Addr = namedtuple('_Addr', 'family address netmask broadcast ptp')

_INTERFACES = {
    'lo': [_Addr(socket.AF_INET, '127.0.0.1', None, None, None), _Addr(socket.AF_INET6, '::1', None, None, None)],
    'eth0': [
        _Addr(socket.AF_INET, '192.168.0.8', None, None, None),
        _Addr(socket.AF_INET6, 'fe80::1%eth0', None, None, None),
        _Addr(psutil.AF_LINK, '02:00:00:00:00:01', None, None, None),
    ],
    'vpn0': [_Addr(socket.AF_INET, '198.51.100.7', None, None, None)],
}


@pytest.fixture
def multi_homed(monkeypatch):
    monkeypatch.setattr(psutil, 'net_if_addrs', lambda: _INTERFACES)


def _certificate(hosts: list[str]) -> x509.Certificate:
    return x509.load_pem_x509_certificate(_generate_self_signed_cert(hosts).certfile.read_bytes())


def _alt_names(cert: x509.Certificate) -> x509.SubjectAlternativeName:
    return cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value


def test_wildcard_bind_serves_every_routable_local_address(multi_homed):
    assert _served_addresses('::') == ['127.0.0.1', '::1', '192.168.0.8', '198.51.100.7']


def test_an_ipv4_wildcard_advertises_no_address_its_listener_cannot_answer(multi_homed):
    assert _served_addresses('0.0.0.0') == ['127.0.0.1', '192.168.0.8', '198.51.100.7']
    assert '::1' in _served_addresses('::')


def test_every_spelling_of_the_wildcard_is_one(multi_homed):
    assert _served_addresses('0:0:0:0:0:0:0:0') == _served_addresses('::')
    assert _served_addresses('') == _served_addresses('0.0.0.0')


def test_a_wildcard_serves_an_ipv4_link_local_address(monkeypatch):
    monkeypatch.setattr(
        psutil, 'net_if_addrs', lambda: {'eth0': [_Addr(socket.AF_INET, '169.254.10.2', None, None, None)]}
    )
    assert _served_addresses('0.0.0.0') == ['169.254.10.2']


def test_concrete_bind_serves_only_the_address_it_binds(multi_homed):
    assert _served_addresses('198.51.100.7') == ['198.51.100.7']
    assert _served_addresses('127.0.0.1') == ['127.0.0.1']


def test_a_name_the_resolver_would_take_stays_a_name(multi_homed):
    assert _served_addresses('rig.local') == ['rig.local']
    assert _served_addresses('localhost') == ['localhost']


def test_certificate_names_every_served_address():
    names = _alt_names(_certificate(['198.51.100.7', '2001:db8::7']))

    assert set(names.get_values_for_type(x509.IPAddress)) == {
        ipaddress.ip_address('198.51.100.7'),
        ipaddress.ip_address('2001:db8::7'),
    }


def test_certificate_names_a_host_name_as_a_dns_entry():
    names = _alt_names(_certificate(['rig.local']))

    assert 'rig.local' in names.get_values_for_type(x509.DNSName)
    assert not names.get_values_for_type(x509.IPAddress)


def test_certificate_drops_a_zone_from_an_ip_it_names():
    names = _alt_names(_certificate(['fe80::1%eth0']))

    assert names.get_values_for_type(x509.IPAddress) == [ipaddress.ip_address('fe80::1')]


def test_certificate_carries_a_subject_a_long_host_name_would_overflow():
    host = 'a' * 60 + '.example.com'
    assert len(host.encode()) > 64
    cert = _certificate([host])

    assert cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value == 'positronic-server'
    assert host in _alt_names(cert).get_values_for_type(x509.DNSName)


def test_a_bind_naming_no_address_still_certifies():
    names = _alt_names(_certificate([]))

    assert 'localhost' in names.get_values_for_type(x509.DNSName)
    assert not names.get_values_for_type(x509.IPAddress)


def test_advertised_url_follows_the_bind_address():
    assert _access_url('http', '127.0.0.1', 8412) == 'http://127.0.0.1:8412'
    assert _access_url('https', '198.51.100.7', 8913) == 'https://198.51.100.7:8913'
    assert _access_url('https', 'rig.local', 8400) == 'https://rig.local:8400'


def test_advertised_url_brackets_an_ipv6_literal():
    assert _access_url('https', '::1', 8400) == 'https://[::1]:8400'


def test_a_loopback_bind_is_told_from_an_exposed_one(multi_homed):
    assert _is_loopback('127.0.0.1')
    assert _is_loopback('::1')
    assert _is_loopback('localhost')
    assert not _is_loopback('192.168.0.8')
    assert not _is_loopback('rig.local')


class _OneEpisodeDataset:
    def __getitem__(self, index):
        return SimpleNamespace(meta={'uid': 'ep-uid'})


@pytest.fixture
def rrd_cache(tmp_path, monkeypatch):
    monkeypatch.setitem(app_state, 'dataset', _OneEpisodeDataset())
    monkeypatch.setitem(app_state, 'cache_dir', str(tmp_path))
    monkeypatch.setitem(app_state, 'root', str(tmp_path))

    def path_under(max_hz: float, max_resolution: int) -> Path:
        return _get_rrd_cache_path(0, max_hz, max_resolution)

    return path_under


def test_a_cached_rrd_written_under_other_caps_is_not_served(rrd_cache):
    assert rrd_cache(30.0, 640) != rrd_cache(30.0, 1280)
    assert rrd_cache(30.0, 640) != rrd_cache(60.0, 640)
    assert rrd_cache(30.000001, 640) != rrd_cache(30.000002, 640)


def test_the_same_caps_reach_the_same_cached_rrd(rrd_cache):
    assert rrd_cache(30.0, 640) == rrd_cache(30.0, 640)


def test_a_stream_that_dies_partway_leaves_no_cached_rrd(rrd_cache, monkeypatch):
    monkeypatch.setitem(app_state, 'loading_state', False)
    monkeypatch.setitem(app_state, 'max_hz', 30.0)
    monkeypatch.setitem(app_state, 'max_resolution', 640)

    def _dies_partway(ds, episode_id, *, max_hz, max_resolution):
        yield b'half an episode'
        raise RuntimeError('encoder died')

    monkeypatch.setattr(positronic_server, 'stream_episode_rrd', _dies_partway)

    with pytest.raises(RuntimeError):
        TestClient(app).get('/api/episode_rrd/0')

    assert not rrd_cache(30.0, 640).exists()
