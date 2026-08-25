import ipaddress
import re
import socket
import subprocess
from collections import namedtuple

import psutil
import pytest

from positronic.server.positronic_server import _access_url, _generate_self_signed_cert, _is_loopback, _served_addresses

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


def _certified_ips(extension: str) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    return {ipaddress.ip_address(address) for address in re.findall(r'IP Address:([0-9A-Fa-f.:]+)', extension)}


def _certificate_text(hosts: list[str], *fields: str) -> str:
    files = _generate_self_signed_cert(hosts)
    return subprocess.run(
        ['openssl', 'x509', '-in', files['ssl_certfile'], '-noout', *fields], check=True, capture_output=True, text=True
    ).stdout


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
    extension = _certificate_text(['198.51.100.7', '2001:db8::7'], '-ext', 'subjectAltName')

    assert _certified_ips(extension) == {ipaddress.ip_address('198.51.100.7'), ipaddress.ip_address('2001:db8::7')}


def test_certificate_names_a_host_name_as_a_dns_entry():
    extension = _certificate_text(['rig.local'], '-ext', 'subjectAltName')

    assert 'DNS:rig.local' in extension
    assert not _certified_ips(extension)


def test_certificate_drops_a_zone_from_an_ip_it_names():
    extension = _certificate_text(['fe80::1%eth0'], '-ext', 'subjectAltName')

    assert _certified_ips(extension) == {ipaddress.ip_address('fe80::1')}


def test_certificate_carries_a_subject_a_long_host_name_would_overflow():
    host = 'a' * 60 + '.example.com'
    assert len(host.encode()) > 64
    text = _certificate_text([host], '-subject', '-ext', 'subjectAltName')

    # OpenSSL prints the subject as `CN = value`, LibreSSL as `CN=value`.
    assert re.search(r'CN\s*=\s*positronic-server', text)
    assert f'DNS:{host}' in text


def test_a_bind_naming_no_address_still_certifies():
    extension = _certificate_text([], '-ext', 'subjectAltName')

    assert 'DNS:localhost' in extension
    assert not _certified_ips(extension)


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
