import ipaddress
import re
import socket
import subprocess
from collections import namedtuple

import psutil
import pytest

from positronic.server.positronic_server import (
    _access_url,
    _generate_self_signed_cert,
    _insecure_context_warning,
    _served_addresses,
    _ssl_kwargs,
    _subject_alt_names,
)

_Addr = namedtuple('_Addr', 'family address netmask broadcast ptp')

# A multi-homed host: loopback, a LAN interface with a link-local companion and a MAC, and a tailnet.
_INTERFACES = {
    'lo': [_Addr(socket.AF_INET, '127.0.0.1', None, None, None), _Addr(socket.AF_INET6, '::1', None, None, None)],
    'eth0': [
        _Addr(socket.AF_INET, '192.168.0.8', None, None, None),
        _Addr(socket.AF_INET6, 'fe80::985a:62ff:fe48:f8e0%eth0', None, None, None),
        _Addr(psutil.AF_LINK, '9a:5a:62:48:f8:e0', None, None, None),
    ],
    'tailscale0': [_Addr(socket.AF_INET, '100.108.71.121', None, None, None)],
}


@pytest.fixture
def multi_homed(monkeypatch):
    monkeypatch.setattr(psutil, 'net_if_addrs', lambda: _INTERFACES)


def test_wildcard_bind_serves_every_routable_local_address(multi_homed):
    assert _served_addresses('::') == ['127.0.0.1', '::1', '192.168.0.8', '100.108.71.121']


def test_an_ipv4_wildcard_advertises_no_address_its_listener_cannot_answer(multi_homed):
    # `0.0.0.0` binds AF_INET, so a v6 URL built from it names an address nothing is listening on —
    # and a certificate carrying it certifies a host the server never serves. `::` accepts v4 too.
    assert _served_addresses('0.0.0.0') == ['127.0.0.1', '192.168.0.8', '100.108.71.121']
    assert '::1' in _served_addresses('::')


def test_every_spelling_of_the_wildcard_is_one(multi_homed):
    # The bind normalizes an address, so two spellings of one wildcard serve the same set. A
    # spelling read as an address of its own is certified and advertised as one.
    assert _served_addresses('0:0:0:0:0:0:0:0') == _served_addresses('::')
    assert _served_addresses('') == _served_addresses('0.0.0.0')


def test_a_wildcard_serves_an_ipv4_link_local_address(monkeypatch):
    # 169.254/16 carries no zone and is a URL like any other, so a wildcard listener answers on it.
    # Only a v6 link-local is unreachable by URL, and that is what the filter is for.
    monkeypatch.setattr(
        psutil, 'net_if_addrs', lambda: {'eth0': [_Addr(socket.AF_INET, '169.254.10.2', None, None, None)]}
    )
    assert _served_addresses('0.0.0.0') == ['169.254.10.2']


def test_concrete_bind_serves_only_the_address_it_binds(multi_homed):
    assert _served_addresses('100.108.71.121') == ['100.108.71.121']
    assert _served_addresses('127.0.0.1') == ['127.0.0.1']
    assert _served_addresses('rig.local') == ['rig.local']


def test_certificate_names_every_served_address():
    assert _subject_alt_names(['127.0.0.1', '::1', '192.168.0.8']) == 'IP:127.0.0.1,IP:::1,IP:192.168.0.8,DNS:localhost'


def test_certificate_names_no_address_beyond_the_bind():
    assert _subject_alt_names(['100.108.71.121']) == 'IP:100.108.71.121'
    assert _subject_alt_names(['rig.local']) == 'DNS:rig.local'


def test_certificate_drops_a_zone_from_an_ip_it_names():
    # The socket layer binds `fe80::1%eth0`; OpenSSL refuses the same string as a bad IP address, so
    # a SAN carrying the zone fails certificate generation and the server never reaches uvicorn.
    assert _subject_alt_names(['fe80::1%eth0']) == 'IP:fe80::1'


def test_generated_certificate_carries_the_bind_addresses_and_no_others():
    hosts = ['100.108.71.121', 'fd7a:115c:a1e0::a53a:477a']
    files = _generate_self_signed_cert(hosts)
    extension = subprocess.run(
        ['openssl', 'x509', '-in', files['ssl_certfile'], '-noout', '-ext', 'subjectAltName'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    certified = {ipaddress.ip_address(address) for address in re.findall(r'IP Address:([0-9A-Fa-f.:]+)', extension)}
    assert certified == {ipaddress.ip_address(host) for host in hosts}
    assert 'DNS:localhost' not in extension


def test_advertised_url_follows_the_bind_address():
    assert _access_url('http', '127.0.0.1', 8412) == 'http://127.0.0.1:8412'
    assert _access_url('https', '100.108.71.121', 8913) == 'https://100.108.71.121:8913'
    assert _access_url('https', 'rig.local', 8400) == 'https://rig.local:8400'


def test_advertised_url_brackets_an_ipv6_literal():
    assert _access_url('https', '::1', 8400) == 'https://[::1]:8400'


def test_advertised_url_encodes_a_zone_the_way_a_url_carries_one():
    # RFC 6874: a bare `%` is not a URL a browser takes.
    assert _access_url('https', 'fe80::1%eth0', 8400) == 'https://[fe80::1%25eth0]:8400'


def test_plain_http_beyond_loopback_warns_that_video_will_not_decode():
    warning = _insecure_context_warning(['192.168.0.8'], https=False)

    assert warning is not None
    assert 'VideoDecoder is not defined' in warning
    assert '192.168.0.8' in warning


def test_plain_http_wildcard_bind_warns(multi_homed):
    assert _insecure_context_warning(_served_addresses('0.0.0.0'), https=False) is not None


def test_loopback_or_https_does_not_warn():
    assert _insecure_context_warning(['127.0.0.1', '::1'], https=False) is None
    assert _insecure_context_warning(['localhost'], https=False) is None
    assert _insecure_context_warning(['192.168.0.8'], https=True) is None


def test_supplied_certificate_is_served_instead_of_a_generated_one():
    assert _ssl_kwargs(True, ['192.168.0.8'], '/etc/cert.pem', '/etc/key.pem') == {
        'ssl_certfile': '/etc/cert.pem',
        'ssl_keyfile': '/etc/key.pem',
    }
    assert _ssl_kwargs(False, ['192.168.0.8'], '/etc/cert.pem', '/etc/key.pem') == {}


def test_certificate_without_its_key_is_refused():
    with pytest.raises(ValueError, match='must be given together'):
        _ssl_kwargs(True, ['192.168.0.8'], '/etc/cert.pem', None)
