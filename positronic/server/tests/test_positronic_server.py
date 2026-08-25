import ipaddress
import re
import socket
import subprocess
from collections import namedtuple

import psutil
import pytest

from positronic.server.positronic_server import (
    _FALLBACK_CERTIFICATE_SUBJECT,
    _access_url,
    _generate_self_signed_cert,
    _insecure_context_warning,
    _served_addresses,
    _ssl_kwargs,
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


def _certificate_text(hosts: list[str], *fields: str) -> str:
    files = _generate_self_signed_cert(hosts)
    return subprocess.run(
        ['openssl', 'x509', '-in', files['ssl_certfile'], '-noout', *fields], check=True, capture_output=True, text=True
    ).stdout


def _certified_ips(extension: str) -> set[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    return {ipaddress.ip_address(address) for address in re.findall(r'IP Address:([0-9A-Fa-f.:]+)', extension)}


def test_wildcard_bind_serves_every_routable_local_address(multi_homed):
    assert _served_addresses('::') == ['127.0.0.1', '::1', '192.168.0.8', '198.51.100.7']


def test_an_ipv4_wildcard_advertises_no_address_its_listener_cannot_answer(multi_homed):
    assert _served_addresses('0.0.0.0') == ['127.0.0.1', '192.168.0.8', '198.51.100.7']
    assert '::1' in _served_addresses('::')


def test_every_spelling_of_the_wildcard_is_one(multi_homed):
    assert _served_addresses('0:0:0:0:0:0:0:0') == _served_addresses('::')
    assert _served_addresses('') == _served_addresses('0.0.0.0')


def test_a_socket_only_spelling_of_the_wildcard_is_one(multi_homed):
    assert _served_addresses('0') == _served_addresses('0.0.0.0')
    assert _served_addresses('0.0') == _served_addresses('0.0.0.0')
    assert _served_addresses('0.0.0') == _served_addresses('0.0.0.0')


def test_a_socket_only_spelling_that_is_not_the_wildcard_names_its_own_address(multi_homed):
    assert _served_addresses('127.1') == ['127.0.0.1']
    assert _served_addresses('1') == ['0.0.0.1']


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


def test_a_wildcard_bind_with_no_address_of_its_family_refuses_to_certify(monkeypatch):
    monkeypatch.setattr(psutil, 'net_if_addrs', lambda: {'lo': [_Addr(socket.AF_INET6, '::1', None, None, None)]})
    assert _served_addresses('0.0.0.0') == []

    with pytest.raises(ValueError, match='no local address'):
        _ssl_kwargs(True, [], None, None)


def test_a_wildcard_bind_with_no_address_of_its_family_still_serves_plain_http(monkeypatch):
    monkeypatch.setattr(psutil, 'net_if_addrs', lambda: {'lo': [_Addr(socket.AF_INET6, '::1', None, None, None)]})
    assert _ssl_kwargs(False, _served_addresses('0.0.0.0'), None, None) == {}
    assert _ssl_kwargs(True, [], '/etc/cert.pem', '/etc/key.pem') == {
        'ssl_certfile': '/etc/cert.pem',
        'ssl_keyfile': '/etc/key.pem',
    }


def test_certificate_names_every_served_address():
    hosts = ['198.51.100.7', '2001:db8::7']
    extension = _certificate_text(hosts, '-ext', 'subjectAltName')

    assert _certified_ips(extension) == {ipaddress.ip_address(host) for host in hosts}
    assert 'DNS:' not in extension


def test_certificate_names_localhost_beside_a_loopback_bind():
    extension = _certificate_text(['127.0.0.1', '::1'], '-ext', 'subjectAltName')

    assert _certified_ips(extension) == {ipaddress.ip_address('127.0.0.1'), ipaddress.ip_address('::1')}
    assert 'DNS:localhost' in extension


def test_certificate_names_a_host_name_as_a_dns_entry():
    extension = _certificate_text(['rig.local'], '-ext', 'subjectAltName')

    assert 'DNS:rig.local' in extension
    assert not _certified_ips(extension)


def test_certificate_drops_a_zone_from_an_ip_it_names():
    extension = _certificate_text(['fe80::1%eth0'], '-ext', 'subjectAltName')

    assert _certified_ips(extension) == {ipaddress.ip_address('fe80::1')}


def test_generated_certificate_subject_is_the_bind_address_when_it_fits():
    assert 'CN = 198.51.100.7' in _certificate_text(['198.51.100.7'], '-subject')

    host = 'b' * 52 + '.example.com'
    assert len(host.encode()) == 64
    assert f'CN = {host}' in _certificate_text([host], '-subject')


def test_a_bind_name_too_long_for_the_subject_field_still_certifies():
    host = 'a' * 60 + '.example.com'
    assert len(host.encode()) > 64
    text = _certificate_text([host], '-subject', '-ext', 'subjectAltName')

    assert f'CN = {_FALLBACK_CERTIFICATE_SUBJECT}' in text
    assert f'DNS:{host}' in text


def test_advertised_url_follows_the_bind_address():
    assert _access_url('http', '127.0.0.1', 8412) == 'http://127.0.0.1:8412'
    assert _access_url('https', '198.51.100.7', 8913) == 'https://198.51.100.7:8913'
    assert _access_url('https', 'rig.local', 8400) == 'https://rig.local:8400'


def test_advertised_url_brackets_an_ipv6_literal():
    assert _access_url('https', '::1', 8400) == 'https://[::1]:8400'


def test_advertised_url_encodes_a_zone_the_way_a_url_carries_one():
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
