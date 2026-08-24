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
    _certificate_subject,
    _generate_self_signed_cert,
    _insecure_context_warning,
    _served_addresses,
    _ssl_kwargs,
    _subject_alt_names,
)

_Addr = namedtuple('_Addr', 'family address netmask broadcast ptp')

# A multi-homed host: loopback, a LAN interface with a link-local companion and a MAC, and a VPN.
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


def test_wildcard_bind_serves_every_routable_local_address(multi_homed):
    assert _served_addresses('::') == ['127.0.0.1', '::1', '192.168.0.8', '198.51.100.7']


def test_an_ipv4_wildcard_advertises_no_address_its_listener_cannot_answer(multi_homed):
    # `0.0.0.0` binds AF_INET, so a v6 URL or SAN built from it names an address nothing serves.
    assert _served_addresses('0.0.0.0') == ['127.0.0.1', '192.168.0.8', '198.51.100.7']
    assert '::1' in _served_addresses('::')


def test_every_spelling_of_the_wildcard_is_one(multi_homed):
    # The bind normalizes the address, so two spellings of one wildcard serve the same set.
    assert _served_addresses('0:0:0:0:0:0:0:0') == _served_addresses('::')
    assert _served_addresses('') == _served_addresses('0.0.0.0')


def test_a_socket_only_spelling_of_the_wildcard_is_one(multi_homed):
    # `socket` takes the legacy short IPv4 forms `ipaddress` refuses, and the bind resolves each of
    # these to the wildcard — so a set derived from them must be the wildcard's, not the string's.
    assert _served_addresses('0') == _served_addresses('0.0.0.0')
    assert _served_addresses('0.0') == _served_addresses('0.0.0.0')
    assert _served_addresses('0.0.0') == _served_addresses('0.0.0.0')


def test_a_socket_only_spelling_that_is_not_the_wildcard_names_its_own_address(multi_homed):
    # The same parser normalizes a concrete short form, which stays the one address it binds.
    assert _served_addresses('127.1') == ['127.0.0.1']
    assert _served_addresses('1') == ['0.0.0.1']
    # A name the resolver would happily turn into an address stays a name: the certificate names it
    # with `DNS:` and the URL keeps it, so normalizing must go no further than the numeric spellings.
    assert _served_addresses('localhost') == ['localhost']
    assert 'IP:' not in _subject_alt_names(['localhost'])


def test_a_wildcard_serves_an_ipv4_link_local_address(monkeypatch):
    # 169.254/16 carries no zone, so the filter that drops a v6 link-local must not reach it.
    monkeypatch.setattr(
        psutil, 'net_if_addrs', lambda: {'eth0': [_Addr(socket.AF_INET, '169.254.10.2', None, None, None)]}
    )
    assert _served_addresses('0.0.0.0') == ['169.254.10.2']


def test_concrete_bind_serves_only_the_address_it_binds(multi_homed):
    assert _served_addresses('198.51.100.7') == ['198.51.100.7']
    assert _served_addresses('127.0.0.1') == ['127.0.0.1']
    assert _served_addresses('rig.local') == ['rig.local']


def test_a_wildcard_bind_with_no_address_of_its_family_refuses_to_certify(monkeypatch):
    # The one interface carries no AF_INET address, so an IPv4 wildcard scans to nothing.
    monkeypatch.setattr(psutil, 'net_if_addrs', lambda: {'lo': [_Addr(socket.AF_INET6, '::1', None, None, None)]})
    assert _served_addresses('0.0.0.0') == []

    with pytest.raises(ValueError, match='no local address'):
        _ssl_kwargs(True, [], None, None)


def test_a_wildcard_bind_with_no_address_of_its_family_still_serves_plain_http(monkeypatch):
    # Only the certificate needs an address to name. The listener answers on the wildcard either
    # way, so the refusal belongs to the certificate path and must not reach the bind itself.
    monkeypatch.setattr(psutil, 'net_if_addrs', lambda: {'lo': [_Addr(socket.AF_INET6, '::1', None, None, None)]})
    assert _ssl_kwargs(False, _served_addresses('0.0.0.0'), None, None) == {}
    assert _ssl_kwargs(True, [], '/etc/cert.pem', '/etc/key.pem') == {
        'ssl_certfile': '/etc/cert.pem',
        'ssl_keyfile': '/etc/key.pem',
    }


def test_certificate_names_every_served_address():
    assert _subject_alt_names(['127.0.0.1', '::1', '192.168.0.8']) == 'IP:127.0.0.1,IP:::1,IP:192.168.0.8,DNS:localhost'


def test_certificate_names_no_address_beyond_the_bind():
    assert _subject_alt_names(['198.51.100.7']) == 'IP:198.51.100.7'
    assert _subject_alt_names(['rig.local']) == 'DNS:rig.local'


def test_certificate_drops_a_zone_from_an_ip_it_names():
    assert _subject_alt_names(['fe80::1%eth0']) == 'IP:fe80::1'


def test_generated_certificate_carries_the_bind_addresses_and_no_others():
    hosts = ['198.51.100.7', '2001:db8::7']
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


def test_generated_certificate_subject_is_the_bind_address_when_it_fits():
    assert _certificate_subject(['198.51.100.7']) == '198.51.100.7'
    assert _certificate_subject(['b' * 52 + '.example.com']) == 'b' * 52 + '.example.com'  # exactly 64 bytes


def test_a_bind_name_too_long_for_the_subject_field_still_certifies():
    host = 'a' * 60 + '.example.com'
    assert len(host.encode()) > 64
    assert _certificate_subject([host]) == _FALLBACK_CERTIFICATE_SUBJECT

    files = _generate_self_signed_cert([host])
    text = subprocess.run(
        ['openssl', 'x509', '-in', files['ssl_certfile'], '-noout', '-subject', '-ext', 'subjectAltName'],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    assert _FALLBACK_CERTIFICATE_SUBJECT in text
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
