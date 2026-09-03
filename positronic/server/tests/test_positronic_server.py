import hashlib
import ipaddress
import os
import re
import socket
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
from cryptography import x509
from cryptography.x509.oid import NameOID
from fastapi.testclient import TestClient

from positronic import keys
from positronic.dataset.episode import META_PATH, META_UID
from positronic.server import positronic_server
from positronic.server.positronic_server import (
    _MAX_COMPONENT_BYTES,
    _QUERY_DIGEST_CHARS,
    _access_url,
    _generate_self_signed_cert,
    _get_rrd_cache_path,
    _is_loopback,
    _path_component,
    _served_addresses,
    app,
    app_state,
    configure_pages,
    normalized_base_href,
    static_export_key,
    static_export_url_path,
    validated_build_id,
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
    def __init__(self, uid: str = 'ep-uid'):
        self._uid = uid

    def __getitem__(self, index):
        return SimpleNamespace(meta={META_UID: self._uid})


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


def test_a_uid_carrying_a_separator_stays_in_the_cache_directory(rrd_cache, monkeypatch):
    inside = rrd_cache(30.0, 640).parent
    monkeypatch.setitem(app_state, 'dataset', _OneEpisodeDataset('../../etc/ep-uid'))

    assert _get_rrd_cache_path(0, 30.0, 640).parent == inside


def test_two_uids_that_differ_reach_different_cached_rrds(rrd_cache, monkeypatch):
    def path_for(uid: str) -> Path:
        monkeypatch.setitem(app_state, 'dataset', _OneEpisodeDataset(uid))
        return _get_rrd_cache_path(0, 30.0, 640)

    assert path_for('camera/left') != path_for('camera_left')
    assert path_for('a%2Fb') != path_for('a/b')
    assert path_for('x' * 400) != path_for('y' * 400)


def test_a_path_component_is_one_short_injective_name():
    assert '/' not in _path_component('../../etc/passwd')
    assert os.sep not in _path_component('../../etc/passwd')
    assert _path_component('camera/left') != _path_component('camera_left')

    long_name = _path_component('x' * 4000)
    assert len(long_name.encode()) <= _MAX_COMPONENT_BYTES
    # A value short enough to survive encoding can never collide with a hashed one.
    assert _path_component(long_name) != long_name


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

    cache_path = rrd_cache(30.0, 640)
    assert not cache_path.exists()
    assert list(cache_path.parent.iterdir()) == []


# A URL that starts at the server root, in an attribute or in a script's string.
_ROOTED_URL = re.compile(r"""["'(`](/[^"'`)\s]*)""")


def _server_rooted_urls(html: str, base_href: str) -> list[str]:
    """Every URL the page pins to the server root, apart from the app-level assets and the base."""
    return [url for url in _ROOTED_URL.findall(html) if not url.startswith('/static/') and url != base_href]


class _StubEpisode:
    meta = {META_PATH: '/datasets/run-7/episode_3', 'size_mb': 12.5}
    static = {keys.TASK: 'pick the cube', 'scene': b'a mesh'}


class _StubDataset:
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> _StubEpisode:
        if index >= len(self):
            raise IndexError(index)
        return _StubEpisode()


@pytest.fixture
def viewer(monkeypatch):
    monkeypatch.setitem(app_state, 'dataset', _StubDataset())
    monkeypatch.setitem(app_state, 'loading_state', False)
    monkeypatch.setitem(app_state, 'root', '/datasets/run-7')
    monkeypatch.setitem(app_state, 'episode_table_cfg', {})
    monkeypatch.setitem(app_state, 'group_tables_cfg', {'leaderboard': None})
    return TestClient(app)


_PAGES = ['/', '/episodes', '/groups/leaderboard', '/episode/0']


@pytest.mark.parametrize('page', _PAGES)
def test_a_viewer_at_the_server_root_says_so(viewer, page):
    body = viewer.get(page).text

    assert '<base href="/" />' in body
    assert 'window.STATIC_EXPORT' not in body


@pytest.mark.parametrize('page', _PAGES)
def test_a_viewer_under_a_prefix_pins_no_page_link_to_the_server_root(viewer, monkeypatch, page):
    monkeypatch.setitem(app_state, 'base_href', '/v/tok/')

    body = viewer.get(page).text

    assert '<base href="/v/tok/" />' in body
    assert _server_rooted_urls(body, '/v/tok/') == []


def test_a_base_href_gains_the_trailing_slash_a_relative_link_needs(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'base_href', '/v/tok')

    assert '<base href="/v/tok/" />' in viewer.get('/').text


def test_a_static_export_tells_the_page_to_read_the_files_it_wrote(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'static_export', True)

    assert 'window.STATIC_EXPORT = true;' in viewer.get('/').text


def test_a_title_stands_in_for_the_dataset_root(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'title', 'Runway rollouts, 29 August')

    for page in ['/', '/episode/0']:
        body = viewer.get(page).text
        assert 'Runway rollouts, 29 August' in body


def test_the_header_falls_back_to_the_dataset_root(viewer):
    assert '/datasets/run-7' in viewer.get('/').text


def test_an_episode_page_hides_where_the_episode_lives(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'show_paths', False)

    body = viewer.get('/episode/0').text

    assert 'Episode path' not in body
    assert '/datasets/run-7' not in body


def test_an_episode_keeps_its_size_where_the_page_hides_paths(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'show_paths', False)

    assert '12.50 MB' in viewer.get('/episode/0').text


def test_two_exports_on_one_origin_keep_their_own_session_state():
    app_js = (Path(positronic_server.__file__).parent / 'static' / 'app.js').read_text()

    assert 'return `${name}:${document.baseURI}`' in app_js
    assert app_js.count("sessionStorage.setItem('") == 0


def test_an_episode_page_shows_where_the_episode_lives_by_default(viewer):
    body = viewer.get('/episode/0').text

    assert 'Episode path' in body
    assert '/datasets/run-7/episode_3' in body


def test_the_api_reports_no_dataset_root_when_the_viewer_hides_paths(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'show_paths', False)

    assert viewer.get('/api/dataset_info').json()['root'] == ''
    assert viewer.get('/api/dataset_status').json()['repo_id'] == ''


def test_the_api_reports_the_dataset_root_by_default(viewer):
    assert viewer.get('/api/dataset_info').json()['root'] == '/datasets/run-7'
    assert viewer.get('/api/dataset_status').json()['repo_id'] == '/datasets/run-7'


def test_a_build_id_carries_the_files_a_browser_caches_for_a_year(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'static_export', True)
    monkeypatch.setitem(app_state, 'build_id', 'k3n9')

    body = viewer.get('/episode/0').text

    assert 'build/k3n9/api/episode_rrd/0' in body
    assert 'build/k3n9/api/episode/0/static/scene' in body


def test_without_a_build_id_the_same_files_sit_beside_the_other_api_calls(viewer):
    body = viewer.get('/episode/0').text

    assert 'api/episode_rrd/0' in body
    assert 'api/episode/0/static/scene' in body
    assert 'build/' not in body


def test_a_live_viewer_asks_for_the_routes_it_serves_whatever_the_build_id_says(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'build_id', 'k3n9')

    body = viewer.get('/episode/0').text

    assert 'build/' not in body
    assert viewer.get('/api/episode/0/static/scene').content == b'a mesh'


def test_an_endpoint_the_export_writes_whole_takes_one_file():
    assert static_export_key('api/episodes') == 'api/episodes.json'
    assert static_export_key('api/dataset_info') == 'api/dataset_info.json'


def test_an_endpoint_the_export_writes_per_filter_set_names_an_empty_set_all():
    assert static_export_key('api/groups/leaderboard', {}) == 'api/groups/leaderboard/all.json'
    assert static_export_key('api/groups/leaderboard', {'model': ''}) == 'api/groups/leaderboard/all.json'


def test_one_filter_set_reaches_one_file_whatever_order_it_arrives_in():
    forwards = static_export_key('api/groups/g', {'model': 'pi0', 'task': 'pick a cube'})
    backwards = static_export_key('api/groups/g', {'task': 'pick a cube', 'model': 'pi0'})

    assert forwards == backwards == 'api/groups/g/model=pi0&task=pick%20a%20cube.json'


def test_a_filter_value_that_carries_a_separator_stays_in_one_component():
    key = static_export_key('api/groups/g', {'task': 'a&b=c/d'})

    assert key == 'api/groups/g/task=a%26b%3Dc%2Fd.json'


def test_a_name_that_carries_a_percent_escape_is_asked_for_with_that_escape_escaped():
    params = {'task': 'pick a cube'}

    assert static_export_key('api/groups/g', params) == 'api/groups/g/task=pick%20a%20cube.json'
    assert static_export_url_path('api/groups/g', params) == 'api/groups/g/task=pick%2520a%2520cube.json'


def test_a_name_that_needs_no_escape_is_asked_for_as_it_stands():
    for params in (None, {}, {'model': 'pi0'}):
        assert static_export_url_path('api/x', params) == static_export_key('api/x', params)


def test_a_filter_set_that_fits_a_file_name_keeps_its_readable_one():
    value = 'x' * (_MAX_COMPONENT_BYTES - len('model='))

    assert static_export_key('api/groups/g', {'model': value}) == f'api/groups/g/model={value}.json'


def test_a_filter_set_too_long_for_a_file_name_takes_a_digest_of_itself():
    value = 'x' * _MAX_COMPONENT_BYTES
    name = static_export_key('api/groups/g', {'model': value}).removeprefix('api/groups/g/').removesuffix('.json')

    assert name == hashlib.sha256(f'model={value}'.encode()).hexdigest()[:_QUERY_DIGEST_CHARS]
    assert len(name.encode()) <= _MAX_COMPONENT_BYTES


def test_two_filter_sets_past_the_budget_that_differ_reach_different_files():
    over = 'x' * _MAX_COMPONENT_BYTES

    assert static_export_key('api/groups/g', {'model': over}) != static_export_key('api/groups/g', {'task': over})


def test_app_js_bounds_a_file_name_by_the_same_numbers():
    app_js = (Path(positronic_server.__file__).parent / 'static' / 'app.js').read_text()

    assert f'const MAX_QUERY_BYTES = {_MAX_COMPONENT_BYTES};' in app_js
    assert f'const QUERY_DIGEST_CHARS = {_QUERY_DIGEST_CHARS};' in app_js


def test_a_base_href_that_starts_at_the_server_root_stands():
    assert normalized_base_href('/') == '/'
    assert normalized_base_href('/v/tok/') == '/v/tok/'
    assert normalized_base_href('/v/tok') == '/v/tok/'


def test_a_base_href_that_is_not_a_path_at_the_server_root_is_refused():
    outside = ('shares/run', '', 'https://app.example.com/v/tok/', '//other.example/', '/p?q/', '/p#f/')

    for value in outside:
        with pytest.raises(ValueError, match='server root'):
            normalized_base_href(value)


def test_a_base_href_a_browser_would_resolve_outside_the_prefix_is_refused():
    """A browser applies a dot segment and a backslash before it reads the `<base>`."""
    escaping = ('/v/../', '/v/%2e%2e/', '/v/%2E./tok/', '/v/./tok/', '/v/tok/..', '/v\\tok/')

    for value in escaping:
        with pytest.raises(ValueError, match='dot segment'):
            normalized_base_href(value)


def test_a_base_href_whose_segment_only_contains_dots_stands():
    for value in ('/v/tok.1/', '/v/..tok/', '/v/tok../', '/v/a.b.c/'):
        assert normalized_base_href(value) == value


def test_configure_pages_checks_what_it_is_given_and_fills_the_state(monkeypatch):
    for key in ('base_href', 'title', 'show_paths', 'static_export', 'build_id'):
        monkeypatch.setitem(app_state, key, app_state[key])

    configure_pages(base_href='/v/tok', title='A run', show_paths=False, static_export=True, build_id='k3n9')

    assert app_state['base_href'] == '/v/tok/'
    assert (app_state['title'], app_state['show_paths'], app_state['static_export']) == ('A run', False, True)
    assert app_state['build_id'] == 'k3n9'
    with pytest.raises(ValueError, match='build_id'):
        configure_pages(build_id='k3/n9')


def test_a_build_id_in_the_token_alphabet_stands():
    assert validated_build_id('') == ''
    assert validated_build_id('k3n9_-A') == 'k3n9_-A'


def test_a_build_id_that_would_cut_a_url_path_short_is_refused():
    for value in ('k3#n9', 'k3?n9', 'k3/n9', 'k3 n9'):
        with pytest.raises(ValueError, match='build_id'):
            validated_build_id(value)


def test_a_filter_key_outside_the_basic_plane_orders_by_its_encoding():
    # Python orders these two by code point and JavaScript by UTF-16 code unit, the other way
    # about; both languages order their percent-encoded ASCII alike.
    key = static_export_key('api/groups/g', {'\U0001f600': '1', '\ue000': '2'})

    assert key == 'api/groups/g/%EE%80%80=2&%F0%9F%98%80=1.json'


def test_the_rrd_path_reaches_the_page_script_as_json(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'static_export', True)
    monkeypatch.setitem(app_state, 'build_id', 'k3n9')

    assert 'appUrl("build/k3n9/api/episode_rrd/0")' in viewer.get('/episode/0').text


def test_a_group_name_reaches_the_page_script_as_json(viewer):
    body = viewer.get('/groups/a&b').text

    assert 'window.API_ENDPOINT = "api/groups/a\\u0026b";' in body
    assert 'a&amp;b' not in body


def test_app_js_asks_for_the_file_name_this_module_writes():
    app_js = (Path(positronic_server.__file__).parent / 'static' / 'app.js').read_text()

    assert static_export_key('api/groups/leaderboard', {'b': '2', 'a': 'x y'}) in app_js
