import ipaddress
import os
import re
import socket
import threading
from collections import namedtuple
from dataclasses import replace
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
    FILTER_VALUES,
    GROUP_FILTERS,
    ColumnConfig,
    GroupTableConfig,
    PageConfig,
    _access_url,
    _generate_self_signed_cert,
    _get_rrd_cache_path,
    _is_loopback,
    _path_component,
    _served_addresses,
    app,
    app_state,
    app_state_restored,
    configure_pages,
    configure_tables,
    download_link,
    download_paths,
    normalized_base_href,
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
    monkeypatch.setitem(app_state, 'pages', PageConfig(base_href='/v/tok/'))

    body = viewer.get(page).text

    assert '<base href="/v/tok/" />' in body
    assert _server_rooted_urls(body, '/v/tok/') == []


def test_a_base_href_gains_the_trailing_slash_a_relative_link_needs(viewer):
    with app_state_restored():
        configure_pages(base_href='/v/tok')

        assert '<base href="/v/tok/" />' in viewer.get('/').text


def test_a_static_export_tells_the_page_to_read_the_files_it_wrote(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'pages', PageConfig(static_export=True))

    assert 'window.STATIC_EXPORT = true;' in viewer.get('/').text


def test_a_title_stands_in_for_the_dataset_root(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'pages', PageConfig(title='Runway rollouts, 29 August'))

    for page in ['/', '/episode/0']:
        body = viewer.get(page).text
        assert 'Runway rollouts, 29 August' in body


def test_the_header_falls_back_to_the_dataset_root(viewer):
    assert '/datasets/run-7' in viewer.get('/').text


def test_an_episode_page_hides_where_the_episode_lives(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'pages', PageConfig(show_paths=False))

    body = viewer.get('/episode/0').text

    assert 'Episode path' not in body
    assert '/datasets/run-7' not in body


def test_an_episode_keeps_its_size_where_the_page_hides_paths(viewer, monkeypatch):
    monkeypatch.setitem(app_state, 'pages', PageConfig(show_paths=False))

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
    monkeypatch.setitem(app_state, 'pages', PageConfig(show_paths=False))

    assert viewer.get('/api/dataset_info').json()['root'] == ''
    assert viewer.get('/api/dataset_status').json()['repo_id'] == ''


def test_the_api_reports_the_dataset_root_by_default(viewer):
    assert viewer.get('/api/dataset_info').json()['root'] == '/datasets/run-7'
    assert viewer.get('/api/dataset_status').json()['repo_id'] == '/datasets/run-7'


def test_the_episode_page_links_its_recording_and_its_downloads_at_their_routes(viewer):
    body = viewer.get('/episode/0').text

    assert 'appUrl("api/episode_rrd/0")' in body
    assert '"api/episode/0/static/scene"' in body
    assert viewer.get('/api/episode/0/static/scene').content == b'a mesh'


def test_a_download_is_named_in_the_header_and_a_name_outside_ascii_is_percent_encoded(viewer, monkeypatch):
    monkeypatch.setattr(_StubEpisode, 'static', {'scene': b'a mesh', 'notes & é': 'n' * 2000})

    assert viewer.get('/api/episode/0/static/scene').headers['content-disposition'] == 'attachment; filename="scene"'
    assert viewer.get('/api/episode/0/static/notes & é').headers['content-disposition'] == (
        "inline; filename*=utf-8''notes%20%26%20%C3%A9.txt"
    )


def test_a_download_link_carries_each_key_encoded_and_the_route_answers_it(viewer, monkeypatch):
    monkeypatch.setattr(_StubEpisode, 'static', {'a?b c': b'a mesh', 'a/b': b'nested'})

    link, slashed = download_link(0, ('a?b c',)), download_link(0, ('a/b',))

    assert (link, slashed) == ('api/episode/0/static/a%3Fb%20c', 'api/episode/0/static/a%2Fb')
    assert viewer.get(f'/{link}').content == b'a mesh' and viewer.get(f'/{slashed}').content == b'nested'
    assert f'"{link}"' in viewer.get('/episode/0').text


def test_a_nested_value_and_a_dotted_top_level_key_get_apart_links_the_route_tells_apart(viewer, monkeypatch):
    monkeypatch.setattr(_StubEpisode, 'static', {'scene.mesh': b'top', 'scene': {'mesh': b'nested'}})

    top, nested = download_link(0, ('scene.mesh',)), download_link(0, ('scene', 'mesh'))

    assert (top, nested) == ('api/episode/0/static/scene.mesh', 'api/episode/0/static/scene/mesh')
    assert viewer.get(f'/{top}').content == b'top'
    assert viewer.get(f'/{nested}').content == b'nested'


def test_a_key_a_browser_rewrites_in_a_path_gets_no_link():
    for keys_ in (('.',), ('..',), ('a', '..', 'b'), ('a', '', 'b')):
        with pytest.raises(ValueError, match='no link'):
            download_link(0, keys_)


def test_reconfiguring_the_tables_drops_a_cached_table_response(grouped):
    before = grouped.get('/api/groups/by_task').json()
    renamed = replace(
        _BY_TASK, format_table={keys.TASK: ColumnConfig(label='Job'), 'count': ColumnConfig(label='Episodes')}
    )

    ep_table_cfg = {keys.TASK: ColumnConfig(label='Task'), ASSISTED: ColumnConfig(label='Assisted')}
    with app_state_restored():
        configure_tables(
            root='',
            cache_dir=Path(),
            ep_table_cfg=ep_table_cfg,
            group_tables={'by_task': renamed},
            home_page=None,
            max_resolution=64,
            max_hz=0,
        )
        after = grouped.get('/api/groups/by_task').json()

    assert [column['label'] for column in before['columns']] == ['Task', 'Episodes']
    assert [column['label'] for column in after['columns']] == ['Job', 'Episodes']


def test_every_static_value_a_page_links_as_a_download_is_named_by_its_field_path():
    static = {
        'scene': b'a mesh',
        'notes': 'n' * 2000,
        'short': 'ok',
        'nested': {'blob': b'x', 'n': 1},
        'many': [{'a': b'x'}, b'y'],
    }

    assert list(download_paths(static)) == [
        ('scene',),
        ('notes',),
        ('nested', 'blob'),
        ('many', '0', 'a'),
        ('many', '1'),
    ]


def test_a_static_value_whose_key_a_browser_rewrites_in_a_path_gets_no_link():
    for static in ({'': b'x'}, {'a': {'': b'x'}}, {'..': b'x'}, {'a': {'.': b'x'}}):
        with pytest.raises(ValueError, match='no link'):
            list(download_paths(static))


def test_a_key_past_a_file_name_s_limit_once_encoded_gets_no_link():
    for key in ('x' * (_MAX_COMPONENT_BYTES + 1), 'é' * (_MAX_COMPONENT_BYTES // 6 + 1)):  # `é` encodes to 6 bytes
        with pytest.raises(ValueError, match='no link'):
            list(download_paths({key: b'x'}))


def test_a_key_within_a_file_name_s_limit_once_encoded_keeps_its_link():
    key = 'x' * _MAX_COMPONENT_BYTES

    assert list(download_paths({key: b'x'})) == [(key,)]
    assert download_link(0, (key,)) == f'api/episode/0/static/{key}'


def test_a_dotted_key_and_a_nested_value_that_spell_alike_each_keep_their_own_link():
    assert list(download_paths({'a.b': b'1', 'a': {'b': b'2'}})) == [('a.b',), ('a', 'b')]


def test_a_base_href_that_starts_at_the_server_root_stands():
    assert normalized_base_href('/') == '/'
    assert normalized_base_href('/v//tok/') == '/v//tok/'
    assert normalized_base_href('/v/tok/') == '/v/tok/'
    assert normalized_base_href('/v/tok') == '/v/tok/'


def test_a_base_href_that_is_not_a_path_at_the_server_root_is_refused():
    outside = (
        'shares/run',
        '',
        'https://app.example.com/v/tok/',
        '//other.example/',
        '////other.example/',
        '/p?q/',
        '/p#f/',
    )

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
    monkeypatch.setitem(app_state, 'pages', app_state['pages'])

    configure_pages(base_href='/v/tok', title='A run', show_paths=False, static_export=True)

    assert app_state['pages'] == PageConfig(base_href='/v/tok/', title='A run', show_paths=False, static_export=True)
    with pytest.raises(ValueError, match='server root'):
        configure_pages(base_href='v/tok')


def test_a_group_name_reaches_the_page_script_as_json(viewer):
    body = viewer.get('/groups/a&b').text

    assert 'window.API_ENDPOINT = "api/groups/a\\u0026b";' in body
    assert 'a&amp;b' not in body


class _Statics:
    """A dataset of episodes that hold static values and nothing else."""

    def __init__(self, *statics: dict):
        self._statics = statics

    def __len__(self) -> int:
        return len(self._statics)

    def __getitem__(self, index: int) -> SimpleNamespace:
        if index >= len(self):
            raise IndexError(index)
        return SimpleNamespace(static=self._statics[index], meta={}, duration_ns=0)


ASSISTED = 'assisted'
_BY_TASK = GroupTableConfig(
    group_keys=keys.TASK,
    group_fn=lambda episodes: {'count': len(episodes)},
    format_table={keys.TASK: ColumnConfig(label='Task'), 'count': ColumnConfig(label='Episodes')},
    group_filter_keys={ASSISTED: 'Assisted'},
)


@pytest.fixture
def grouped(monkeypatch):
    episodes = _Statics(
        {keys.TASK: 'fold', ASSISTED: True},
        {keys.TASK: 'fold', ASSISTED: False},
        {keys.TASK: 'stack', ASSISTED: False},
        {keys.TASK: 'stack'},
    )
    monkeypatch.setitem(app_state, 'dataset', episodes)
    monkeypatch.setitem(app_state, 'loading_state', False)
    monkeypatch.setitem(app_state, 'group_tables_cfg', {'by_task': _BY_TASK})
    monkeypatch.setattr(positronic_server, '_api_cache', {})
    return TestClient(app)


def test_a_group_filter_offers_each_value_as_the_string_a_query_carries_and_no_absent_one(grouped):
    table = grouped.get('/api/groups/by_task').json()

    assert table[GROUP_FILTERS][ASSISTED][FILTER_VALUES] == ['False', 'True']


def test_a_group_filter_matches_the_value_it_offered(grouped):
    assisted = grouped.get('/api/groups/by_task', params={ASSISTED: 'True'}).json()
    unassisted = grouped.get('/api/groups/by_task', params={ASSISTED: 'False'}).json()

    assert [row[1][0] for row in assisted['episodes']] == ['fold']
    assert [row[1][0] for row in unassisted['episodes']] == ['fold', 'stack']


def test_a_second_holder_of_the_app_state_waits_for_the_first():
    first_holds, released, second_holds = threading.Event(), threading.Event(), threading.Event()

    def first():
        with app_state_restored():
            first_holds.set()
            released.wait(5)

    def second():
        with app_state_restored():
            second_holds.set()

    holders = [threading.Thread(target=first), threading.Thread(target=second)]
    holders[0].start()
    assert first_holds.wait(5)
    holders[1].start()

    assert not second_holds.wait(0.2)
    released.set()
    for holder in holders:
        holder.join(5)
    assert second_holds.is_set()


def _configure(ep_table_cfg, group_tables):
    configure_tables(
        root='',
        cache_dir=Path(),
        ep_table_cfg=ep_table_cfg,
        group_tables=group_tables,
        home_page=None,
        max_resolution=64,
        max_hz=0,
    )


def test_a_group_key_that_is_no_episode_column_is_refused():
    ep_table_cfg = {keys.TASK: ColumnConfig(label='Task')}
    group_tables = {'by_object': replace(_BY_TASK, group_keys='object', group_filter_keys={})}
    with app_state_restored(), pytest.raises(ValueError, match='no column of the episode table'):
        _configure(ep_table_cfg, group_tables)


def test_a_group_filter_key_that_is_no_episode_column_is_refused():
    ep_table_cfg = {keys.TASK: ColumnConfig(label='Task')}
    with app_state_restored(), pytest.raises(ValueError, match='no column of the episode table'):
        _configure(ep_table_cfg, {'by_task': _BY_TASK})  # _BY_TASK filters on ASSISTED, which is no column here


def test_group_keys_that_are_episode_columns_are_accepted():
    ep_table_cfg = {keys.TASK: ColumnConfig(label='Task'), ASSISTED: ColumnConfig(label='Assisted')}
    with app_state_restored():
        _configure(ep_table_cfg, {'by_task': _BY_TASK})  # the group key and the filter key are both episode columns


def test_the_flat_table_applies_a_url_key_that_names_any_episode_column():
    """A View link carries a group key; the flat table filters on it when it is a column, not only a filter one."""
    app_js = (Path(positronic_server.__file__).parent / 'static' / 'app.js').read_text()

    assert 'function readFiltersFromURL(serverFilterKeys, columns, episodes)' in app_js
    assert 'const index = columns.findIndex((c) => c.key === key);' in app_js
    assert 'state.filters[key] = value;' in app_js


def test_the_flat_table_compares_a_url_filter_with_a_cell_s_raw_value():
    """A formatted cell is `[raw, formatted]`, and a View link carries the raw value as the server spells it."""
    app_js = (Path(positronic_server.__file__).parent / 'static' / 'app.js').read_text()

    assert 'return String(rawValue(episodeData[colIdx])) === value;' in app_js
    assert 'const v = rawValue(episodeData[index]);' in app_js
