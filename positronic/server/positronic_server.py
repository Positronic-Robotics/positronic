"""A FastAPI web server for visualizing Positronic LocalDatasets using Rerun."""

import atexit
import hashlib
import ipaddress
import logging
import os
import shutil
import socket
import subprocess
import tempfile
import threading
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, unquote, urlsplit

import configuronic as cfn
import pos3
import psutil
import rerun as rr
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import positronic.cfg.ds
from pimm.logging import init_logging
from positronic import keys
from positronic.dataset import CachedDataset, Dataset, Episode
from positronic.dataset.episode import META_PATH, META_UID
from positronic.dataset.local_dataset import LocalDataset
from positronic.server.dataset_utils import (
    DEFAULT_MAX_HZ,
    DEFAULT_MAX_RESOLUTION,
    get_dataset_root,
    get_episodes_list,
    stream_episode_rrd,
)

# Response cache for api_groups and api_episodes (dataset is immutable once loaded)
_api_cache: dict[tuple, dict] = {}

# Global app state
app_state: dict[str, object] = {
    'dataset': None,
    'loading_state': True,
    'root': '',
    'cache_dir': '',
    'episode_keys': {},
    'max_resolution': DEFAULT_MAX_RESOLUTION,
    'max_hz': DEFAULT_MAX_HZ,
    'group_tables_cfg': {},
    'home_page': None,  # None = episodes, or group name like 'tasks'
    'base_href': '/',
    'title': '',  # empty = the dataset root
    'show_paths': True,
    'static_export': False,
}


def _pkg_path(*parts: str) -> str:
    return str(Path(__file__).resolve().parent.joinpath(*parts))


def require_dataset(func):
    """Decorator that checks if dataset is loaded before executing the endpoint."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        if app_state['loading_state']:
            raise HTTPException(status_code=202, detail='Dataset is loading...')
        ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
        if ds is None:
            raise HTTPException(status_code=500, detail='Dataset failed to load')
        return await func(*args, **kwargs)

    return wrapper


_MAX_COMPONENT_BYTES = 200


def _path_component(value: str) -> str:
    """``value`` as one filename component, injectively and within any filesystem's name limit."""
    encoded = quote(value, safe='')
    if len(encoded.encode()) <= _MAX_COMPONENT_BYTES:
        return encoded
    # A digest is never read back as an encoded value: `quote` escapes '='.
    return '=' + hashlib.sha256(value.encode()).hexdigest()


def _get_rrd_cache_path(episode_id: int, max_hz: float, max_resolution: int) -> Path:
    ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
    if ds is None:
        raise RuntimeError('Dataset not loaded')
    ds_id = _path_component(str(Path(str(app_state['root'])).resolve()))
    episode_cache_dir = Path(str(app_state['cache_dir'])) / ds_id
    episode_cache_dir.mkdir(parents=True, exist_ok=True)
    # The uid, because an episode's position is view-dependent.
    uid = _path_component(str(cast(Episode, ds[episode_id]).meta[META_UID]))
    return episode_cache_dir / f'{uid}-{max_hz!r}hz-{max_resolution}px.rrd'


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    pass


app = FastAPI(lifespan=lifespan)

# Static files and templates (packaged relative to this file)
_static_dir = _pkg_path('static')
_templates_dir = _pkg_path('templates')
app.mount('/static', StaticFiles(directory=_static_dir), name='static')
templates = Jinja2Templates(directory=_templates_dir)


@app.middleware('http')
async def cache_rerun_assets(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith('/static/rerun/'):
        response.headers['Cache-Control'] = 'public, max-age=31536000, immutable'
    return response


# The field a group table's response carries its filters under, and the field of each filter
# holding its values. The export reads them back to write one file per filter set.
GROUP_FILTERS = 'group_filters'
FILTER_VALUES = 'values'


def _shown_root() -> str:
    """The dataset root the pages report; empty when the viewer hides paths."""
    return str(app_state['root']) if app_state['show_paths'] else ''


def _page_context() -> dict[str, Any]:
    """The template context every page shares."""
    home_page = app_state.get('home_page')
    group_tables = app_state.get('group_tables_cfg', {})

    nav_items = []
    episodes_url = 'episodes' if home_page else '.'

    # If home_page is set, it goes first
    if home_page and home_page in group_tables:
        label = home_page.replace('_', ' ').title()
        nav_items.append({'name': home_page, 'url': '.', 'label': label})

    # Other group links
    for group_name in group_tables.keys():
        if group_name == home_page:
            continue  # Already added as home
        url = f'groups/{group_name}'
        label = group_name.replace('_', ' ').title()
        nav_items.append({'name': group_name, 'url': url, 'label': label})

    # Episodes link last (unless it's the home page)
    if home_page:
        nav_items.append({'name': 'episodes', 'url': episodes_url, 'label': 'Episodes'})
    else:
        # Episodes is home, insert at beginning
        nav_items.insert(0, {'name': 'episodes', 'url': '.', 'label': 'Episodes'})

    return {
        'nav_items': nav_items,
        'home_page': home_page,
        'episodes_url': episodes_url,
        'base_href': normalized_base_href(str(app_state['base_href'])),
        'title': str(app_state['title']) if app_state['title'] else _shown_root(),
        'show_paths': app_state['show_paths'],
        'static_export': app_state['static_export'],
    }


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    home_page = app_state.get('home_page')
    if home_page:
        # Render group view as home page
        return templates.TemplateResponse(
            request,
            'grouped.html',
            {'api_endpoint': f'api/groups/{home_page}', **_page_context(), 'current_page': home_page},
        )
    # Default: render episodes
    return templates.TemplateResponse(request, 'index.html', {**_page_context(), 'current_page': 'episodes'})


@app.get('/episodes', response_class=HTMLResponse)
async def episodes_view(request: Request):
    """Episodes list view (used when home_page is set to a group)."""
    return templates.TemplateResponse(request, 'index.html', {**_page_context(), 'current_page': 'episodes'})


def normalized_base_href(value: str) -> str:
    """`value` with the trailing slash a relative link needs.

    Only a path at the server root: a relative one, a netloc, a query or a fragment each send a
    relative link somewhere other than the prefix. A browser resolves a dot segment and a backslash
    before it reads the `<base>`, so a path holding one names a different prefix than it shows.
    """
    parts = urlsplit(value)
    if parts.scheme or parts.netloc or parts.query or parts.fragment or not parts.path.startswith('/'):
        raise ValueError(f'base_href must be a path at the server root and nothing else, got {value!r}')
    if '\\' in parts.path or any(unquote(segment) in ('.', '..') for segment in parts.path.split('/')):
        raise ValueError(f'base_href must hold no dot segment and no backslash, got {value!r}')
    return parts.path if parts.path.endswith('/') else parts.path + '/'


def configure_pages(
    *, base_href: str = '/', title: str = '', show_paths: bool = True, static_export: bool = False
) -> None:
    """Set where the pages are served and what they show; see `main` for what each value does."""
    app_state['base_href'] = normalized_base_href(base_href)
    app_state['title'] = title
    app_state['show_paths'] = show_paths
    app_state['static_export'] = static_export


def install_dataset(dataset: Dataset) -> None:
    """Serve `dataset` from now on."""
    app_state['dataset'] = dataset
    _api_cache.clear()
    app_state['loading_state'] = False


@contextmanager
def app_state_restored() -> Iterator[None]:
    """Give the app its state back on exit, for a caller that composes the app in its own process."""
    saved = dict(app_state)
    try:
        yield
    finally:
        app_state.clear()
        app_state.update(saved)
        _api_cache.clear()


def is_download(value: object) -> bool:
    """Whether a static value reaches a page as a download link rather than inline."""
    return isinstance(value, bytes) or (isinstance(value, str) and len(value) > 1024)


def download_paths(static: Mapping[str, Any], path: str = '') -> Iterator[str]:
    """The field path of every static value an episode page links as a download."""
    for key, value in static.items():
        field_path = f'{path}.{key}' if path else key
        if is_download(value):
            yield field_path
        elif isinstance(value, dict):
            yield from download_paths(value, field_path)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield from download_paths(item, field_path)
                elif is_download(item):
                    yield field_path


@app.get('/episode/{episode_id}', response_class=HTMLResponse)
@require_dataset
async def episode_viewer(request: Request, episode_id: int):
    ds = app_state.get('dataset')

    try:
        episode = ds[episode_id]
    except IndexError as e:
        raise HTTPException(status_code=404, detail='Episode not found') from e

    meta = episode.meta
    size_mb = meta.get('size_mb')
    size_mb_display = f'{size_mb:.2f}' if isinstance(size_mb, int | float) else None

    def _make_serializable(obj, path=''):
        if is_download(obj):
            return {
                '__download__': f'api/episode/{episode_id}/static/{path}',
                'size': len(obj),
                'type': 'bytes' if isinstance(obj, bytes) else 'text',
            }
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: _make_serializable(v, f'{path}.{k}' if path else k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v, path) for v in obj]
        return obj

    return templates.TemplateResponse(
        request,
        'episode.html',
        {
            'episode_id': episode_id,
            'num_episodes': len(ds),
            'viewer_path': f'/static/rerun/{rr.__version__}/index.html',
            'task': episode.static.get(keys.TASK, None),
            'rrd_path': f'api/episode_rrd/{episode_id}',
            'episode_path': meta.get(META_PATH),
            'episode_size_mb': size_mb_display,
            'static_data': _make_serializable(episode.static),
            **_page_context(),
        },
    )


@app.get('/api/dataset_info')
@require_dataset
async def api_dataset_info():
    ds = cast(Dataset, app_state['dataset'])
    return {'root': _shown_root(), 'num_episodes': len(ds)}


@dataclass
class RendererConfig:
    """Custom cell renderer sent to the frontend.

    Attributes:
        type: Renderer type dispatched in app.js — ``'badge'`` or ``'icon'``.
        options: Value-keyed mapping. For badges: ``{value: {'label', 'variant'}}``.
            For icons: ``{value: {'src', 'label?', 'tags?', 'class?', 'href?'}}``.
            Icon entries may include:
              - ``tags``: list of string labels rendered as chips below the name.
              - ``class``: CSS class added to the cell element (useful for row-level
                styling via ``tr:has(.my-class)``).
              - ``href``: makes the name a link. ``{value}`` in the string is replaced
                with the URL-encoded raw cell value.
            The reserved key ``'_tagStyles'`` maps tag names to style dicts
            ``{'bg', 'color', 'border'}`` applied as inline styles on tag chips.
    """

    type: str
    options: dict = field(default_factory=dict)


@dataclass
class ColumnConfig:
    """Column definition for an episodes or grouped table.

    Attributes:
        label: Header text shown in the table.
        format: Printf-style format string, e.g. ``'%.1f'``, ``'%Y-%m-%d %H:%M'``.
        default: Fallback value when the cell is None/missing.
        renderer: Optional custom renderer (badge, icon, …).
        filter: Whether to show a client-side filter dropdown for this column.
    """

    label: str
    subtitle: str | None = None
    format: str | None = None
    default: Any = None
    renderer: RendererConfig | None = None
    filter: bool = False
    align: str | None = None
    sortable: bool = True


@dataclass
class SortConfig:
    """Default sort order for a table.

    Attributes:
        column: Column key to sort by.
        direction: ``'asc'`` or ``'desc'``.
    """

    column: str
    direction: str = 'desc'


TableConfig = dict[str, ColumnConfig]


@dataclass
class GroupTableConfig:
    """Configuration for a grouped table view (e.g. leaderboard).

    Attributes:
        group_keys: Column key(s) to group episodes by. A single string or tuple of strings.
        group_fn: Aggregation function that takes a list of episodes and returns a row dict.
        format_table: Column definitions for the output table.
        group_filter_keys: Server-side filter dropdowns — maps column key to display label.
        default_sort: Initial sort order applied on first page load.
    """

    group_keys: str | tuple[str, ...]
    group_fn: Any  # Callable[[list[Episode]], dict]
    format_table: TableConfig
    group_filter_keys: dict[str, str] = field(default_factory=dict)
    default_sort: SortConfig | None = None


def parse_table_cfg(table_cfg: TableConfig) -> tuple:
    columns = []
    formatters = {}
    defaults = {}
    for key, cfg in table_cfg.items():
        column: dict[str, Any] = {'key': key, 'label': cfg.label}
        formatters[key] = cfg.format
        defaults[key] = cfg.default

        if cfg.subtitle:
            column['subtitle'] = cfg.subtitle

        if cfg.align:
            column['align'] = cfg.align

        if cfg.renderer:
            column['renderer'] = asdict(cfg.renderer)

        if cfg.filter:
            column['filter'] = cfg.filter

        if not cfg.sortable:
            column['sortable'] = False

        columns.append(column)
    return columns, formatters, defaults


@app.get('/api/episodes')
@require_dataset
async def api_episodes(request: Request):
    cache_key = ('episodes', tuple(sorted(request.query_params.items())))
    if cache_key in _api_cache:
        return _api_cache[cache_key]

    ds = app_state.get('dataset')
    config = app_state['episode_table_cfg']
    columns, formatters, defaults = parse_table_cfg(config)
    filters = {k: v for k, v in request.query_params.items() if v}

    def matches(ep: Episode) -> bool:
        return filters is None or all(str(ep.static.get(k)) == v for k, v in filters.items())

    ep_it = (
        {'__episode_index__': i, '__meta__': ep.meta, '__duration__': ep.duration_ns / 1e9, **ep.static}
        for i, ep in enumerate(ds)
        if matches(ep)
    )
    episodes = get_episodes_list(ep_it, config.keys(), formatters=formatters, defaults=defaults)
    result = {'columns': columns, 'episodes': episodes}
    _api_cache[cache_key] = result
    return result


def _group_id(episode: Episode, group_keys: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(episode.static.get(k) for k in group_keys)


@app.get('/api/groups/{suffix}')
@require_dataset
async def api_groups(request: Request, suffix: str):  # noqa: C901
    cache_key = ('groups', suffix, tuple(sorted(request.query_params.items())))
    if cache_key in _api_cache:
        return _api_cache[cache_key]

    ds = app_state.get('dataset')
    group_tables = app_state.get('group_tables_cfg', {})
    if not isinstance(group_tables, dict) or suffix not in group_tables:
        raise HTTPException(
            status_code=404,
            detail=f'Group configuration "{suffix}" not found, available: {", ".join(group_tables.keys())}',
        )

    cfg = group_tables[suffix]
    group_keys = (cfg.group_keys,) if isinstance(cfg.group_keys, str) else cfg.group_keys

    # Ensure group keys are visible in the output table
    for k in group_keys:
        assert k in cfg.format_table, f'Group key {k} not found in format_table'

    columns, formatters, defaults = parse_table_cfg(cfg.format_table)

    # Take only those query parameters that are in group_filter_keys
    active_filters = {}
    for filter_key in cfg.group_filter_keys:
        filter_value = request.query_params.get(filter_key)
        if filter_value:
            active_filters[filter_key] = filter_value

    groups = defaultdict(list)
    group_filters = {key: {'label': label or key, FILTER_VALUES: set()} for key, label in cfg.group_filter_keys.items()}
    for episode in ds:
        # Always collect all filter values regardless of active filters
        for filter_key in cfg.group_filter_keys:
            group_filters[filter_key][FILTER_VALUES].add(episode.static.get(filter_key))
        # Apply filters for grouping
        match = all(episode.static[key] == value for key, value in active_filters.items())
        if match:
            groups[_group_id(episode, group_keys)].append(episode)

    rows = []
    for group_id, episodes in groups.items():
        key_fields = {k: group_id[i] for i, k in enumerate(group_keys)}
        rows.append({**key_fields, '__meta__': {'group': key_fields}, **cfg.group_fn(episodes)})

    episodes = get_episodes_list(rows, cfg.format_table.keys(), formatters=formatters, defaults=defaults)
    result = {'columns': columns, 'episodes': episodes, GROUP_FILTERS: group_filters}
    if cfg.default_sort:
        result['default_sort'] = asdict(cfg.default_sort)
    _api_cache[cache_key] = result
    return result


@app.get('/groups/{suffix}', response_class=HTMLResponse)
async def grouped_view(request: Request, suffix: str):
    return templates.TemplateResponse(
        request, 'grouped.html', {'api_endpoint': f'api/groups/{suffix}', **_page_context(), 'current_page': suffix}
    )


@app.get('/api/dataset_status')
async def api_dataset_status():
    return {
        'loading': app_state['loading_state'],
        'loaded': app_state.get('dataset', None) is not None,
        'repo_id': _shown_root(),
    }


@app.get('/api/episode/{episode_id}/static/{field_path:path}')
@require_dataset
async def api_episode_static_field(episode_id: int, field_path: str):
    ds = app_state.get('dataset')
    try:
        episode = ds[episode_id]
    except IndexError as e:
        raise HTTPException(status_code=404, detail='Episode not found') from e

    # Navigate dotted path, greedily matching dict keys (handles keys with dots like "link0.stl")
    value = episode.static
    remaining = field_path
    while remaining:
        if not isinstance(value, dict):
            raise HTTPException(status_code=404, detail=f'Field not found: {field_path}')
        # Try the full remaining path first, then progressively shorter prefixes
        parts = remaining.split('.')
        matched = False
        for i in range(len(parts), 0, -1):
            key = '.'.join(parts[:i])
            if key in value:
                value = value[key]
                remaining = '.'.join(parts[i:])
                matched = True
                break
        if not matched:
            raise HTTPException(status_code=404, detail=f'Field not found: {field_path}')

    filename = field_path.rsplit('.', 1)[-1] if '.' in field_path else field_path
    if isinstance(value, bytes):
        return Response(
            content=value,
            media_type='application/octet-stream',
            headers={'Content-Disposition': f'attachment; filename={filename}'},
        )
    if isinstance(value, str):
        return Response(
            content=value.encode(),
            media_type='text/plain; charset=utf-8',
            headers={'Content-Disposition': f'inline; filename={filename}.txt'},
        )
    raise HTTPException(status_code=400, detail=f'Field {field_path} is not downloadable')


@app.get('/api/episode_rrd/{episode_id}')
@require_dataset
async def api_episode_rrd(episode_id: int):
    ds = cast(Dataset, app_state['dataset'])
    max_hz = cast(float, app_state['max_hz'])
    max_resolution = cast(int, app_state['max_resolution'])
    cache_path = _get_rrd_cache_path(episode_id, max_hz, max_resolution)

    if cache_path.exists():
        logging.debug(f'Serving cached RRD for episode {episode_id} from {cache_path}')
        return FileResponse(cache_path, media_type='application/octet-stream', filename=f'episode_{episode_id}.rrd')

    def _stream_and_cache():
        # The final cache path denotes a complete file.
        fd, name = tempfile.mkstemp(dir=cache_path.parent, prefix=f'{cache_path.name}.', suffix='.partial')
        partial = Path(name)
        published = False
        try:
            with os.fdopen(fd, 'wb') as cache_file:
                for chunk in stream_episode_rrd(ds, episode_id, max_hz=max_hz, max_resolution=max_resolution):
                    cache_file.write(chunk)
                    yield chunk
            os.replace(partial, cache_path)
            published = True
        finally:
            if not published:
                partial.unlink(missing_ok=True)

    return StreamingResponse(
        _stream_and_cache(),
        media_type='application/octet-stream',
        headers={'Content-Disposition': f'attachment; filename=episode_{episode_id}.rrd'},
    )


@cfn.config()
def default_table() -> TableConfig:
    return {
        '__index__': ColumnConfig(label='#', format='%d'),
        '__duration__': ColumnConfig(label='Duration', format='%.2f sec'),
        'task': ColumnConfig(label='Task', filter=True),
    }


def _as_ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """The host parsed as an IP literal, or None when it is a name. A name is never resolved here."""
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _local_ip_addresses(version: int | None = None) -> list[str]:
    """Every IP address on a local interface, of `version` when one is named.

    An IPv6 link-local address is left out: it carries a zone no URL a browser is given can name.
    """
    addresses: list[str] = []
    for interface in psutil.net_if_addrs().values():
        for addr in interface:
            if addr.family not in (socket.AF_INET, socket.AF_INET6):
                continue
            ip = _as_ip(addr.address.split('%')[0])
            if ip is None or (version is not None and ip.version != version):
                continue
            if ip.version == 6 and ip.is_link_local:
                continue
            if str(ip) not in addresses:
                addresses.append(str(ip))
    return addresses


def _served_addresses(host: str) -> list[str]:
    """The addresses a server bound to `host` answers on.

    A concrete host answers on itself, a name on itself. A wildcard answers on the local addresses
    of the family it binds: `0.0.0.0` and an empty host listen on AF_INET alone, `::` takes IPv4 too.
    """
    if host == '':
        return _local_ip_addresses(version=4)
    ip = _as_ip(host)
    if ip is None:
        return [host]
    if not ip.is_unspecified:
        return [str(ip)]
    return _local_ip_addresses() if ip.version == 6 else _local_ip_addresses(version=4)


def _is_loopback(host: str) -> bool:
    ip = _as_ip(host)
    return ip.is_loopback if ip is not None else host == 'localhost'


def _access_url(scheme: str, host: str, port: int) -> str:
    """The URL a browser reaches `host` on."""
    ip = _as_ip(host)
    literal = f'[{host}]' if ip is not None and ip.version == 6 else host
    return f'{scheme}://{literal}:{port}'


@dataclass(frozen=True)
class _SelfSignedCert:
    keyfile: Path
    certfile: Path


def _generate_self_signed_cert(hosts: list[str]) -> _SelfSignedCert:
    """A self-signed certificate naming every host the server answers on.

    The subject is a fixed name: a client matches `subjectAltName` (RFC 6125), and X.509 caps the
    subject CN at 64 bytes, which a longer host name exceeds. An IP entry drops its zone, which
    OpenSSL refuses as a bad address.
    """
    entries = [f'IP:{host.split("%")[0]}' if _as_ip(host) is not None else f'DNS:{host}' for host in hosts]

    ssl_dir = Path(tempfile.mkdtemp(prefix='positronic-ssl-'))
    keyfile, certfile = ssl_dir / 'key.pem', ssl_dir / 'cert.pem'
    subprocess.run(
        [
            'openssl',
            'req',
            '-x509',
            '-newkey',
            'rsa:2048',
            '-keyout',
            keyfile,
            '-out',
            certfile,
            '-days',
            '365',
            '-nodes',
            '-subj',
            '/CN=positronic-server',
            '-addext',
            f'subjectAltName={",".join([*entries, "DNS:localhost"])}',
        ],
        check=True,
        capture_output=True,
    )
    atexit.register(shutil.rmtree, ssl_dir, True)
    return _SelfSignedCert(keyfile, certfile)


def configure_tables(
    *,
    root: str,
    cache_dir: Path,
    ep_table_cfg: TableConfig | None,
    group_tables: dict[str, GroupTableConfig] | None,
    home_page: str | None,
    max_resolution: int,
    max_hz: float,
) -> None:
    """Set what the tables show and how a recording is built; see `main` for what each value does."""
    app_state['root'] = root
    app_state['cache_dir'] = cache_dir
    app_state['episode_table_cfg'] = ep_table_cfg or {}
    app_state['group_tables_cfg'] = group_tables or {}
    app_state['max_resolution'] = max_resolution
    app_state['max_hz'] = max_hz
    app_state['home_page'] = home_page


@cfn.config(
    dataset=positronic.cfg.ds.local_all,
    ep_table_cfg=default_table,
    max_resolution=DEFAULT_MAX_RESOLUTION,
    group_tables=None,
)
def main(
    dataset: Dataset,
    ep_table_cfg: TableConfig | None,
    max_resolution: int,
    max_hz: float = DEFAULT_MAX_HZ,
    cache_dir: str = '~/.cache/positronic/server/',
    host: str = '0.0.0.0',
    port: int = 8400,
    debug: bool = False,
    https: bool = True,
    reset_cache: bool = False,
    group_tables: dict[str, GroupTableConfig] | None = None,
    home_page: str | None = None,
    base_href: str = '/',
    title: str = '',
    show_paths: bool = True,
    static_export: bool = False,
):
    """Visualize a Dataset with Rerun.

    Episode viewer URL params:
        /episode/<id>?t=<seconds>      — open paused at seconds from episode start
        /episode/<id>?ts_ns=<nanos>    — open paused at absolute nanosecond timestamp

    Args:
        dataset: Dataset to visualize
        max_resolution: Long side an episode RRD's videos are re-encoded down to
        max_hz: Rate an episode RRD's numeric signals are thinned to; 0 keeps every sample
        cache_dir: Directory to cache generated RRD files
        host: Server host
        port: Server port
        debug: Enable debug logging
        reset_cache: If True, clear cache_dir at startup
        ep_table_cfg: Mapping of episode static data keys to display in episode list,
            where the value is either:
            - A string label to display as the column header
            - A dict with 'label' and optional 'format' and 'renderer' keys
                - 'label': Column header label
                - 'format': (optional) Format string for displaying the value
                - 'default': (optional) Default value to use if the actual value is missing
                - 'renderer': (optional) Renderer configuration for custom display
                - 'filter': (optional) Boolean indicating if the column is filterable

            There are special keys:
            - '__index__': Episode index
            - '__duration__': Episode duration in seconds

            Example:
            {
                '__duration__': {'label': 'Duration', 'format': '%.2f sec'},
                'task': 'Task',
                'status': {
                    'label': 'Status',
                    'renderer': {
                        'type': 'badge',
                        'options': {
                            'degraded': {'label': 'Degraded', 'variant': 'danger'},
                            'assist': {'label': 'Assist', 'variant': 'warning'},
                            'pass': {'label': 'Pass', 'variant': 'success'},
                        },
                    },
                },
            }
        group_tables: Mapping of group name to GroupTableConfig
        base_href: Path at the server root that every page link and API call resolves against
        title: Header text; the dataset root when empty
        show_paths: Whether the pages report where the dataset lives
        static_export: Whether the pages ask for the file names a static export writes
    """
    root = get_dataset_root(dataset) or 'unknown_dataset'
    deb_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=deb_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # configuronic passes this CLI value through uncoerced.
    cache_root = Path(cache_dir).expanduser()
    configure_tables(
        root=root,
        cache_dir=cache_root,
        ep_table_cfg=ep_table_cfg,
        group_tables=group_tables,
        home_page=home_page,
        max_resolution=max_resolution,
        max_hz=max_hz,
    )
    configure_pages(base_href=base_href, title=title, show_paths=show_paths, static_export=static_export)
    app_state['loading_state'] = True

    if reset_cache and cache_root.exists():
        logging.info(f'Clearing RRD cache directory: {cache_root.resolve()}')
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    logging.info(f'Loading dataset from: {root}')
    logging.info(f'RRD cache directory: {cache_root.resolve()}')

    def load_dataset():
        try:
            ds = CachedDataset(dataset)
            logging.info(f'Dataset loaded. Episodes: {len(ds)}')
            install_dataset(ds)
        except Exception as e:
            logging.error(f'Failed to load dataset: {e}', exc_info=True)
            app_state['loading_state'] = False

    # Load dataset in background
    t = threading.Thread(target=load_dataset, daemon=True)
    t.start()

    served = _served_addresses(host)
    cert = _generate_self_signed_cert(served) if https else None
    ssl_kwargs = {'ssl_keyfile': str(cert.keyfile), 'ssl_certfile': str(cert.certfile)} if cert else {}
    scheme = 'https' if https else 'http'

    exposed = [] if https else [address for address in served if not _is_loopback(address)]
    if exposed:
        logging.warning(
            f'Serving plain HTTP on {", ".join(exposed)}. A browser exposes WebCodecs only to a secure context, '
            f'so video panels fail to decode there with "VideoDecoder is not defined". Serve over HTTPS, or '
            f'reach the server on a loopback address.'
        )
    urls = [_access_url(scheme, address, port) for address in served] or [_access_url(scheme, host, port)]
    logging.info(f'Starting server on {", ".join(urls)}')

    uvicorn.run(app, host=host, port=port, log_level='debug' if debug else 'info', **ssl_kwargs)


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
