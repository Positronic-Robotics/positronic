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
from collections.abc import Callable, Iterable, Iterator, Sequence
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


@dataclass(frozen=True)
class PageConfig:
    """The settings every page is rendered with."""

    base_href: str = '/'
    title: str = ''  # empty = the dataset root
    show_paths: bool = True
    static_export: bool = False


# The app state's key for the `PageConfig` every page reads.
_PAGE_CONFIG_KEY = 'pages'

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
    _PAGE_CONFIG_KEY: PageConfig(),
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


MAX_COMPONENT_BYTES = 200


def _path_component(value: str) -> str:
    """``value`` as one filename component, injectively and within any filesystem's name limit."""
    encoded = quote(value, safe='')
    if len(encoded.encode()) <= MAX_COMPONENT_BYTES:
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
# holding its values.
GROUP_FILTERS = 'group_filters'
FILTER_VALUES = 'values'


def _page_config() -> PageConfig:
    return cast(PageConfig, app_state[_PAGE_CONFIG_KEY])


def _shown_root() -> str:
    """The dataset root the pages report; empty when the viewer hides paths."""
    return str(app_state['root']) if _page_config().show_paths else ''


def _route(endpoint: Callable, **params: str) -> str:
    """The path `endpoint` answers at, relative to the base href."""
    return app.url_path_for(endpoint.__name__, **params).removeprefix('/')


def episodes_link() -> str:
    return _route(episodes_view)


def episode_link(episode_id: int) -> str:
    return _route(episode_viewer, episode_id=str(episode_id))


def episode_rrd_link(episode_id: int) -> str:
    return _route(api_episode_rrd, episode_id=str(episode_id))


def _download_segment(key: str) -> str:
    """`key` as one segment of a download link: percent-encoded, and within any filesystem's name limit.

    A browser rewrites a segment that is empty or all dots, and a filesystem refuses a name past its
    limit, so such a key has no link.
    """
    if key in ('', '.', '..'):
        raise ValueError(f'a static value with a key {key!r} has no link a browser keeps')
    encoded = quote(key, safe='')
    if len(encoded.encode()) > MAX_COMPONENT_BYTES:
        raise ValueError(
            f'a static value with a key of {len(encoded.encode())} encoded bytes has no link a file is named by'
        )
    return encoded


def _download_path(field_path: tuple[str, ...]) -> str:
    """`field_path` as the segments of a download link, one percent-encoded segment per key."""
    return '/'.join(_download_segment(key) for key in field_path)


def download_link(episode_id: int, field_path: tuple[str, ...]) -> str:
    """The download's path: one percent-encoded segment per key, so a `.`, `/`, `?` or space inside a key
    stays in its segment and two static values never share a link."""
    return _route(api_episode_static_field, episode_id=str(episode_id), field_path=_download_path(field_path))


def group_link(name: str) -> str:
    return _route(grouped_view, suffix=name)


def group_api_link(name: str) -> str:
    return _route(api_groups, suffix=name)


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
        url = group_link(group_name)
        label = group_name.replace('_', ' ').title()
        nav_items.append({'name': group_name, 'url': url, 'label': label})

    # Episodes link last (unless it's the home page)
    if home_page:
        nav_items.append({'name': 'episodes', 'url': episodes_url, 'label': 'Episodes'})
    else:
        # Episodes is home, insert at beginning
        nav_items.insert(0, {'name': 'episodes', 'url': '.', 'label': 'Episodes'})

    pages = _page_config()
    return {
        'nav_items': nav_items,
        'home_page': home_page,
        'episodes_url': episodes_url,
        'base_href': pages.base_href,
        'title': pages.title or _shown_root(),
        'show_paths': pages.show_paths,
        'static_export': pages.static_export,
    }


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    home_page = app_state.get('home_page')
    if home_page:
        # Render group view as home page
        return templates.TemplateResponse(
            request,
            'grouped.html',
            {'api_endpoint': group_api_link(str(home_page)), **_page_context(), 'current_page': home_page},
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
    relative link somewhere other than the prefix, and so does a path opening with two slashes, which
    a browser reads as a host. A browser resolves a dot segment and a backslash before it reads the
    `<base>`, so a path holding one names a different prefix than it shows.
    """
    parts = urlsplit(value)
    rooted = parts.path.startswith('/') and not parts.path.startswith('//')
    if parts.scheme or parts.netloc or parts.query or parts.fragment or not rooted:
        raise ValueError(f'base_href must be a path at the server root and nothing else, got {value!r}')
    if '\\' in parts.path or any(unquote(segment) in ('.', '..') for segment in parts.path.split('/')):
        raise ValueError(f'base_href must hold no dot segment and no backslash, got {value!r}')
    return parts.path if parts.path.endswith('/') else parts.path + '/'


def configure_pages(
    *, base_href: str = '/', title: str = '', show_paths: bool = True, static_export: bool = False
) -> None:
    """Set where the pages are served and what they show.

    `base_href` is the path at the server root every page link and API call resolves against. `title` is
    the header text, the dataset root when empty. `show_paths` says whether a page reports where the
    dataset lives. `static_export` makes the pages read the files a static export writes.
    """
    app_state[_PAGE_CONFIG_KEY] = PageConfig(
        base_href=normalized_base_href(base_href), title=title, show_paths=show_paths, static_export=static_export
    )


def install_dataset(dataset: Dataset) -> None:
    """Serve `dataset` from now on."""
    app_state['dataset'] = dataset
    _api_cache.clear()
    app_state['loading_state'] = False


_APP_STATE_HELD = threading.Lock()


@contextmanager
def app_state_restored() -> Iterator[None]:
    """Hold the app's state for the block and give it back on exit; a second holder waits."""
    with _APP_STATE_HELD:
        saved = dict(app_state)
        try:
            yield
        finally:
            app_state.clear()
            app_state.update(saved)
            _api_cache.clear()


# The field of a static value's sidebar entry that holds its download link.
DOWNLOAD_LINK = '__download__'


def is_download(value: object) -> bool:
    """Whether a static value reaches a page as a download link rather than inline."""
    return isinstance(value, bytes) or (isinstance(value, str) and len(value) > 1024)


def _downloads(value: object, prefix: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], object]]:
    if is_download(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _downloads(item, prefix + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _downloads(item, prefix + (str(index),))


def download_paths(static: dict) -> Iterator[tuple[str, ...]]:
    """The key path of every static value an episode page links as a download; a list item by its index.

    Each key is one URL segment, so distinct paths never share a link. A key that makes no segment —
    empty, `.`, `..`, or past a file name's limit once encoded — is refused.
    """
    for key_path, _ in _downloads(static, ()):
        _download_path(key_path)
        yield key_path


def download_at(static: dict, key_path: tuple[str, ...]) -> bytes | str | None:
    """The value a page links as a download at `key_path` into `static`, a list item by its index; None where
    `static` holds no download there."""
    value: object = static
    for key in key_path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isdigit() and str(int(key)) == key and int(key) < len(value):
            value = value[int(key)]
        else:
            return None
    return cast(bytes | str, value) if is_download(value) else None


@dataclass(frozen=True)
class DownloadMetadata:
    """What a page says about a download beside its link: `bytes` or `text`, and its size."""

    type: str
    size: int


def download_metadata(value: bytes | str) -> DownloadMetadata:
    return DownloadMetadata('bytes' if isinstance(value, bytes) else 'text', len(value))


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

    links = {path: download_link(episode_id, path) for path in download_paths(episode.static)}

    def _make_serializable(obj, key_path=()):
        if is_download(obj):
            link = links[key_path]
            return {DOWNLOAD_LINK: link, **asdict(download_metadata(obj))}
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: _make_serializable(v, key_path + (k,)) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v, key_path + (str(i),)) for i, v in enumerate(obj)]
        return obj

    return templates.TemplateResponse(
        request,
        'episode.html',
        {
            'episode_id': episode_id,
            'num_episodes': len(ds),
            'viewer_path': f'/static/rerun/{rr.__version__}/index.html',
            'task': episode.static.get(keys.TASK, None),
            'rrd_path': episode_rrd_link(episode_id),
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


def filter_spelling(value: object) -> str | None:
    """The string a page offers a static value as and sends back in a query; None for an absent value."""
    return None if value is None else str(value)


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
        return all(filter_spelling(ep.static.get(k)) == v for k, v in filters.items())

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


def _grouped(
    ds: Dataset, group_keys: tuple[str, ...], filter_keys: Iterable[str], active_filters: dict[str, str]
) -> tuple[dict[str, set[str]], dict[tuple[Any, ...], list[Episode]]]:
    """The values each filter offers, and the episodes that match `active_filters`, by group id."""
    offered: dict[str, set[str]] = {key: set() for key in filter_keys}
    groups: dict[tuple[Any, ...], list[Episode]] = defaultdict(list)
    for episode in ds:
        # Every value is offered, whichever filters are active
        for filter_key in offered:
            spelling = filter_spelling(episode.static.get(filter_key))
            if spelling is not None:
                offered[filter_key].add(spelling)
        if all(filter_spelling(episode.static.get(key)) == value for key, value in active_filters.items()):
            groups[_group_id(episode, group_keys)].append(episode)
    return offered, groups


@app.get('/api/groups/{suffix}')
@require_dataset
async def api_groups(request: Request, suffix: str):
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

    offered, groups = _grouped(cast(Dataset, ds), group_keys, cfg.group_filter_keys, active_filters)
    group_filters = {
        key: {'label': label or key, FILTER_VALUES: sorted(offered[key])}
        for key, label in cfg.group_filter_keys.items()
    }

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
        request, 'grouped.html', {'api_endpoint': group_api_link(suffix), **_page_context(), 'current_page': suffix}
    )


@app.get('/api/dataset_status')
async def api_dataset_status():
    return {
        'loading': app_state['loading_state'],
        'loaded': app_state.get('dataset', None) is not None,
        'repo_id': _shown_root(),
    }


def _static_at(static: dict, key_path: Sequence[str]) -> object:
    """The value each key of `key_path` names in turn into `static`, or an HTTP 404.

    Every key is one path segment, so a key holding a dot (`link0.stl`) is one step; a list is walked by
    an index.
    """
    value: object = static
    for key in key_path:
        if isinstance(value, list):
            if not key.isdigit() or int(key) >= len(value):
                raise HTTPException(status_code=404, detail=f'Field not found: {"/".join(key_path)}')
            value = value[int(key)]
        elif isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise HTTPException(status_code=404, detail=f'Field not found: {"/".join(key_path)}')
    return value


def _content_disposition(disposition: str, filename: str) -> str:
    """The `Content-Disposition` value naming a download `filename`; a name outside the plain ASCII characters,
    a `/` included, is percent-encoded."""
    encoded = quote(filename, safe='')
    if encoded == filename:
        return f'{disposition}; filename="{filename}"'
    return f"{disposition}; filename*=utf-8''{encoded}"


@app.get('/api/episode/{episode_id}/static/{field_path:path}')
@require_dataset
async def api_episode_static_field(episode_id: int, field_path: str, request: Request):
    ds = app_state.get('dataset')
    try:
        episode = ds[episode_id]
    except IndexError as e:
        raise HTTPException(status_code=404, detail='Episode not found') from e

    # The raw path keeps `%2F` apart from a `/` that separates two keys, which the decoded path loses.
    prefix = '/' + _route(api_episode_static_field, episode_id=str(episode_id), field_path='')
    tail = request.scope['raw_path'].decode('ascii').split('?', 1)[0].removeprefix(prefix)
    key_path = [unquote(segment) for segment in tail.split('/')]
    value = _static_at(episode.static, key_path)
    filename = key_path[-1]
    if isinstance(value, bytes):
        return Response(
            content=value,
            media_type='application/octet-stream',
            headers={'Content-Disposition': _content_disposition('attachment', filename)},
        )
    if isinstance(value, str):
        return Response(
            content=value.encode(),
            media_type='text/plain; charset=utf-8',
            headers={'Content-Disposition': _content_disposition('inline', f'{filename}.txt')},
        )
    raise HTTPException(status_code=400, detail=f'Field {"/".join(key_path)} is not downloadable')


def _recording_cache_path(episode_id: int) -> Path:
    return _get_rrd_cache_path(episode_id, cast(float, app_state['max_hz']), cast(int, app_state['max_resolution']))


def _recording_chunks_cached(episode_id: int, cache_path: Path) -> Iterator[bytes]:
    """The recording's chunks as they are built, written to `cache_path`; the path names a complete file only."""
    ds = cast(Dataset, app_state['dataset'])
    max_hz = cast(float, app_state['max_hz'])
    max_resolution = cast(int, app_state['max_resolution'])
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


def episode_rrd_path(episode_id: int) -> Path:
    """The complete cached recording of `episode_id`, built when the cache holds none."""
    cache_path = _recording_cache_path(episode_id)
    if not cache_path.exists():
        for _ in _recording_chunks_cached(episode_id, cache_path):
            pass
    return cache_path


@app.get('/api/episode_rrd/{episode_id}')
@require_dataset
async def api_episode_rrd(episode_id: int):
    cache_path = _recording_cache_path(episode_id)
    if cache_path.exists():
        logging.debug(f'Serving cached RRD for episode {episode_id} from {cache_path}')
        return FileResponse(cache_path, media_type='application/octet-stream', filename=f'episode_{episode_id}.rrd')
    return StreamingResponse(
        _recording_chunks_cached(episode_id, cache_path),
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
    """Set what the tables show and how a recording is built.

    `ep_table_cfg` maps an episode's static keys to the columns of the episode table. `group_tables` holds
    each grouped table by name, and `home_page` names the one served at the root, or None for the episodes.
    A recording's videos are re-encoded down to `max_resolution` on the long side, and its numeric signals
    are thinned to `max_hz`; 0 keeps every sample. `root` is the dataset path the pages report, and the
    recordings are cached under `cache_dir`. A table response cached under the previous settings is dropped.
    """
    episode_columns = set(ep_table_cfg or {})
    for name, cfg in (group_tables or {}).items():
        if not name or name in ('.', '..') or quote(name, safe='') != name or len(name) > MAX_COMPONENT_BYTES:
            raise ValueError(
                f'a group name is one URL path segment of at most {MAX_COMPONENT_BYTES} letters, digits, "_", "-" '
                f'and ".", got {name!r}'
            )
        group_keys = (cfg.group_keys,) if isinstance(cfg.group_keys, str) else cfg.group_keys
        for key in (*group_keys, *cfg.group_filter_keys):
            if key not in episode_columns:
                raise ValueError(
                    f'group table {name!r} sends {key!r} to a View link, and it is no column of the episode '
                    f'table, so the flat table cannot filter on it; the episode columns are {sorted(episode_columns)}'
                )
    if home_page and home_page not in (group_tables or {}):
        raise ValueError(f'home_page {home_page!r} names no group table; the tables are {sorted(group_tables or {})}')
    app_state['root'] = root
    app_state['cache_dir'] = cache_dir
    app_state['episode_table_cfg'] = ep_table_cfg or {}
    app_state['group_tables_cfg'] = group_tables or {}
    app_state['max_resolution'] = max_resolution
    app_state['max_hz'] = max_hz
    app_state['home_page'] = home_page
    _api_cache.clear()


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
    configure_pages(base_href=base_href, title=title, show_paths=show_paths)
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
