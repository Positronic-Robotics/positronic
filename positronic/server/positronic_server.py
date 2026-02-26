"""A FastAPI web server for visualizing Positronic LocalDatasets using Rerun."""

import logging
import os
import shutil
import subprocess
import tempfile
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import configuronic as cfn
import pos3
import rerun as rr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import positronic.cfg.ds
from positronic import utils
from positronic.dataset import CachedDataset, Dataset, Episode
from positronic.dataset.local_dataset import LocalDataset
from positronic.server.dataset_utils import get_dataset_root, get_episodes_list, stream_episode_rrd
from positronic.utils.logging import init_logging

# Global app state
app_state: dict[str, object] = {
    'dataset': None,
    'loading_state': True,
    'root': '',
    'cache_dir': '',
    'episode_keys': {},
    'max_resolution': 640,
    'group_tables_cfg': {},
    'home_page': None,  # None = episodes, or group name like 'tasks'
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


def _get_rrd_cache_path(episode_id: int) -> str:
    ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
    if ds is None:
        raise RuntimeError('Dataset not loaded')
    cache_root = str(app_state['cache_dir'])
    ds_id = str(Path(str(app_state['root'])).resolve()).replace(os.sep, '_').replace(':', '')
    episode_cache_dir = os.path.join(cache_root, ds_id)
    os.makedirs(episode_cache_dir, exist_ok=True)
    return os.path.join(episode_cache_dir, f'episode_{episode_id}.rrd')


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


def _iter_file_chunks(path: str, *, chunk_size: int = 128 * 1024):
    with open(path, 'rb') as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            yield chunk


def _get_nav_context() -> dict[str, Any]:
    """Build navigation context for templates."""
    home_page = app_state.get('home_page')
    group_tables = app_state.get('group_tables_cfg', {})

    nav_items = []
    episodes_url = '/episodes' if home_page else '/'

    # If home_page is set, it goes first
    if home_page and home_page in group_tables:
        label = home_page.replace('_', ' ').title()
        nav_items.append({'name': home_page, 'url': '/', 'label': label})

    # Other group links
    for group_name in group_tables.keys():
        if group_name == home_page:
            continue  # Already added as home
        url = f'/groups/{group_name}'
        label = group_name.replace('_', ' ').title()
        nav_items.append({'name': group_name, 'url': url, 'label': label})

    # Episodes link last (unless it's the home page)
    if home_page:
        nav_items.append({'name': 'episodes', 'url': episodes_url, 'label': 'Episodes'})
    else:
        # Episodes is home, insert at beginning
        nav_items.insert(0, {'name': 'episodes', 'url': '/', 'label': 'Episodes'})

    return {'nav_items': nav_items, 'home_page': home_page, 'episodes_url': episodes_url}


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    home_page = app_state.get('home_page')
    if home_page:
        # Render group view as home page
        return templates.TemplateResponse(
            'grouped.html',
            {
                'request': request,
                'repo_id': app_state['root'],
                'api_endpoint': f'/api/groups/{home_page}',
                **_get_nav_context(),
                'current_page': home_page,
            },
        )
    # Default: render episodes
    return templates.TemplateResponse(
        'index.html',
        {'request': request, 'repo_id': app_state['root'], **_get_nav_context(), 'current_page': 'episodes'},
    )


@app.get('/episodes', response_class=HTMLResponse)
async def episodes_view(request: Request):
    """Episodes list view (used when home_page is set to a group)."""
    return templates.TemplateResponse(
        'index.html',
        {'request': request, 'repo_id': app_state['root'], **_get_nav_context(), 'current_page': 'episodes'},
    )


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

    # Ensure static_data is JSON serializable (e.g. handle datetime)
    def _make_serializable(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        return obj

    return templates.TemplateResponse(
        'episode.html',
        {
            'request': request,
            'episode_id': episode_id,
            'num_episodes': len(ds),
            'rerun_version': rr.__version__,
            'task': episode.static.get('task', None),
            'repo_id': app_state['root'],
            'episode_path': meta.get('path'),
            'episode_size_mb': size_mb_display,
            'static_data': _make_serializable(episode.static),
        },
    )


@app.get('/api/dataset_info')
@require_dataset
async def api_dataset_info():
    ds = app_state.get('dataset')
    return {'root': app_state['root'], 'num_episodes': len(ds)}


@dataclass
class RendererConfig:
    """Custom cell renderer sent to the frontend.

    Attributes:
        type: Renderer type dispatched in app.js — ``'badge'`` or ``'icon'``.
        options: Value-keyed mapping. For badges: ``{value: {'label', 'variant'}}``.
            For icons: ``{value: {'src', 'label?'}}``.
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
    format: str | None = None
    default: Any = None
    renderer: RendererConfig | None = None
    filter: bool = False


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

        if cfg.renderer:
            column['renderer'] = asdict(cfg.renderer)

        if cfg.filter:
            column['filter'] = cfg.filter

        columns.append(column)
    return columns, formatters, defaults


@app.get('/api/episodes')
@require_dataset
async def api_episodes(request: Request):
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
    return {'columns': columns, 'episodes': episodes}


def _group_id(episode: Episode, group_keys: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(episode.static.get(k) for k in group_keys)


@app.get('/api/groups/{suffix}')
@require_dataset
async def api_groups(request: Request, suffix: str):
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
    group_filters = {key: {'label': label or key, 'values': set()} for key, label in cfg.group_filter_keys.items()}
    for episode in ds:
        # Apply filters
        match = all(episode.static[key] == value for key, value in active_filters.items())
        if match:
            groups[_group_id(episode, group_keys)].append(episode)
            for filter_key in cfg.group_filter_keys:
                group_filters[filter_key]['values'].add(episode.static.get(filter_key))

    rows = []
    for group_id, episodes in groups.items():
        key_fields = {k: group_id[i] for i, k in enumerate(group_keys)}
        rows.append({**key_fields, '__meta__': {'group': key_fields}, **cfg.group_fn(episodes)})

    episodes = get_episodes_list(rows, cfg.format_table.keys(), formatters=formatters, defaults=defaults)
    result = {'columns': columns, 'episodes': episodes, 'group_filters': group_filters}
    if cfg.default_sort:
        result['default_sort'] = asdict(cfg.default_sort)
    return result


@app.get('/groups/{suffix}', response_class=HTMLResponse)
async def grouped_view(request: Request, suffix: str):
    return templates.TemplateResponse(
        'grouped.html',
        {
            'request': request,
            'repo_id': app_state['root'],
            'api_endpoint': f'/api/groups/{suffix}',
            **_get_nav_context(),
            'current_page': suffix,
        },
    )


@app.get('/api/dataset_status')
async def api_dataset_status():
    return {
        'loading': app_state['loading_state'],
        'loaded': app_state.get('dataset', None) is not None,
        'repo_id': app_state['root'],
    }


@app.get('/api/episode_rrd/{episode_id}')
@require_dataset
async def api_episode_rrd(episode_id: int):
    ds = app_state.get('dataset')
    cache_path = _get_rrd_cache_path(episode_id)

    if os.path.exists(cache_path):
        logging.debug(f'Serving cached RRD for episode {episode_id} from {cache_path}')
        return StreamingResponse(
            _iter_file_chunks(cache_path),
            media_type='application/octet-stream',
            headers={'Content-Disposition': f'attachment; filename=episode_{episode_id}.rrd'},
        )

    def _stream_and_cache():
        success = False
        try:
            with open(cache_path, 'wb') as cache_file:
                for chunk in stream_episode_rrd(ds, episode_id):
                    cache_file.write(chunk)
                    yield chunk
            success = True
        finally:
            if not success:
                shutil.rmtree(cache_path, ignore_errors=True)

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


@cfn.config(dataset=positronic.cfg.ds.local_all, ep_table_cfg=default_table, max_resolution=640, group_tables=None)
def main(
    dataset: Dataset,
    ep_table_cfg: TableConfig | None,
    max_resolution: int,
    cache_dir: str = os.path.expanduser('~/.cache/positronic/server/'),
    host: str = '0.0.0.0',
    port: int = 5000,
    debug: bool = False,
    https: bool = True,
    reset_cache: bool = False,
    group_tables: dict[str, GroupTableConfig] | None = None,
    home_page: str | None = None,
):
    """Visualize a Dataset with Rerun.

    Args:
        dataset: Dataset to visualize
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
    """
    root = get_dataset_root(dataset) or 'unknown_dataset'
    deb_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=deb_level, format='%(asctime)s - %(levelname)s - %(message)s')

    app_state['root'] = root
    app_state['cache_dir'] = cache_dir
    app_state['loading_state'] = True
    app_state['episode_table_cfg'] = ep_table_cfg or {}
    app_state['group_tables_cfg'] = group_tables or {}
    app_state['max_resolution'] = max_resolution
    app_state['home_page'] = home_page

    if reset_cache and os.path.exists(cache_dir):
        logging.info(f'Clearing RRD cache directory: {os.path.abspath(cache_dir)}')
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    logging.info(f'Loading dataset from: {root}')
    logging.info(f'RRD cache directory: {os.path.abspath(cache_dir)}')

    def load_dataset():
        try:
            ds = CachedDataset(dataset)
            logging.info(f'Dataset loaded. Episodes: {len(ds)}')
            app_state['dataset'] = ds
            app_state['loading_state'] = False
        except Exception as e:
            logging.error(f'Failed to load dataset: {e}', exc_info=True)
            app_state['loading_state'] = False

    # Load dataset in background
    t = threading.Thread(target=load_dataset, daemon=True)
    t.start()

    ssl_kwargs = {}
    if https:
        ssl_dir = tempfile.mkdtemp(prefix='positronic-ssl-')
        keyfile = os.path.join(ssl_dir, 'key.pem')
        certfile = os.path.join(ssl_dir, 'cert.pem')
        primary_host = utils.resolve_host_ip()
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
                f'/CN={primary_host}',
                '-addext',
                f'subjectAltName=DNS:localhost,IP:{primary_host}',
            ],
            check=True,
            capture_output=True,
        )
        ssl_kwargs = {'ssl_keyfile': keyfile, 'ssl_certfile': certfile}

    primary_host = utils.resolve_host_ip()
    scheme = 'https' if https else 'http'
    logging.info(f'Starting server on {scheme}://{primary_host}:{port}')
    uvicorn.run(app, host=host, port=port, log_level='debug' if debug else 'info', **ssl_kwargs)


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
