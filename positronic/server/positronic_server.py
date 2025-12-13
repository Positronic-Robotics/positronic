"""A FastAPI web server for visualizing Positronic LocalDatasets using Rerun."""

import logging
import os
import shutil
import threading
from collections import defaultdict
from collections.abc import Callable
from contextlib import asynccontextmanager
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

import positronic.cfg.dataset
from positronic import utils
from positronic.dataset import Dataset, Episode
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


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request, 'repo_id': app_state['root']})


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


def parse_table_cfg(table_cfg: dict[str, Any]) -> tuple:
    columns = []
    formatters = {}
    defaults = {}
    for key, value in table_cfg.items():
        column = {}
        if isinstance(value, dict):
            column['label'] = value.get('label', key)
            formatters[key] = value.get('format')
            defaults[key] = value.get('default')

            if 'renderer' in value:
                column['renderer'] = value['renderer']

            if 'filter' in value:
                column['filter'] = value['filter']
        else:
            column['label'] = value or key

        columns.append(column)
    return columns, formatters, defaults


@app.get('/api/episodes')
@require_dataset
async def api_episodes():
    ds = app_state.get('dataset')
    config = app_state['episode_table_cfg']
    columns, formatters, defaults = parse_table_cfg(config)
    ep_it = ({'__meta__': ep.meta, '__duration__': ep.duration_ns / 1e9, **ep.static} for ep in ds)
    episodes = get_episodes_list(ep_it, config.keys(), formatters=formatters, defaults=defaults)
    return {'columns': columns, 'episodes': episodes}


@app.get('/api/groups')
@require_dataset
async def api_groups():
    ds = app_state.get('dataset')
    group_key, group_fn, format_table = app_state.get('group_table_cfg')
    columns, formatters, defaults = parse_table_cfg(format_table)

    groups = defaultdict(list)
    for episode in ds:
        groups[episode.static[group_key]].append(episode)

    rows = [{group_key: key, '__meta__': {'group': key}, **group_fn(group)} for key, group in groups.items()]
    episodes = get_episodes_list(rows, format_table.keys(), formatters=formatters, defaults=defaults)
    return {'columns': columns, 'episodes': episodes}


@app.get('/grouped', response_class=HTMLResponse)
async def grouped_view(request: Request):
    return templates.TemplateResponse(
        'grouped.html', {'request': request, 'repo_id': app_state['root'], 'api_endpoint': '/api/groups'}
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
    max_resolution: int = app_state.get('max_resolution')  # type: ignore[assignment]

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
                for chunk in stream_episode_rrd(ds, episode_id, max_resolution):
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


TableConfig = dict[str, dict[str, Any]]


@cfn.config()
def default_table() -> TableConfig:
    return {
        '__index__': {'label': '#', 'format': '%d'},
        '__duration__': {'label': 'Duration', 'format': '%.2f sec'},
        'task': {'label': 'Task', 'filter': True},
    }


@cfn.config()
def eval_table() -> TableConfig:
    return {
        'task_code': {'label': 'Task', 'filter': True},
        'model': {'label': 'Model', 'filter': True},
        'units': {'label': 'Units'},
        'uph': {'label': 'UPH', 'format': '%.1f'},
        'success': {'label': 'Success', 'format': '%.1f%%'},
        'started': {'label': 'Started', 'format': '%Y-%m-%d %H:%M:%S'},
        'eval.outcome': {
            'label': 'Status',
            'filter': True,
            'renderer': {
                'type': 'badge',
                'options': {
                    # TODO: Currently the filter happens by original data, not the rendered value
                    'Success': {'label': 'Pass', 'variant': 'success'},
                    'Stalled': {'label': 'Fail', 'variant': 'warning'},
                    'Ran out of time': {'label': 'Fail', 'variant': 'warning'},
                    'System': {'label': 'Fail', 'variant': 'warning'},
                    'Safety': {'label': 'Safety violation', 'variant': 'danger'},
                },
            },
        },
        '__duration__': {'label': 'Duration', 'format': '%.1f sec'},
    }


@cfn.config()
def model_perf_table():
    group_key = 'model'

    def group_fn(episodes: list[Episode]) -> dict[str, Any]:
        duration, suc_items, total_items, assists = 0, 0, 0, 0
        for ep in episodes:
            duration += ep['eval.duration']
            suc_items += ep['eval.successful_items']
            total_items += ep['eval.total_items']
            assists += ep['eval.outcome'] == 'Success'

        return {
            'model': episodes[0]['model'],
            'UPH': suc_items / (duration / 3600),
            'Success': 100 * suc_items / total_items,
            'MTBF/A': (duration / assists) if assists > 0 else None,
        }

    format_table = {
        'model': {'label': 'Model'},
        'UPH': {'format': '%.1f'},
        'Success': {'format': '%.2f%%'},
        'MTBF/A': {'format': '%.1f sec', 'default': '-'},
    }

    return group_key, group_fn, format_table


@cfn.config(dataset=positronic.cfg.dataset.local_all, ep_table_cfg=default_table, group_table=model_perf_table)
def main(
    dataset: Dataset,
    cache_dir: str = os.path.expanduser('~/.cache/positronic/server/'),
    host: str = '0.0.0.0',
    port: int = 5000,
    debug: bool = False,
    reset_cache: bool = False,
    max_resolution: int = 640,
    ep_table_cfg: TableConfig | None = None,
    group_table: tuple[str, Callable[[list[Any]], dict[str, Any]], TableConfig] | None = None,
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
    """
    root = get_dataset_root(dataset) or 'unknown_dataset'
    deb_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=deb_level, format='%(asctime)s - %(levelname)s - %(message)s')

    app_state['root'] = root
    app_state['cache_dir'] = cache_dir
    app_state['loading_state'] = True
    app_state['episode_table_cfg'] = ep_table_cfg or {}
    app_state['group_table_cfg'] = group_table
    app_state['max_resolution'] = max_resolution

    if reset_cache and os.path.exists(cache_dir):
        logging.info(f'Clearing RRD cache directory: {os.path.abspath(cache_dir)}')
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    logging.info(f'Loading dataset from: {root}')
    logging.info(f'RRD cache directory: {os.path.abspath(cache_dir)}')

    def load_dataset():
        try:
            ds = dataset
            logging.info(f'Dataset loaded. Episodes: {len(ds)}')
            app_state['dataset'] = ds
            app_state['loading_state'] = False
        except Exception as e:
            logging.error(f'Failed to load dataset: {e}', exc_info=True)
            app_state['loading_state'] = False

    # Load dataset in background
    t = threading.Thread(target=load_dataset, daemon=True)
    t.start()

    primary_host = utils.resolve_host_ip()
    logging.info(f'Starting server on http://{primary_host}:{port}')
    uvicorn.run(app, host=host, port=port, log_level='debug' if debug else 'info')


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
