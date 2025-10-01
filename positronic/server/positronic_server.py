"""A FastAPI web server for visualizing Positronic LocalDatasets using Rerun."""

import logging
import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import configuronic as cfn
import rerun as rr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from positronic import utils
from positronic.dataset.local_dataset import LocalDataset
from positronic.server.dataset_utils import get_dataset_info, get_episodes_list, stream_episode_rrd

# Global app state
app_state: dict[str, object] = {
    'dataset': None,
    'loading_state': True,
    'root': '',
    'cache_dir': '',
}


def _pkg_path(*parts: str) -> str:
    return str(Path(__file__).resolve().parent.joinpath(*parts))


def _get_rrd_cache_path(episode_id: int) -> str:
    ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
    if ds is None:
        raise RuntimeError('Dataset not loaded')
    cache_root = str(app_state['cache_dir'])
    ds_id = str(Path(ds.root).resolve()).replace(os.sep, '_').replace(':', '')
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

# CORS for convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

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
async def episode_viewer(request: Request, episode_id: int):
    if app_state['loading_state']:
        raise HTTPException(status_code=202, detail='Dataset is still loading. Please wait...')
    ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
    if ds is None:
        raise HTTPException(status_code=500, detail='Dataset failed to load')

    try:
        episode = ds[episode_id]
    except IndexError as e:
        raise HTTPException(status_code=404, detail='Episode not found') from e

    return templates.TemplateResponse(
        'episode.html',
        {
            'request': request,
            'episode_id': episode_id,
            'num_episodes': len(ds),
            'rerun_version': rr.__version__,
            'task': episode.static.get('task', None),
            'repo_id': app_state['root'],
        },
    )


@app.get('/api/dataset_info')
async def api_dataset_info():
    if app_state['loading_state']:
        raise HTTPException(status_code=202, detail='Dataset is loading...')
    ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
    if ds is None:
        raise HTTPException(status_code=500, detail='Dataset failed to load')
    return get_dataset_info(ds)


@app.get('/api/episodes')
async def api_episodes():
    if app_state['loading_state']:
        raise HTTPException(status_code=202, detail='Dataset is loading...')
    ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
    if ds is None:
        raise HTTPException(status_code=500, detail='Dataset failed to load')
    return get_episodes_list(ds)


@app.get('/api/dataset_status')
async def api_dataset_status():
    return {
        'loading': app_state['loading_state'],
        'loaded': app_state.get('dataset', None) is not None,
        'repo_id': app_state['root'],
    }


@app.get('/api/episode_rrd/{episode_id}')
async def api_episode_rrd(episode_id: int):
    if app_state['loading_state']:
        raise HTTPException(status_code=202, detail='Dataset is still loading')
    ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
    if ds is None:
        raise HTTPException(status_code=500, detail='Dataset failed to load')

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
def main(
    root: str,
    cache_dir: str = os.path.expanduser('~/.cache/positronic/server/'),
    host: str = '0.0.0.0',
    port: int = 5000,
    debug: bool = False,
    reset_cache: bool = False,
):
    """Visualize a LocalDataset with Rerun.

    Args:
        root: Path to dataset root directory
        cache_dir: Directory to cache generated RRD files
        host: Server host
        port: Server port
        debug: Enable debug logging
        reset_cache: If True, clear cache_dir at startup
    """
    deb_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=deb_level, format='%(asctime)s - %(levelname)s - %(message)s')

    app_state['root'] = str(root)
    app_state['cache_dir'] = cache_dir
    app_state['loading_state'] = True

    if reset_cache and os.path.exists(cache_dir):
        logging.info(f'Clearing RRD cache directory: {os.path.abspath(cache_dir)}')
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    logging.info(f'Loading dataset from: {root}')
    logging.info(f'RRD cache directory: {os.path.abspath(cache_dir)}')

    def load_dataset():
        try:
            ds = LocalDataset(Path(root))
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


if __name__ == '__main__':
    cfn.cli(main)
