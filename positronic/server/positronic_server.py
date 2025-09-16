"""A FastAPI web server for visualizing Positronic LocalDatasets using Rerun.

Requires `positronic[server]` extras to be installed.
"""

import logging
import os
import shutil
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import rerun as rr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

import configuronic as cfn
from positronic.dataset.local_dataset import LocalDataset
from positronic.server.dataset_utils import generate_episode_rrd, get_dataset_info, get_episodes_list

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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates (packaged relative to this file)
_static_dir = _pkg_path('static')
_templates_dir = _pkg_path('templates')
app.mount("/static", StaticFiles(directory=_static_dir), name="static")
templates = Jinja2Templates(directory=_templates_dir)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "repo_id": app_state['root']})


@app.get("/episode/{episode_id}", response_class=HTMLResponse)
async def episode_viewer(request: Request, episode_id: int):
    if app_state['loading_state']:
        raise HTTPException(status_code=202, detail="Dataset is still loading. Please wait...")
    ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
    if ds is None:
        raise HTTPException(status_code=500, detail="Dataset failed to load")

    try:
        episode = ds[episode_id]
    except IndexError:
        raise HTTPException(status_code=404, detail="Episode not found")

    return templates.TemplateResponse(
        "episode.html",
        {
            "request": request,
            "episode_id": episode_id,
            "num_episodes": len(ds),
            "rerun_version": rr.__version__,
            "task": episode.static.get('task', None),
            "repo_id": app_state['root'],
        },
    )


@app.get("/api/dataset_info")
async def api_dataset_info():
    if app_state['loading_state']:
        raise HTTPException(status_code=202, detail="Dataset is loading...")
    ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
    if ds is None:
        raise HTTPException(status_code=500, detail="Dataset failed to load")
    return get_dataset_info(ds)


@app.get("/api/episodes")
async def api_episodes():
    if app_state['loading_state']:
        raise HTTPException(status_code=202, detail="Dataset is loading...")
    ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
    if ds is None:
        raise HTTPException(status_code=500, detail="Dataset failed to load")
    return get_episodes_list(ds)


@app.get("/api/dataset_status")
async def api_dataset_status():
    return {
        'loading': app_state['loading_state'],
        'loaded': app_state.get('dataset', None) is not None,
        'repo_id': app_state['root'],
    }


@app.get("/api/episode_rrd/{episode_id}")
async def api_episode_rrd(episode_id: int):
    if app_state['loading_state']:
        raise HTTPException(status_code=202, detail="Dataset is still loading")
    ds: LocalDataset | None = app_state.get('dataset')  # type: ignore[assignment]
    if ds is None:
        raise HTTPException(status_code=500, detail="Dataset failed to load")

    try:
        cache_path = _get_rrd_cache_path(episode_id)
        rrd_path = generate_episode_rrd(ds, episode_id, cache_path)
        if not os.path.exists(rrd_path):
            logging.error(f'RRD file not found at {rrd_path}')
            raise HTTPException(status_code=500, detail="RRD file generation failed")

        return FileResponse(path=rrd_path, media_type='application/octet-stream', filename=f'episode_{episode_id}.rrd')
    except Exception as e:
        logging.error(f'Error serving RRD file for episode {episode_id}: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@cfn.config()
def main(root: str,
         cache_dir: str = os.path.expanduser('~/.cache/positronic/server/'),
         host: str = '0.0.0.0',
         port: int = 5000,
         debug: bool = False,
         reset_cache: bool = False):
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

    logging.info(f'Starting server on {host}:{port}')
    uvicorn.run(app, host=host, port=port, log_level="debug" if debug else "info")


if __name__ == '__main__':
    cfn.cli(main)
