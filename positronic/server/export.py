"""Write the viewer for one dataset as static files.

The app is composed in this process and read with a test client, so the export holds what the
server answers. A page lands at `<route>/index.html`, an API response at `api/<route>.json`, a group
table as one file per filter set with `index.json` beside them, and a recording or a download under
`build/<build_id>/` when a build id is given.
"""

import itertools
import json
import logging
import mimetypes
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import configuronic as cfn
import pos3
from fastapi.testclient import TestClient

import positronic.cfg.ds
from pimm.logging import init_logging
from positronic.dataset import CachedDataset, Dataset, Episode
from positronic.server.dataset_utils import DEFAULT_MAX_HZ, DEFAULT_MAX_RESOLUTION, get_dataset_root
from positronic.server.positronic_server import (
    FILTER_VALUES,
    GROUP_FILTERS,
    GroupTableConfig,
    TableConfig,
    app,
    app_state_restored,
    configure_pages,
    configure_tables,
    default_table,
    download_paths,
    install_dataset,
)

logger = logging.getLogger(__name__)

# What a page, an API response and a group table's files are written as. A page is a directory
# with an index file, so a static host answers `episode/3` with it.
PAGE_FILE = 'index.html'
API_DIR = 'api'
BUILD_DIR = 'build'
GROUP_INDEX_FILE = 'index.json'
UNFILTERED_FILE = 'all.json'
# The app's own assets, at the server root, so every export a host serves shares one copy.
ASSET_DIR = 'static'

# The routes whose responses a page reads whole. `episodes` is filtered in the browser.
_WHOLE_API_ROUTES = ('dataset_info', 'dataset_status', 'episodes')


@dataclass(frozen=True)
class ExportedFile:
    """One file the export wrote, at `path` under the output directory."""

    path: PurePosixPath
    content_type: str
    size: int


@dataclass(frozen=True)
class GroupFile:
    """One file of a group table: the filter set it was read with, and its name beside the index."""

    params: dict[str, str]
    file: str


class _Output:
    def __init__(self, directory: Path):
        self._directory = directory
        self.files: list[ExportedFile] = []

    def write(self, path: str, body: bytes, content_type: str) -> ExportedFile:
        target = self._directory / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        written = ExportedFile(PurePosixPath(path), content_type, len(body))
        self.files.append(written)
        return written


def _fetch(client: TestClient, path: str, params: dict[str, str] | None = None) -> tuple[bytes, str]:
    response = client.get(path, params=params or {})
    response.raise_for_status()
    return response.content, response.headers.get('content-type', '')


def _filter_values(response: dict, key: str) -> list[str]:
    """The values a group's filter offers, as the page sends them.

    None is left out: the server compares a filter against an episode's own value, and no query
    matches a value that is absent.
    """
    values = response.get(GROUP_FILTERS, {}).get(key, {}).get(FILTER_VALUES, [])
    return sorted(str(value) for value in values if value is not None)


def filter_sets(response: dict) -> Iterator[dict[str, str]]:
    """Every filter set a group page can ask for, the empty one first.

    Each key offers its values and the choice of not filtering on it, so k keys of v values each
    give (v+1)^k sets.
    """
    keys = sorted(response.get(GROUP_FILTERS, {}))
    options = [[None, *_filter_values(response, key)] for key in keys]
    for chosen in itertools.product(*options):
        yield {key: value for key, value in zip(keys, chosen, strict=True) if value is not None}


def large_file_links_under(html: str, build_id: str) -> str:
    """`html` with every recording and download link moved under `build/<build_id>/`."""
    if not build_id:
        return html
    prefix = f'{BUILD_DIR}/{build_id}/'
    return html.replace('"api/episode_rrd/', f'"{prefix}api/episode_rrd/').replace(
        '"api/episode/', f'"{prefix}api/episode/'
    )


def _large_file_path(link: str, build_id: str) -> str:
    return f'{BUILD_DIR}/{build_id}/{link}' if build_id else link


def _write_pages(client: TestClient, out: _Output, group_names: list[str], count: int, build_id: str) -> list[str]:
    """Write every page, and give back the recording and download links the episode pages held."""
    body, content_type = _fetch(client, '/')
    out.write(PAGE_FILE, body, content_type)
    for route in ('episodes', *(f'groups/{name}' for name in group_names)):
        body, content_type = _fetch(client, f'/{route}')
        out.write(f'{route}/{PAGE_FILE}', body, content_type)
    links = []
    for index in range(count):
        body, content_type = _fetch(client, f'/episode/{index}')
        page = body.decode()
        moved = large_file_links_under(page, build_id)
        rrd_link = f'{API_DIR}/episode_rrd/{index}'
        if build_id and rrd_link not in page:
            raise RuntimeError(f'episode page {index} carries no recording link to move under the build')
        out.write(f'episode/{index}/{PAGE_FILE}', moved.encode(), content_type)
        links.append(rrd_link)
    return links


def _write_group(client: TestClient, out: _Output, name: str) -> None:
    route = f'{API_DIR}/groups/{name}'
    body, content_type = _fetch(client, f'/{route}')
    index = [GroupFile({}, UNFILTERED_FILE)]
    out.write(f'{route}/{UNFILTERED_FILE}', body, content_type)
    for number, params in enumerate((chosen for chosen in filter_sets(json.loads(body)) if chosen), start=1):
        filtered, filtered_type = _fetch(client, f'/{route}', params)
        entry = GroupFile(params, f'{number}.json')
        out.write(f'{route}/{entry.file}', filtered, filtered_type)
        index.append(entry)
    listing = json.dumps([{'params': entry.params, 'file': entry.file} for entry in index]).encode()
    out.write(f'{route}/{GROUP_INDEX_FILE}', listing, 'application/json')


# What the viewer ships and `mimetypes` does not answer for on every box.
_CONTENT_TYPE_BY_SUFFIX = {'.wasm': 'application/wasm', '.rrd': 'application/octet-stream'}


def asset_content_type(path: Path) -> str:
    """The `Content-Type` an asset file is served under."""
    if path.suffix in _CONTENT_TYPE_BY_SUFFIX:
        return _CONTENT_TYPE_BY_SUFFIX[path.suffix]
    return mimetypes.guess_type(path.name)[0] or 'application/octet-stream'


def _write_assets(out: _Output) -> None:
    static_dir = Path(__file__).resolve().parent / ASSET_DIR
    for path in sorted(p for p in static_dir.rglob('*') if p.is_file()):
        relative = path.relative_to(static_dir).as_posix()
        out.write(f'{ASSET_DIR}/{relative}', path.read_bytes(), asset_content_type(path))


def export_static(
    dataset: Dataset,
    out_dir: Path,
    *,
    ep_table_cfg: TableConfig | None = None,
    group_tables: dict[str, GroupTableConfig] | None = None,
    home_page: str | None = None,
    max_resolution: int = DEFAULT_MAX_RESOLUTION,
    max_hz: float = DEFAULT_MAX_HZ,
    base_href: str = '/',
    title: str = '',
    show_paths: bool = False,
    build_id: str = '',
    full_dataset: Dataset | None = None,
    assets: bool = True,
    cache_dir: Path | None = None,
) -> list[ExportedFile]:
    """Write the viewer for `dataset` under `out_dir` and give back every file written.

    The pages and the tables read `dataset`. The recordings and the downloads read `full_dataset`
    when given: the recording builder reads an episode's robot model out of its static values, so a
    caller that hides static values from the pages passes the unhidden dataset here. `assets` writes
    the app's own scripts, styles and viewer under `static/`, which every export at one host shares.
    """
    out = _Output(Path(out_dir))
    shown = CachedDataset(dataset)
    with app_state_restored(), tempfile.TemporaryDirectory() as scratch:
        configure_tables(
            root=get_dataset_root(dataset) or 'unknown_dataset',
            cache_dir=Path(cache_dir) if cache_dir else Path(scratch),
            ep_table_cfg=ep_table_cfg,
            group_tables=group_tables,
            home_page=home_page,
            max_resolution=max_resolution,
            max_hz=max_hz,
        )
        configure_pages(base_href=base_href, title=title, show_paths=show_paths, static_export=True)
        install_dataset(shown)
        client = TestClient(app)
        group_names = list(group_tables or {})

        links = _write_pages(client, out, group_names, len(shown), build_id)
        for route in _WHOLE_API_ROUTES:
            body, content_type = _fetch(client, f'/{API_DIR}/{route}')
            out.write(f'{API_DIR}/{route}.json', body, content_type)
        for name in group_names:
            _write_group(client, out, name)

        install_dataset(CachedDataset(full_dataset) if full_dataset is not None else shown)
        for index in range(len(shown)):
            static = cast(Episode, shown[index]).static
            links.extend(f'{API_DIR}/episode/{index}/static/{field}' for field in download_paths(static))
        for link in links:
            body, content_type = _fetch(client, f'/{link}')
            out.write(_large_file_path(link, build_id), body, content_type)
    if assets:
        _write_assets(out)
    logger.info('wrote %d files under %s', len(out.files), out_dir)
    return out.files


@cfn.config(dataset=positronic.cfg.ds.local_all, ep_table_cfg=default_table, group_tables=None)
def main(
    dataset: Dataset,
    out_dir: str,
    ep_table_cfg: TableConfig | None,
    max_resolution: int = DEFAULT_MAX_RESOLUTION,
    max_hz: float = DEFAULT_MAX_HZ,
    group_tables: dict[str, GroupTableConfig] | None = None,
    home_page: str | None = None,
    base_href: str = '/',
    title: str = '',
    show_paths: bool = False,
    build_id: str = '',
    assets: bool = True,
):
    """Write the viewer for a Dataset as static files, for any static host.

    Args:
        dataset: Dataset to export
        out_dir: Directory the files are written under; it is created
        ep_table_cfg, max_resolution, max_hz, group_tables, home_page: As `positronic-server` takes them
        base_href: Path at the host root the export is served under
        title: Header text; the dataset root when empty
        show_paths: Whether the pages report where the dataset lives
        build_id: Names one build; the recordings and the downloads are written under `build/<build_id>/`
        assets: Whether to write the app's own assets under `static/`
    """
    written = export_static(
        dataset,
        Path(out_dir).expanduser(),
        ep_table_cfg=ep_table_cfg,
        group_tables=group_tables,
        home_page=home_page,
        max_resolution=max_resolution,
        max_hz=max_hz,
        base_href=base_href,
        title=title,
        show_paths=show_paths,
        build_id=build_id,
        assets=assets,
    )
    logging.info(f'{len(written)} files, {sum(file.size for file in written) / 1e6:.1f} MB, under {out_dir}')


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
