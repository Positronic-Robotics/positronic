"""Write the viewer for one dataset as static files.

The app is composed in this process and read with a test client, so the export holds what the
server answers. A page lands at `<route>/index.html` and an API response at `api/<route>.json`.
"""

import itertools
import json
import logging
import mimetypes
import re
import shutil
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import quote

import configuronic as cfn
import pos3
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from jinja2.utils import htmlsafe_json_dumps

import positronic.cfg.ds
from pimm.logging import init_logging
from positronic.dataset import CachedDataset, Dataset, Episode
from positronic.dataset.episode import META_UID
from positronic.server.dataset_utils import DEFAULT_MAX_HZ, DEFAULT_MAX_RESOLUTION, get_dataset_root
from positronic.server.positronic_server import (
    DOWNLOAD_LINK,
    MAX_COMPONENT_BYTES,
    GroupTableConfig,
    TableConfig,
    app,
    app_state_restored,
    configure_pages,
    configure_tables,
    default_table,
    download_link,
    download_paths,
    episode_link,
    episode_rrd_link,
    episode_rrd_path,
    episodes_link,
    filter_spelling,
    group_api_link,
    group_link,
    install_dataset,
    normalized_base_href,
)

logger = logging.getLogger(__name__)

# A page is a directory with an index file, so a static host answers `episode/3` with it.
PAGE_FILE = 'index.html'
API_DIR = 'api'
# The recordings and the downloads sit under `build/<build_id>/`, a path a rebuild never rewrites.
BUILD_DIR = 'build'
# A group table is one file per filter set, and the index beside them says which file holds which.
GROUP_INDEX_FILE = 'index.json'
UNFILTERED_FILE = 'all.json'
# A group table is one file per filter set some episode satisfies, up to 2^k per episode for k filter keys.
MAX_FILTER_KEYS_PER_GROUP = 6
# Each file of a group table is one read of the whole dataset.
MAX_FILTER_SETS_PER_GROUP = 1024
# The object key limit of an S3-style host; with an output directory in front it stays within a filesystem's path limit.
MAX_PATH_BYTES = 1024
# Windows reads these as devices, with or without a suffix, and trims a trailing dot off a name.
_WINDOWS_DEVICES = frozenset([
    'con',
    'prn',
    'aux',
    'nul',
    *(f'com{n}' for n in range(1, 10)),
    *(f'lpt{n}' for n in range(1, 10)),
])
# The app's own assets, at the server root, so every export a host serves shares one copy.
ASSET_DIR = 'static'

# The `secrets.token_urlsafe` alphabet: a build id is a path segment and sits inside a script string.
_BUILD_ID = re.compile(r'[A-Za-z0-9_-]*')


def validated_build_id(value: str) -> str:
    """`value` when a path, a file name and a page can carry it as it is; empty names no build."""
    if not _BUILD_ID.fullmatch(value) or len(value) > MAX_COMPONENT_BYTES:
        raise ValueError(f'build_id must match {_BUILD_ID.pattern!r} within {MAX_COMPONENT_BYTES} bytes, got {value!r}')
    return value


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


class _PortableTree:
    """The paths written so far, as any host or filesystem the tree lands on holds them.

    A path past a host's key limit is refused. So is a component Windows reads as a device or trims, and a
    path that folds onto a file written before, or onto a directory above one, or whose own directory folds
    onto a file: the export writes one tree, wherever it lands.
    """

    def __init__(self):
        self._files: set[str] = set()
        self._directories: set[str] = set()

    def add(self, path: str) -> None:
        if len(path.encode()) > MAX_PATH_BYTES:
            raise ValueError(f'{path!r} is past the {MAX_PATH_BYTES}-byte key limit of a host')
        folded = PurePosixPath(path.casefold())
        for part in folded.parts:
            if part.partition('.')[0] in _WINDOWS_DEVICES or part.endswith('.'):
                raise ValueError(f'{path!r} has a component Windows reads as a device or trims, {part!r}')
        directories = [str(parent) for parent in folded.parents if str(parent) != '.']
        taken = str(folded) in self._files or str(folded) in self._directories
        if taken or any(directory in self._files for directory in directories):
            raise ValueError(f'{path!r} is one file with another the export writes on a filesystem that folds case')
        self._files.add(str(folded))
        self._directories.update(directories)


class _Output:
    def __init__(self, directory: Path):
        self.directory = directory
        self.files: list[ExportedFile] = []
        self._tree = _PortableTree()

    def target(self, path: str) -> Path:
        """The file `path` is written to; a path that would land outside the directory, or that a host or a
        filesystem the tree lands on does not hold as it is, is refused."""
        relative = PurePosixPath(path)
        if relative.is_absolute() or '\\' in path or '..' in relative.parts:
            raise ValueError(f'{path!r} would land outside {self.directory}')
        self._tree.add(path)
        return self.directory.joinpath(*relative.parts)

    def write(self, path: str, body: bytes, content_type: str) -> ExportedFile:
        target = self.target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return self._record(path, content_type, len(body))

    def copy(self, path: str, source: Path) -> ExportedFile:
        """Copy the file at `source` to `path` without holding it in memory."""
        target = self.target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return self._record(path, asset_content_type(source), target.stat().st_size)

    def _record(self, path: str, content_type: str, size: int) -> ExportedFile:
        written = ExportedFile(PurePosixPath(path), content_type, size)
        self.files.append(written)
        return written


def _fetch(client: TestClient, path: str, params: dict[str, str] | None = None) -> tuple[bytes, str]:
    response = client.get(path, params=params or {})
    response.raise_for_status()
    return response.content, response.headers.get('content-type', '')


def _filter_values(dataset: Dataset, keys: Iterable[str]) -> Iterator[dict[str, str]]:
    """Each episode's values on the filter `keys`, as a filter spells them; an absent value is left out."""
    for episode in dataset:
        spelled = ((key, filter_spelling(cast(Episode, episode).static.get(key))) for key in keys)
        yield {key: value for key, value in spelled if value is not None}


def filter_sets(episode_values: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    """Every filter set some episode satisfies, the empty one first.

    An episode satisfies each subset of its own values, so k filter keys give at most 2^k sets per
    episode, and a set no episode satisfies gets no file.
    """
    satisfied: set[tuple[tuple[str, str], ...]] = set()
    for values in episode_values:
        items = sorted(values.items())
        satisfied.update(chosen for n in range(len(items) + 1) for chosen in itertools.combinations(items, n))
    return [dict(chosen) for chosen in sorted(satisfied, key=lambda chosen: (len(chosen), chosen))]


def _large_file_path(link: str, build_id: str) -> str:
    return f'{BUILD_DIR}/{build_id}/{link}' if build_id else link


def _on_disk(link: str, build_id: str) -> str:
    """Where the file a large-file `link` names is written: its segments as a directory tree, each keeping the
    encoded spelling the browser asks it by, under the build."""
    return _large_file_path(link, build_id)


def _page_spelling(link: str) -> str:
    """`link` as a page carries it: a quoted JSON string, escaped as Jinja's `tojson` writes one."""
    return str(htmlsafe_json_dumps(link))


def _link_nodes(link: str) -> list[str]:
    """The nodes of an episode page that carry `link`: the recording's `appUrl(...)` call, a download's field."""
    spelling = _page_spelling(link)
    return [f'appUrl({spelling})', f'{_page_spelling(DOWNLOAD_LINK)}: {spelling}']


def _host_spelling(path: str) -> str:
    """The path a browser asks a static host for the file at `path`: a host percent-decodes a request path once
    before it looks the file up, so each `%` of the file's own name is escaped."""
    return quote(path, safe='/')


def large_file_links_under(html: str, links: Iterable[str], build_id: str) -> str:
    """`html` with each of `links`, in the nodes a page carries them, moved under `build/<build_id>/` and
    spelled as a static host resolves them to the files on disk."""
    for link in links:
        moved = _host_spelling(_large_file_path(link, build_id))
        for node, moved_node in zip(_link_nodes(link), _link_nodes(moved), strict=True):
            html = html.replace(node, moved_node)
    return html


@dataclass(frozen=True)
class _EpisodeFiles:
    """The large files one episode page links, as the page spells them."""

    index: int
    recording: str
    downloads: list[str]

    @property
    def links(self) -> list[str]:
        return [self.recording, *self.downloads]


def _episode_files(dataset: Dataset, index: int) -> _EpisodeFiles:
    static = cast(Episode, dataset[index]).static
    downloads = [download_link(index, field) for field in download_paths(static)]
    return _EpisodeFiles(index, episode_rrd_link(index), downloads)


def _write_pages(
    client: TestClient, out: _Output, group_names: list[str], episodes: list[_EpisodeFiles], build_id: str
) -> None:
    """Write every page; an episode page carries its links moved under the build."""
    body, content_type = _fetch(client, '/')
    out.write(PAGE_FILE, body, content_type)
    for route in (episodes_link(), *(group_link(name) for name in group_names)):
        body, content_type = _fetch(client, f'/{route}')
        out.write(f'{route}/{PAGE_FILE}', body, content_type)
    for files in episodes:
        body, content_type = _fetch(client, f'/{episode_link(files.index)}')
        page = body.decode()
        if not all(any(node in page for node in _link_nodes(link)) for link in files.links):
            raise RuntimeError(f'episode page {files.index} does not carry every link the export expects')
        moved = large_file_links_under(page, files.links, build_id)
        out.write(f'{episode_link(files.index)}/{PAGE_FILE}', moved.encode(), content_type)


def _write_group(client: TestClient, out: _Output, name: str, sets: Iterable[dict[str, str]]) -> None:
    route = group_api_link(name)
    body, content_type = _fetch(client, f'/{route}')
    index = [GroupFile({}, UNFILTERED_FILE)]
    out.write(f'{route}/{UNFILTERED_FILE}', body, content_type)
    for number, params in enumerate((chosen for chosen in sets if chosen), start=1):
        filtered, filtered_type = _fetch(client, f'/{route}', params)
        entry = GroupFile(params, f'{number}.json')
        out.write(f'{route}/{entry.file}', filtered, filtered_type)
        index.append(entry)
    listing = json.dumps([asdict(entry) for entry in index]).encode()
    out.write(f'{route}/{GROUP_INDEX_FILE}', listing, 'application/json')


# `mimetypes` answers for neither on every box.
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


def _whole_api_routes() -> list[str]:
    """The API routes a page reads whole: every GET under `api/` that takes no path parameter.

    The flat episode table is one of them; a static page filters it in the browser.
    """
    return [
        route.path.removeprefix('/')
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith(f'/{API_DIR}/') and not route.param_convertors
    ]


def _aligned(full: Dataset, shown: Dataset) -> Dataset:
    """`full`, once it holds the episodes of `shown` at the same indexes, by uid, with every download `shown` links."""
    if len(full) != len(shown):
        raise ValueError(f'full_dataset holds {len(full)} episodes and dataset holds {len(shown)}')
    for index in range(len(shown)):
        full_episode, shown_episode = cast(Episode, full[index]), cast(Episode, shown[index])
        if full_episode.meta[META_UID] != shown_episode.meta[META_UID]:
            raise ValueError(
                f'full_dataset episode {index} has uid {full_episode.meta[META_UID]!r} and dataset has '
                f'{shown_episode.meta[META_UID]!r}'
            )
        missing = set(download_paths(shown_episode.static)) - set(download_paths(full_episode.static))
        if missing:
            path = '/'.join(min(missing))
            raise ValueError(f'full_dataset episode {index} has no download at {path!r}, which dataset links')
    return full


def _filter_sets_by_group(
    dataset: Dataset, group_tables: dict[str, GroupTableConfig] | None
) -> dict[str, list[dict[str, str]]]:
    """The filter sets each group table gets a file for; a group past either bound is refused before a write."""
    sets_by_group: dict[str, list[dict[str, str]]] = {}
    for name, cfg in (group_tables or {}).items():
        if len(cfg.group_filter_keys) > MAX_FILTER_KEYS_PER_GROUP:
            raise ValueError(
                f'group table {name!r} has {len(cfg.group_filter_keys)} filter keys; an export writes up to 2^k '
                f'files per episode, so a group table takes at most {MAX_FILTER_KEYS_PER_GROUP}'
            )
        sets = filter_sets(_filter_values(dataset, cfg.group_filter_keys))
        if len(sets) > MAX_FILTER_SETS_PER_GROUP:
            raise ValueError(
                f'group table {name!r} has {len(sets)} filter sets and an export reads the dataset once per set, '
                f'so a group table takes at most {MAX_FILTER_SETS_PER_GROUP}; a filter key with a value per episode '
                f'is the usual cause'
            )
        sets_by_group[name] = sets
    return sets_by_group


def _refuse_unportable_paths(group_names: Iterable[str], episodes: Iterable[_EpisodeFiles], build_id: str) -> None:
    """Refuse a group directory or a large file that a host or a filesystem does not hold as it is, before a write.

    Past `configure_tables`, every group name is one segment the route builder spells.
    """
    named = _PortableTree()
    for name in group_names:
        named.add(group_link(name))
    for files in episodes:
        for link in files.links:
            named.add(_on_disk(link, build_id))


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
    scratch_dir: Path | None = None,
) -> list[ExportedFile]:
    """Write the viewer for `dataset` under `out_dir` and give back every file written.

    The pages and the tables read `dataset`. The recordings and the downloads read `full_dataset`
    when given, which holds the same episodes in the same order; the recording builder reads an
    episode's robot model out of its static values. The recordings are built in a directory of this
    export's own under `scratch_dir`, the system's temporary directory when None, and copied in from
    there, so no recording is held in memory whole and no export reads another's; the directory is
    removed at the end. `assets` writes the app's own scripts, styles and viewer under `static/`, which
    the pages request at the host root, so it goes with the root base href only; an export under a
    prefix shares the host's copy. An export holds the app's state for its duration, so a second
    export in the process waits for it; one into the same directory is then refused, as the directory
    holds the first.
    """
    out = _Output(Path(out_dir))
    if scratch_dir is not None and Path(scratch_dir).resolve().is_relative_to(out.directory.resolve()):
        raise ValueError(
            f'scratch_dir {scratch_dir} lies inside out_dir {out.directory}; the recordings are built beside it'
        )
    if assets and normalized_base_href(base_href) != '/':
        raise ValueError('assets sit under static/ at the host root; an export under a prefix takes assets=False')
    validated_build_id(build_id)
    shown = CachedDataset(dataset)
    sets_by_group = _filter_sets_by_group(shown, group_tables)
    full = _aligned(CachedDataset(full_dataset), shown) if full_dataset is not None else shown
    episodes = [_episode_files(shown, index) for index in range(len(shown))]
    with app_state_restored(), tempfile.TemporaryDirectory(dir=scratch_dir) as scratch:
        # Under the lock, so a second export into the directory of a running one finds it full.
        if out.directory.exists() and any(out.directory.iterdir()):
            raise ValueError(f'{out.directory} is not empty; an export goes into a new or an empty directory')
        configure_tables(
            root=get_dataset_root(dataset) or 'unknown_dataset',
            cache_dir=Path(scratch),
            ep_table_cfg=ep_table_cfg,
            group_tables=group_tables,
            home_page=home_page,
            max_resolution=max_resolution,
            max_hz=max_hz,
        )
        configure_pages(base_href=base_href, title=title, show_paths=show_paths, static_export=True)
        _refuse_unportable_paths(group_tables or {}, episodes, build_id)
        install_dataset(shown)
        client = TestClient(app)
        group_names = list(group_tables or {})

        _write_pages(client, out, group_names, episodes, build_id)
        for route in _whole_api_routes():
            body, content_type = _fetch(client, f'/{route}')
            out.write(f'{route}.json', body, content_type)
        for name, sets in sets_by_group.items():
            _write_group(client, out, name, sets)

        install_dataset(full)
        for files in episodes:
            out.copy(_on_disk(files.recording, build_id), episode_rrd_path(files.index))
            for link in files.downloads:
                body, content_type = _fetch(client, f'/{link}')
                out.write(_on_disk(link, build_id), body, content_type)
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
        ep_table_cfg: Columns of the episode table, by static key
        max_resolution: Long side an episode's videos are re-encoded down to
        max_hz: Rate an episode's numeric signals are thinned to; 0 keeps every sample
        group_tables: Grouped tables, by name
        home_page: The group table served at the root, or None for the episodes
        base_href: Path at the host root the export is served under
        title: Header text; the dataset root when empty
        show_paths: Whether the pages report where the dataset lives
        build_id: Names one build; the recordings and the downloads are written under `build/<build_id>/`
        assets: Whether to write the app's own assets under `static/`; with the root base href only
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
