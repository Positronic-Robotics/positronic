"""Write the viewer for one dataset as static files.

The app is composed in this process and read with a test client. A page lands at
`<route>/index.html` and an API response at `api/<route>.json`.
"""

import itertools
import json
import logging
import mimetypes
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from multiprocessing.pool import ThreadPool
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
    API_FILE_SUFFIX,
    API_ROUTE,
    ASSET_ROUTE,
    DOWNLOAD_LINK,
    GROUP_INDEX_FILE,
    MAX_COMPONENT_BYTES,
    GroupFile,
    GroupTableConfig,
    TableConfig,
    app,
    app_state_restored,
    configure_pages,
    configure_tables,
    default_table,
    download_at,
    download_link,
    download_metadata,
    download_paths,
    ensure_episode_rrd,
    episode_link,
    episode_rrd_link,
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
# The recordings and the downloads sit under `build/<build_id>/`, a path a rebuild never rewrites.
BUILD_DIR = 'build'
# The file of a group table read with no filter, beside the index that names it.
UNFILTERED_FILE = 'all.json'
# An episode satisfies up to 2^k filter sets for k filter keys.
MAX_FILTER_KEYS_PER_GROUP = 6
# Each file of a group table is one read of the whole dataset.
MAX_FILTER_SETS_PER_GROUP = 1024
# The object key limit of an S3-style host.
MAX_KEY_BYTES = 1024


def _available_cpus() -> int:
    """The CPUs this process may run on; a cgroup quota without a cpuset is not visible to a process."""
    if hasattr(os, 'sched_getaffinity'):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 2


# A recording's build holds about two cores, so this many at once fill the machine.
DEFAULT_WORKERS = max(1, _available_cpus() // 2)
# Windows reports no path limit; its `MAX_PATH` counts UTF-16 units, the end mark included, unless a machine opts
# into long paths.
_REPORTS_PATH_MAX = hasattr(os, 'pathconf')
_WINDOWS_PATH_MAX = 260
# Windows reads these as devices, with or without a suffix, and trims a trailing dot off a name.
_WINDOWS_DEVICES = frozenset([
    'con',
    'prn',
    'aux',
    'nul',
    *(f'com{n}' for n in range(1, 10)),
    *(f'lpt{n}' for n in range(1, 10)),
])

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


# `mimetypes` answers for neither on every box.
_CONTENT_TYPE_BY_SUFFIX = {'.wasm': 'application/wasm', '.rrd': 'application/octet-stream'}


def asset_content_type(path: Path) -> str:
    """The `Content-Type` an asset file is served under."""
    if path.suffix in _CONTENT_TYPE_BY_SUFFIX:
        return _CONTENT_TYPE_BY_SUFFIX[path.suffix]
    return mimetypes.guess_type(path.name)[0] or 'application/octet-stream'


def _path_max(directory: Path) -> int:
    """The longest path, its end mark included, the filesystem under `directory` takes: read off the nearest
    ancestor that exists where the platform reports it, and Windows' `MAX_PATH` where it does not."""
    if not _REPORTS_PATH_MAX:
        return _WINDOWS_PATH_MAX
    existing = next(candidate for candidate in (directory, *directory.parents) if candidate.exists())
    return os.pathconf(existing, 'PC_PATH_MAX')


def _path_length(path: Path) -> int:
    """`path` as its filesystem's limit counts it: bytes, or UTF-16 units under Windows' `MAX_PATH`."""
    return len(os.fsencode(path)) if _REPORTS_PATH_MAX else len(str(path).encode('utf-16-le')) // 2


class _Output:
    """The files the export writes under `directory`, as any host or filesystem holds them.

    Every path is planned before the first write, a write of a path outside the plan is refused, and `plan` refuses:
    - a key past a host's limit, with `key_prefix`, the base href, in front;
    - a local path past the filesystem's limit, with `directory` in front;
    - a component Windows reads as a device or trims;
    - a path that folds onto a planned file or onto a directory above one, or whose directory folds onto a file.
    """

    def __init__(self, directory: Path, key_prefix: str = ''):
        # Absolute, so a path is measured with the working directory in front, as the filesystem measures it.
        self.directory = directory.absolute()
        self._key_prefix = key_prefix
        self.files: list[ExportedFile] = []
        self._path_max = _path_max(self.directory)
        self._planned: set[PurePosixPath] = set()
        self._folded_files: set[PurePosixPath] = set()
        self._folded_directories: set[PurePosixPath] = set()

    def plan(self, paths: Iterable[PurePosixPath]) -> None:
        """Register every path the export writes, before the first write."""
        for path in paths:
            if path.is_absolute() or '\\' in str(path) or '..' in path.parts:
                raise ValueError(f'{path} would land outside {self.directory}')
            key = self._key_prefix + str(path)
            if len(key.encode()) > MAX_KEY_BYTES:
                raise ValueError(f'{key} is past the {MAX_KEY_BYTES}-byte key limit of a host')
            local = self.directory.joinpath(*path.parts)
            if _path_length(local) >= self._path_max:
                raise ValueError(f'{local} is past the path limit of its filesystem, {self._path_max}')
            folded = PurePosixPath(str(path).casefold())
            for part in folded.parts:
                if part.partition('.')[0] in _WINDOWS_DEVICES or part.endswith('.'):
                    raise ValueError(f'{path} has a component Windows reads as a device or trims, {part!r}')
            directories = [parent for parent in folded.parents if parent.parts]
            taken = folded in self._folded_files or folded in self._folded_directories
            if taken or any(directory in self._folded_files for directory in directories):
                raise ValueError(f'{path} is one file with another the export writes on a filesystem that folds case')
            self._folded_files.add(folded)
            self._folded_directories.update(directories)
            self._planned.add(path)

    def write(self, path: PurePosixPath, body: bytes, content_type: str) -> ExportedFile:
        target = self._target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
        return self._record(path, content_type, len(body))

    def copy(self, path: PurePosixPath, source: Path) -> ExportedFile:
        """Copy the file at `source` to `path` without holding it in memory."""
        target = self._target(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return self._record(path, asset_content_type(source), target.stat().st_size)

    def _target(self, path: PurePosixPath) -> Path:
        if path not in self._planned:
            raise ValueError(f'{path} is not in the export plan')
        return self.directory.joinpath(*path.parts)

    def _record(self, path: PurePosixPath, content_type: str, size: int) -> ExportedFile:
        written = ExportedFile(path, content_type, size)
        self.files.append(written)
        return written


@dataclass(frozen=True)
class _Planned:
    """One file the export writes at `path`; `write` writes it while the app serves `reads`."""

    path: PurePosixPath
    reads: Dataset
    write: Callable[[_Output], ExportedFile]


def _fetch(client: TestClient, path: str, params: dict[str, str] | None = None) -> tuple[bytes, str]:
    response = client.get(path, params=params or {})
    response.raise_for_status()
    return response.content, response.headers.get('content-type', '')


def _planned_fetch(
    client: TestClient, route: str, path: PurePosixPath, reads: Dataset, params: dict[str, str] | None = None
) -> _Planned:
    """The plan to write the response of `route`, read with `params` while the app serves `reads`, at `path`."""
    return _Planned(path, reads, lambda out: out.write(path, *_fetch(client, route, params)))


def filter_sets(episode_values: Iterable[Mapping[str, str]], most: int) -> list[dict[str, str]]:
    """Every non-empty filter set some episode satisfies, the shortest first; more than `most` of them is
    refused at the episode that passes the count, before the rest are read.

    An episode satisfies each subset of its own values, so k filter keys give at most 2^k sets per
    episode, and a set no episode satisfies gets no file.
    """
    satisfied: set[tuple[tuple[str, str], ...]] = set()
    for values in episode_values:
        items = sorted(values.items())
        satisfied.update(chosen for n in range(1, len(items) + 1) for chosen in itertools.combinations(items, n))
        if len(satisfied) > most:
            raise ValueError(f'more than {most} filter sets; a filter key with a value per episode is the usual cause')
    return [dict(chosen) for chosen in sorted(satisfied, key=lambda chosen: (len(chosen), chosen))]


def _large_file_path(link: str, build_id: str) -> PurePosixPath:
    """Where the file a large-file `link` names is written: its segments as a directory tree, each keeping the
    encoded spelling the browser asks it by, under the build."""
    return PurePosixPath(BUILD_DIR, build_id, link) if build_id else PurePosixPath(link)


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
        moved = _host_spelling(str(_large_file_path(link, build_id)))
        for node, moved_node in zip(_link_nodes(link), _link_nodes(moved), strict=True):
            html = html.replace(node, moved_node)
    return html


@dataclass(frozen=True)
class _EpisodeLinks:
    """The large files one episode page links, as the page spells them."""

    index: int
    recording_link: str
    download_links: list[str]


def _episode_links(dataset: Dataset, index: int) -> _EpisodeLinks:
    static = cast(Episode, dataset[index]).static
    downloads = [download_link(index, field) for field in download_paths(static)]
    return _EpisodeLinks(index, episode_rrd_link(index), downloads)


def _episode_page(client: TestClient, reads: Dataset, links: _EpisodeLinks, build_id: str) -> _Planned:
    """The plan to write the page of one episode, with its links moved under the build."""
    route = episode_link(links.index)
    path = PurePosixPath(route) / PAGE_FILE
    every = [links.recording_link, *links.download_links]

    def write(out: _Output) -> ExportedFile:
        body, content_type = _fetch(client, f'/{route}')
        page = body.decode()
        if not all(any(node in page for node in _link_nodes(link)) for link in every):
            raise RuntimeError(f'episode page {links.index} does not carry every link the export expects')
        return out.write(path, large_file_links_under(page, every, build_id).encode(), content_type)

    return _Planned(path, reads, write)


def _page_plans(
    client: TestClient, reads: Dataset, group_names: Iterable[str], episodes: Iterable[_EpisodeLinks], build_id: str
) -> list[_Planned]:
    """The plans to write every page."""
    routes = [episodes_link(), *(group_link(name) for name in group_names)]
    return [
        _planned_fetch(client, '/', PurePosixPath(PAGE_FILE), reads),
        *(_planned_fetch(client, f'/{route}', PurePosixPath(route) / PAGE_FILE, reads) for route in routes),
        *(_episode_page(client, reads, links, build_id) for links in episodes),
    ]


def _group_plans(client: TestClient, reads: Dataset, name: str, sets: Iterable[dict[str, str]]) -> list[_Planned]:
    """The plans to write one file per filter set of the group table `name`, and the index naming them."""
    route = PurePosixPath(group_api_link(name))
    index = [GroupFile({}, UNFILTERED_FILE), *(GroupFile(params, f'{n}.json') for n, params in enumerate(sets, 1))]
    listing = json.dumps([asdict(entry) for entry in index]).encode()
    return [
        *(_planned_fetch(client, f'/{route}', route / entry.file, reads, entry.params) for entry in index),
        _Planned(
            route / GROUP_INDEX_FILE,
            reads,
            lambda out: out.write(route / GROUP_INDEX_FILE, listing, 'application/json'),
        ),
    ]


def _large_file_plans(client: TestClient, reads: Dataset, links: _EpisodeLinks, build_id: str) -> list[_Planned]:
    """The plans to write the recording and the downloads of one episode, under the build; the recording is copied
    from the file the app builds, the downloads are read through the client."""
    recording = _large_file_path(links.recording_link, build_id)
    downloads = [(link, _large_file_path(link, build_id)) for link in links.download_links]
    return [
        _Planned(recording, reads, lambda out: out.copy(recording, ensure_episode_rrd(links.index))),
        *(_planned_fetch(client, f'/{link}', path, reads) for link, path in downloads),
    ]


def _asset_files() -> list[tuple[PurePosixPath, Path]]:
    """The app's own scripts, styles and viewer, each with its path under `static/`."""
    static_dir = Path(__file__).resolve().parent / ASSET_ROUTE
    files = sorted(p for p in static_dir.rglob('*') if p.is_file())
    return [(PurePosixPath(ASSET_ROUTE) / file.relative_to(static_dir).as_posix(), file) for file in files]


def _whole_api_routes() -> list[str]:
    """The API routes a page reads whole: every GET under the API segment that takes no path parameter.

    The flat episode table is one of them; a static page filters it in the browser.
    """
    return [
        route.path.removeprefix('/')
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith(f'/{API_ROUTE}/') and not route.param_convertors
    ]


def _full_checked_against(full: Dataset, shown: Dataset) -> Dataset:
    """`full`, once it holds the episodes of `shown` at the same indexes, by uid, with every download `shown` links,
    of the type and the size the page reports beside the link."""
    if len(full) != len(shown):
        raise ValueError(f'full_dataset holds {len(full)} episodes and dataset holds {len(shown)}')
    for index in range(len(shown)):
        full_episode, shown_episode = cast(Episode, full[index]), cast(Episode, shown[index])
        if full_episode.meta[META_UID] != shown_episode.meta[META_UID]:
            raise ValueError(
                f'full_dataset episode {index} has uid {full_episode.meta[META_UID]!r} and dataset has '
                f'{shown_episode.meta[META_UID]!r}'
            )
        for path in download_paths(shown_episode.static):
            linked = download_metadata(cast(bytes | str, download_at(shown_episode.static, path)))
            held = download_at(full_episode.static, path)
            if held is None:
                raise ValueError(
                    f'full_dataset episode {index} has no download at {"/".join(path)!r}, which dataset links'
                )
            if download_metadata(held) != linked:
                raise ValueError(
                    f'full_dataset episode {index} holds {download_metadata(held).type} of size '
                    f'{download_metadata(held).size} at {"/".join(path)!r}, and dataset links {linked.type} of size '
                    f'{linked.size}'
                )
    return full


def _write_serving(out: _Output, plans: Iterable[_Planned]) -> None:
    """Write each of `plans` with the app serving the dataset it reads."""
    serving: Dataset | None = None
    for planned in plans:
        if planned.reads is not serving:
            install_dataset(planned.reads)
            serving = planned.reads
        planned.write(out)


def _filter_sets_by_group(
    dataset: Dataset, group_tables: dict[str, GroupTableConfig] | None
) -> dict[str, list[dict[str, str]]]:
    """The non-empty filter sets each group table gets a file for; a group past either bound is refused."""
    sets_by_group: dict[str, list[dict[str, str]]] = {}
    for name, cfg in (group_tables or {}).items():
        if len(cfg.group_filter_keys) > MAX_FILTER_KEYS_PER_GROUP:
            raise ValueError(
                f'group table {name!r} has {len(cfg.group_filter_keys)} filter keys; an export writes up to 2^k '
                f'files per episode, so a group table takes at most {MAX_FILTER_KEYS_PER_GROUP}'
            )
        # Each episode's values on the group's filter keys, as a filter spells them; an absent value is left out.
        episode_values = (
            {
                key: spelling
                for key in cfg.group_filter_keys
                if (spelling := filter_spelling(cast(Episode, episode).static.get(key))) is not None
            }
            for episode in dataset
        )
        try:
            # One of the files is the unfiltered one.
            sets_by_group[name] = filter_sets(episode_values, MAX_FILTER_SETS_PER_GROUP - 1)
        except ValueError as past_bound:
            raise ValueError(
                f'group table {name!r} has {past_bound}; an export reads the dataset once per set'
            ) from None
    return sets_by_group


def _build_recordings(reads: Dataset, workers: int) -> None:
    """Build the recording of each episode of `reads` into the cache, `workers` at a time, with the app serving
    `reads`; one worker builds them on the calling thread.

    The workers are threads: the decoder and the encoder release the interpreter lock, and a forked worker
    hangs on the threads a recording built earlier in the process leaves behind.
    """
    if workers < 1:
        raise ValueError(f'workers={workers}; a recording is built by at least one')
    install_dataset(reads)
    if workers == 1:
        for index in range(len(reads)):
            ensure_episode_rrd(index)
        return
    with ThreadPool(workers) as pool:
        pool.map(ensure_episode_rrd, range(len(reads)), chunksize=1)


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
    workers: int = DEFAULT_WORKERS,
) -> list[ExportedFile]:
    """Write the viewer for `dataset` under `out_dir` and give back every file written.

    The pages and the tables read `dataset`. The recordings and the downloads read `full_dataset`
    when given, which holds the same episodes in the same order; the recording builder reads an
    episode's robot model out of its static values. The recordings are built `workers` at a time in a
    directory of this export's own under `scratch_dir`, the system's temporary directory when None,
    copied in from there, and the directory is removed at the end. `assets` writes the app's own scripts, styles
    and viewer under `static/`, which the pages request at the host root, so it goes with the root
    base href only; an export under a prefix shares the host's copy. An export holds the app's state
    for its duration, so a second export in the process waits for it; one into the same directory is
    then refused, as the directory holds the first.
    """
    out = _Output(Path(out_dir), normalized_base_href(base_href).removeprefix('/'))
    if scratch_dir is not None and Path(scratch_dir).resolve().is_relative_to(out.directory.resolve()):
        raise ValueError(
            f'scratch_dir {scratch_dir} lies inside out_dir {out.directory}; the recordings are built beside it'
        )
    if assets and normalized_base_href(base_href) != '/':
        raise ValueError('assets sit under static/ at the host root; an export under a prefix takes assets=False')
    validated_build_id(build_id)
    shown = CachedDataset(dataset)
    sets_by_group = _filter_sets_by_group(shown, group_tables)
    full = _full_checked_against(CachedDataset(full_dataset), shown) if full_dataset is not None else shown
    episodes = [_episode_links(shown, index) for index in range(len(shown))]
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
        client = TestClient(app)
        # Past `configure_tables`, every group name is one segment the route builders spell.
        plans = [
            *_page_plans(client, shown, sets_by_group, episodes, build_id),
            *(
                _planned_fetch(client, f'/{route}', PurePosixPath(route + API_FILE_SUFFIX), shown)
                for route in _whole_api_routes()
            ),
            *itertools.chain.from_iterable(
                _group_plans(client, shown, name, sets) for name, sets in sets_by_group.items()
            ),
            *itertools.chain.from_iterable(_large_file_plans(client, full, links, build_id) for links in episodes),
        ]
        asset_files = _asset_files() if assets else []
        out.plan(itertools.chain((planned.path for planned in plans), (path for path, _ in asset_files)))
        _build_recordings(full, workers)
        _write_serving(out, plans)
    for path, file in asset_files:
        out.copy(path, file)
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
    workers: int = DEFAULT_WORKERS,
):
    """Write the viewer for a Dataset as static files, for any static host.

    Args:
        dataset: Dataset to export
        out_dir: Directory the files are written under; it is created
        ep_table_cfg: Columns of the episode table, by static key
        max_resolution: Long side an episode's videos are re-encoded down to
        max_hz: Rate an episode's videos and numeric signals are thinned to; 0 keeps every frame and sample
        group_tables: Grouped tables, by name
        home_page: The group table served at the root, or None for the episodes
        base_href: Path at the host root the export is served under
        title: Header text; the dataset root when empty
        show_paths: Whether the pages report where the dataset lives
        build_id: Names one build; the recordings and the downloads are written under `build/<build_id>/`
        assets: Whether to write the app's own assets under `static/`; with the root base href only
        workers: Recordings built at once; half the machine's cores by default
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
        workers=workers,
    )
    logging.info(f'{len(written)} files, {sum(file.size for file in written) / 1e6:.1f} MB, under {out_dir}')


@pos3.with_mirror()
def _internal_main():
    init_logging()
    cfn.cli(main)


if __name__ == '__main__':
    _internal_main()
