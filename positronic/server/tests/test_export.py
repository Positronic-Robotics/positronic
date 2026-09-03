"""What one export writes, where, and what it keeps off a page."""

import json
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from positronic import keys
from positronic.dataset.episode import Episode, EpisodeContainer
from positronic.dataset.local_dataset import LocalDatasetWriter, load_all_datasets
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.transforms.episode import EpisodeTransform
from positronic.server.export import (
    GROUP_INDEX_FILE,
    UNFILTERED_FILE,
    asset_content_type,
    export_static,
    filter_sets,
    large_file_links_under,
)
from positronic.server.positronic_server import FILTER_VALUES, GROUP_FILTERS, ColumnConfig, GroupTableConfig, app_state

OUTCOME = 'eval.outcome'
OBJECT = 'eval.object'
HIDDEN_KEY = 'checkpoint_path'
HIDDEN_VALUE = 's3://internal/checkpoints/step_1000'
OBJECTS = ('Plastic banana', 'Wooden block')

TABLE = {'__index__': ColumnConfig(label='#', format='%d'), OUTCOME: ColumnConfig(label='Outcome', filter=True)}
GROUPS = {
    'outcomes': GroupTableConfig(
        group_keys=OUTCOME,
        group_fn=lambda episodes: {'count': len(episodes)},
        format_table={OUTCOME: ColumnConfig(label='Outcome'), 'count': ColumnConfig(label='Episodes', format='%d')},
        group_filter_keys={OBJECT: 'Object'},
    )
}


class _KeepStatic(EpisodeTransform):
    def __init__(self, kept: tuple[str, ...]):
        self._kept = frozenset(kept)

    def __call__(self, episode: Episode) -> Episode:
        static = {name: value for name, value in episode.static.items() if name in self._kept}
        return EpisodeContainer({**episode.signals, **static}, meta=episode.meta)


@pytest.fixture
def dataset(tmp_path):
    """Two episodes; one static value is long enough to become a download, one is hidden."""
    root = tmp_path / 'dataset'
    with LocalDatasetWriter(root) as writer:
        for index, object_name in enumerate(OBJECTS):
            with writer.new_episode() as episode:
                episode.set_static(keys.TASK, 'Put the banana on the plate')
                episode.set_static(OUTCOME, 'Success' if index == 0 else 'Failure')
                episode.set_static(OBJECT, object_name)
                episode.set_static('notes', 'n' * 2000)
                episode.set_static(HIDDEN_KEY, HIDDEN_VALUE)
                for step in range(4):
                    episode.append('robot_state.q', np.zeros(7, dtype=np.float32), ts_ns=10_000 + step * 1_000)
    return load_all_datasets(root)


def shown(dataset):
    return TransformedDataset(dataset, _KeepStatic((keys.TASK, OUTCOME, OBJECT, 'notes')))


def an_export(dataset, out, **overrides):
    settings = {'ep_table_cfg': TABLE, 'max_resolution': 64, 'assets': False, 'full_dataset': dataset}
    return export_static(shown(dataset), out, **{**settings, **overrides})


def paths_of(files) -> set[str]:
    return {str(file.path) for file in files}


def test_a_page_is_a_directory_with_an_index_file_and_an_api_response_is_json(dataset, tmp_path):
    written = paths_of(an_export(dataset, tmp_path / 'out'))

    assert {'index.html', 'episodes/index.html', 'episode/0/index.html', 'episode/1/index.html'} <= written
    assert {'api/episodes.json', 'api/dataset_info.json', 'api/dataset_status.json'} <= written


def test_every_file_named_is_on_disk_and_nothing_else_is(dataset, tmp_path):
    out = tmp_path / 'out'
    written = an_export(dataset, out)

    on_disk = {str(PurePosixPath(path.relative_to(out))) for path in out.rglob('*') if path.is_file()}
    assert on_disk == paths_of(written)
    assert all(file.size == (out / file.path).stat().st_size for file in written)


def test_a_group_table_gets_one_file_per_filter_set_and_an_index_naming_them(dataset, tmp_path):
    out = tmp_path / 'out'
    an_export(dataset, out, group_tables=GROUPS)

    index = json.loads((out / 'api/groups/outcomes' / GROUP_INDEX_FILE).read_text())
    assert index[0] == {'params': {}, 'file': UNFILTERED_FILE}
    assert [entry['params'] for entry in index[1:]] == [{OBJECT: OBJECTS[0]}, {OBJECT: OBJECTS[1]}]
    for entry in index:
        assert (out / 'api/groups/outcomes' / entry['file']).is_file()


def test_a_filtered_group_file_holds_only_that_filter_s_episodes(dataset, tmp_path):
    out = tmp_path / 'out'
    an_export(dataset, out, group_tables=GROUPS)

    index = json.loads((out / 'api/groups/outcomes' / GROUP_INDEX_FILE).read_text())
    block = next(entry for entry in index if entry['params'] == {OBJECT: 'Wooden block'})
    table = json.loads((out / 'api/groups/outcomes' / block['file']).read_text())
    outcome = [column['key'] for column in table['columns']].index(OUTCOME)
    assert [row[1][outcome] for row in table['episodes']] == ['Failure']


def test_a_build_id_moves_the_recordings_and_the_downloads_under_it(dataset, tmp_path):
    out = tmp_path / 'out'
    written = paths_of(an_export(dataset, out, build_id='bld'))

    page = (out / 'episode/0/index.html').read_text()
    assert '"build/bld/api/episode_rrd/0"' in page
    assert '"build/bld/api/episode/0/static/notes"' in page
    assert 'build/bld/api/episode_rrd/0' in written
    assert 'build/bld/api/episode/0/static/notes' in written
    assert (out / 'build/bld/api/episode/0/static/notes').read_bytes() == b'n' * 2000


def test_without_a_build_id_the_large_files_sit_at_their_routes(dataset, tmp_path):
    out = tmp_path / 'out'
    written = paths_of(an_export(dataset, out))

    page = (out / 'episode/0/index.html').read_text()
    assert '"api/episode_rrd/0"' in page and 'build/' not in page
    assert {'api/episode_rrd/0', 'api/episode_rrd/1', 'api/episode/0/static/notes'} <= written


def test_a_recording_is_built_from_the_full_dataset(dataset, tmp_path):
    """The recording builder reads an episode's robot model out of its static values."""
    out = tmp_path / 'out'
    files = {str(file.path): file for file in an_export(dataset, out)}

    assert files['api/episode_rrd/0'].size > 0
    assert files['api/episode_rrd/0'].content_type == 'application/octet-stream'


def test_a_hidden_static_value_reaches_no_page_and_no_file(dataset, tmp_path):
    out = tmp_path / 'out'
    an_export(dataset, out, group_tables=GROUPS)

    for path in out.rglob('*'):
        if path.is_file() and not str(path).endswith('episode_rrd/0') and not str(path).endswith('episode_rrd/1'):
            assert HIDDEN_VALUE.encode() not in path.read_bytes(), path
            assert b'/dataset/' not in path.read_bytes(), path


def test_a_prefix_and_a_title_reach_every_page(dataset, tmp_path):
    out = tmp_path / 'out'
    an_export(dataset, out, base_href='/v/tok/', title='A run')

    for page in ('index.html', 'episodes/index.html', 'episode/1/index.html'):
        body = (out / page).read_text()
        assert '<base href="/v/tok/" />' in body
        assert 'A run' in body
        assert 'window.STATIC_EXPORT = true;' in body


def test_the_assets_are_written_only_when_asked(dataset, tmp_path):
    without = paths_of(an_export(dataset, tmp_path / 'bare'))
    with_assets = paths_of(an_export(dataset, tmp_path / 'whole', assets=True))

    assert not [path for path in without if path.startswith('static/')]
    assert 'static/app.js' in with_assets and 'static/styles.css' in with_assets
    assert any(path.startswith('static/rerun/') and path.endswith('.wasm') for path in with_assets)


def test_every_filter_set_a_page_can_ask_for_is_listed_with_the_empty_one_first():
    response = {GROUP_FILTERS: {'b': {FILTER_VALUES: ['2', '1']}, 'a': {FILTER_VALUES: ['x']}}}

    assert list(filter_sets(response)) == [
        {},
        {'b': '1'},
        {'b': '2'},
        {'a': 'x'},
        {'a': 'x', 'b': '1'},
        {'a': 'x', 'b': '2'},
    ]


def test_a_filter_value_no_episode_carries_asks_for_no_file():
    """The server matches a filter against an episode's own value, and never against an absent one."""
    assert list(filter_sets({GROUP_FILTERS: {'a': {FILTER_VALUES: [None, 'x']}}})) == [{}, {'a': 'x'}]


def test_a_link_is_moved_under_the_build_and_a_page_route_is_not():
    page = 'appUrl("api/episode_rrd/3") "api/episode/3/static/scene" href="episodes" "api/episodes.json"'

    moved = large_file_links_under(page, 'bld')

    assert '"build/bld/api/episode_rrd/3"' in moved
    assert '"build/bld/api/episode/3/static/scene"' in moved
    assert 'href="episodes"' in moved and '"api/episodes.json"' in moved
    assert large_file_links_under(page, '') == page


def test_an_export_leaves_the_app_state_as_it_found_it(dataset, tmp_path):
    before = dict(app_state)
    an_export(dataset, tmp_path / 'out', base_href='/v/tok/', title='A run')

    assert app_state == before


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        ('viewer_bg.wasm', 'application/wasm'),
        ('episode_0.rrd', 'application/octet-stream'),
        ('app.js', 'text/javascript'),
        ('styles.css', 'text/css'),
    ],
)
def test_an_asset_is_served_under_the_type_a_browser_needs(name, expected):
    assert asset_content_type(Path(name)) == expected
