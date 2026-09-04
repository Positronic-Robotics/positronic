"""What one export writes, where, and what it keeps off a page."""

import json
import threading
from dataclasses import fields
from pathlib import Path, PurePosixPath
from typing import cast

import numpy as np
import pytest

from positronic import keys
from positronic.dataset import Dataset
from positronic.dataset.dataset import FilterDataset
from positronic.dataset.episode import Episode, EpisodeContainer
from positronic.dataset.local_dataset import LocalDatasetWriter, load_all_datasets
from positronic.dataset.transforms import TransformedDataset
from positronic.dataset.transforms.episode import EpisodeTransform
from positronic.server import positronic_server
from positronic.server.export import (
    GROUP_INDEX_FILE,
    UNFILTERED_FILE,
    GroupFile,
    _fetch,
    _Output,
    asset_content_type,
    export_static,
    filter_sets,
    large_file_links_under,
    validated_build_id,
)
from positronic.server.positronic_server import DOWNLOAD_LINK, ColumnConfig, GroupTableConfig, app_state

OUTCOME = 'eval.outcome'
OBJECT = 'eval.object'
ATTEMPT = 'attempt'
HIDDEN_KEY = 'checkpoint_path'
HIDDEN_VALUE = 's3://internal/checkpoints/step_1000'
OBJECTS = ('Plastic banana', 'Wooden block')

TABLE = {
    '__index__': ColumnConfig(label='#', format='%d'),
    OUTCOME: ColumnConfig(label='Outcome', filter=True),
    OBJECT: ColumnConfig(label='Object', filter=True),
    ATTEMPT: ColumnConfig(label='Attempt'),
}
GROUPS = {
    'outcomes': GroupTableConfig(
        group_keys=OUTCOME,
        group_fn=lambda episodes: {'count': len(episodes)},
        format_table={OUTCOME: ColumnConfig(label='Outcome'), 'count': ColumnConfig(label='Episodes', format='%d')},
        group_filter_keys={OBJECT: 'Object'},
    ),
    'attempts': GroupTableConfig(
        group_keys=OUTCOME,
        group_fn=lambda episodes: {'count': len(episodes)},
        format_table={OUTCOME: ColumnConfig(label='Outcome'), 'count': ColumnConfig(label='Episodes', format='%d')},
        group_filter_keys={ATTEMPT: 'Attempt'},
    ),
    'pairs': GroupTableConfig(
        group_keys=OUTCOME,
        group_fn=lambda episodes: {'count': len(episodes)},
        format_table={OUTCOME: ColumnConfig(label='Outcome'), 'count': ColumnConfig(label='Episodes', format='%d')},
        group_filter_keys={OBJECT: 'Object', ATTEMPT: 'Attempt'},
    ),
}


class _KeepStatic(EpisodeTransform):
    def __init__(self, kept: tuple[str, ...]):
        self._kept = frozenset(kept)

    def __call__(self, episode: Episode) -> Episode:
        static = {name: value for name, value in episode.static.items() if name in self._kept}
        return EpisodeContainer({**episode.signals, **static}, meta=episode.meta)


def a_dataset(root: Path, *statics: dict) -> Dataset:
    """One episode per static-value dict, each four joint samples long."""
    with LocalDatasetWriter(root) as writer:
        for static in statics:
            with writer.new_episode() as episode:
                for name, value in static.items():
                    episode.set_static(name, value)
                for step in range(4):
                    episode.append(keys.JOINTS, np.zeros(7, dtype=np.float32), ts_ns=10_000 + step * 1_000)
    return load_all_datasets(root)


@pytest.fixture
def dataset(tmp_path):
    """Two episodes; one static value is long enough to become a download, one is hidden."""
    return a_dataset(
        tmp_path / 'dataset',
        *(
            {
                keys.TASK: 'Put the banana on the plate',
                OUTCOME: 'Success' if index == 0 else 'Failure',
                OBJECT: object_name,
                ATTEMPT: index + 1,
                'notes': 'n' * 2000,
                'artifacts': [b'first blob', 'short'],
                HIDDEN_KEY: HIDDEN_VALUE,
            }
            for index, object_name in enumerate(OBJECTS)
        ),
    )


def shown(dataset):
    return TransformedDataset(dataset, _KeepStatic((keys.TASK, OUTCOME, OBJECT, ATTEMPT, 'notes', 'artifacts')))


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


def test_a_recording_is_copied_from_the_cache_file_and_not_read_through_the_client(dataset, tmp_path, monkeypatch):
    def fetch_no_recording(client, path, params=None):
        assert 'episode_rrd' not in path, path
        return _fetch(client, path, params)

    monkeypatch.setattr('positronic.server.export._fetch', fetch_no_recording)
    cache = tmp_path / 'cache'

    files = an_export(dataset, tmp_path / 'out', cache_dir=cache)

    cached = {path.read_bytes() for path in cache.rglob('*.rrd')}
    written = {(tmp_path / 'out' / path).read_bytes() for path in paths_of(files) if 'episode_rrd' in path}
    assert len(written) == 2 and written == cached


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


def test_every_filter_set_an_episode_satisfies_is_listed_once_with_the_empty_one_first():
    episodes = [{'a': 'x', 'b': '1'}, {'b': '2'}, {'a': 'x', 'b': '1'}]

    assert filter_sets(episodes) == [{}, {'a': 'x'}, {'b': '1'}, {'b': '2'}, {'a': 'x', 'b': '1'}]


def test_a_filter_set_no_episode_satisfies_gets_no_file(dataset, tmp_path):
    out = tmp_path / 'out'
    an_export(dataset, out, group_tables=GROUPS)

    index = json.loads((out / 'api/groups/pairs' / GROUP_INDEX_FILE).read_text())
    assert index[0]['params'] == {}
    assert sorted(json.dumps(entry['params'], sort_keys=True) for entry in index[1:]) == sorted(
        json.dumps(params, sort_keys=True)
        for params in (
            {OBJECT: OBJECTS[0]},
            {OBJECT: OBJECTS[1]},
            {ATTEMPT: '1'},
            {ATTEMPT: '2'},
            {OBJECT: OBJECTS[0], ATTEMPT: '1'},
            {OBJECT: OBJECTS[1], ATTEMPT: '2'},
        )
    )


def test_a_link_is_moved_in_the_nodes_a_page_carries_it_and_an_equal_value_is_not():
    links = ['api/episode_rrd/3', 'api/episode/3/static/scene']
    download = {DOWNLOAD_LINK: 'api/episode/3/static/scene', 'size': 6}
    values = {'twin': 'api/episode_rrd/3', 'note': 'api/episode/3/static/note'}
    page = f'appUrl("api/episode_rrd/3") {json.dumps(download)} {json.dumps(values)} href="episodes"'

    moved = large_file_links_under(page, links, 'bld')

    assert 'appUrl("build/bld/api/episode_rrd/3")' in moved
    assert json.dumps({**download, DOWNLOAD_LINK: 'build/bld/api/episode/3/static/scene'}) in moved
    assert json.dumps(values) in moved and 'href="episodes"' in moved
    assert large_file_links_under(page, links, '') == page


def test_a_static_value_equal_to_a_link_stays_as_it_is(tmp_path):
    dataset = a_dataset(tmp_path / 'dataset', {keys.TASK: 'Put the banana on the plate', 'twin': 'api/episode_rrd/0'})
    out = tmp_path / 'out'

    export_static(dataset, out, ep_table_cfg=TABLE, max_resolution=64, assets=False, build_id='bld')

    page = (out / 'episode/0/index.html').read_text()
    assert 'appUrl("build/bld/api/episode_rrd/0")' in page
    assert json.dumps({'twin': 'api/episode_rrd/0'})[1:-1] in page


def test_a_path_with_a_parent_segment_or_a_backslash_is_refused_before_a_write(tmp_path):
    out = _Output(tmp_path / 'out')

    for path in ('../index.html', '..\\..\\index.html', '/index.html'):
        with pytest.raises(ValueError, match='outside'):
            out.write(path, b'', 'text/html')
    assert not (tmp_path / 'out').exists()


def test_a_static_value_whose_key_a_browser_rewrites_stops_the_export_before_it_writes(tmp_path):
    dataset = a_dataset(tmp_path / 'dataset', {keys.TASK: 'Put the banana on the plate', 'a': {'..': b'a step'}})

    with pytest.raises(ValueError, match='no link'):
        export_static(dataset, tmp_path / 'out', ep_table_cfg=TABLE, max_resolution=64, assets=False)
    assert not (tmp_path / 'out').exists()


def test_a_static_value_named_with_a_backslash_is_linked_encoded_and_written(tmp_path):
    dataset = a_dataset(tmp_path / 'dataset', {keys.TASK: 'Put the banana on the plate', 'models\\scene': b'a mesh'})
    out = tmp_path / 'out'

    written = paths_of(export_static(dataset, out, ep_table_cfg=TABLE, max_resolution=64, assets=False))

    assert '"api/episode/0/static/models%5Cscene"' in (out / 'episode/0/index.html').read_text()
    assert 'api/episode/0/static/models%5Cscene' in written
    assert (out / 'api/episode/0/static/models%5Cscene').read_bytes() == b'a mesh'


def test_a_dotted_top_level_key_and_a_nested_value_that_spell_alike_are_written_apart(tmp_path):
    dataset = a_dataset(
        tmp_path / 'dataset',
        {keys.TASK: 'Put the banana on the plate', 'scene.mesh': b'top', 'scene': {'mesh': b'nested'}},
    )
    out = tmp_path / 'out'

    written = paths_of(export_static(dataset, out, ep_table_cfg=TABLE, max_resolution=64, assets=False))

    assert {'api/episode/0/static/scene.mesh', 'api/episode/0/static/scene/mesh'} <= written
    assert (out / 'api/episode/0/static/scene.mesh').read_bytes() == b'top'
    assert (out / 'api/episode/0/static/scene/mesh').read_bytes() == b'nested'


def test_a_short_key_and_a_slashed_key_that_share_a_prefix_are_written_apart(tmp_path):
    dataset = a_dataset(tmp_path / 'dataset', {keys.TASK: 'Put the banana on the plate', 'a': b'1', 'a/b': b'2'})
    out = tmp_path / 'out'

    written = paths_of(export_static(dataset, out, ep_table_cfg=TABLE, max_resolution=64, assets=False))

    assert {'api/episode/0/static/a', 'api/episode/0/static/a%2Fb'} <= written
    assert (out / 'api/episode/0/static/a').read_bytes() == b'1'
    assert (out / 'api/episode/0/static/a%2Fb').read_bytes() == b'2'


def test_a_cache_inside_the_output_directory_is_refused(dataset, tmp_path):
    out = tmp_path / 'out'

    for cache in (out, out / 'cache'):
        with pytest.raises(ValueError, match='outside'):
            an_export(dataset, out, cache_dir=cache)
    assert not out.exists()


def test_the_assets_go_with_the_root_base_href_only(dataset, tmp_path):
    with pytest.raises(ValueError, match='assets'):
        an_export(dataset, tmp_path / 'out', base_href='/v/tok/', assets=True)
    assert not (tmp_path / 'out').exists()


def test_a_build_id_outside_the_token_alphabet_is_refused(dataset, tmp_path):
    for value in ('../../shared', 'k"3', 'k3/n9', 'k3 n9'):
        with pytest.raises(ValueError, match='build_id'):
            an_export(dataset, tmp_path / 'out', build_id=value)
    assert validated_build_id('k3n9_-A') == 'k3n9_-A'


def test_an_export_refuses_a_directory_that_holds_something(dataset, tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    (out / 'episode').mkdir()
    (out / 'episode' / 'stale').write_text('an older export')

    with pytest.raises(ValueError, match='not empty'):
        an_export(dataset, out)
    assert (out / 'episode' / 'stale').read_text() == 'an older export'


def test_an_empty_directory_is_exported_into(dataset, tmp_path):
    out = tmp_path / 'out'
    out.mkdir()

    assert an_export(dataset, out)


def test_a_second_export_into_the_directory_of_a_running_one_is_refused(dataset, tmp_path):
    out = tmp_path / 'out'
    refusals: list[ValueError] = []

    def second():
        try:
            an_export(dataset, out)
        except ValueError as error:
            refusals.append(error)

    with positronic_server.app_state_restored():  # the first export, holding the state
        thread = threading.Thread(target=second)
        thread.start()
        thread.join(0.2)
        out.mkdir()
        (out / 'index.html').write_text('the first export')
    thread.join()

    assert [str(error).partition(';')[0] for error in refusals] == [f'{out} is not empty']
    assert [path.name for path in out.iterdir()] == ['index.html']
    assert (out / 'index.html').read_text() == 'the first export'


def test_a_download_inside_a_list_is_linked_by_its_index_and_served(dataset, tmp_path):
    out = tmp_path / 'out'
    written = paths_of(an_export(dataset, out))

    assert '"api/episode/0/static/artifacts/0"' in (out / 'episode/0/index.html').read_text()
    assert 'api/episode/0/static/artifacts/0' in written
    assert (out / 'api/episode/0/static/artifacts/0').read_bytes() == b'first blob'
    assert 'api/episode/0/static/artifacts/1' not in written


def test_a_filter_on_a_number_reaches_the_episodes_that_carry_it(dataset, tmp_path):
    out = tmp_path / 'out'
    an_export(dataset, out, group_tables=GROUPS)

    index = json.loads((out / 'api/groups/attempts' / GROUP_INDEX_FILE).read_text())
    second = next(entry for entry in index if entry['params'] == {ATTEMPT: '2'})
    table = json.loads((out / 'api/groups/attempts' / second['file']).read_text())
    outcome = [column['key'] for column in table['columns']].index(OUTCOME)
    assert [row[1][outcome] for row in table['episodes']] == ['Failure']


def test_an_export_leaves_the_app_state_as_it_found_it(dataset, tmp_path):
    before = dict(app_state)
    an_export(dataset, tmp_path / 'out', base_href='/v/tok/', title='A run')

    assert app_state == before


class _Reversed(Dataset):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def _get_episode(self, index: int) -> Episode:
        return cast(Episode, self._dataset[len(self._dataset) - 1 - index])


def test_a_full_dataset_holding_other_episodes_at_the_shown_indexes_is_refused(dataset, tmp_path):
    with pytest.raises(ValueError, match='uid'):
        an_export(dataset, tmp_path / 'reversed', full_dataset=_Reversed(dataset))
    with pytest.raises(ValueError, match='holds 1 episodes'):
        an_export(
            dataset, tmp_path / 'shorter', full_dataset=FilterDataset(dataset, lambda ep: ep.static[ATTEMPT] == 1)
        )
    assert not (tmp_path / 'reversed').exists() and not (tmp_path / 'shorter').exists()


ESCAPED_KEY = 'notes & é?'


def test_a_download_whose_field_name_a_url_cannot_carry_as_it_is_is_linked_encoded_and_written(tmp_path):
    dataset = a_dataset(tmp_path / 'dataset', {keys.TASK: 'Put the banana on the plate', ESCAPED_KEY: 'n' * 2000})
    out = tmp_path / 'out'

    written = paths_of(export_static(dataset, out, ep_table_cfg=TABLE, max_resolution=64, assets=False, build_id='bld'))

    encoded = 'notes%20%26%20%C3%A9%3F'
    assert f'"build/bld/api/episode/0/static/{encoded}"' in (out / 'episode/0/index.html').read_text()
    assert f'build/bld/api/episode/0/static/{encoded}' in written
    assert (out / 'build/bld/api/episode/0/static' / encoded).read_bytes() == b'n' * 2000


def test_a_group_name_that_is_not_one_url_path_segment_is_refused(dataset, tmp_path):
    for name in ('', 'a?b', 'a#b', 'a b', 'a/b', '.', '../..'):
        with pytest.raises(ValueError, match='group name'):
            an_export(dataset, tmp_path / 'out', group_tables={name: GROUPS['outcomes']})
    assert not (tmp_path / 'out').exists()


def test_a_home_page_that_names_no_group_table_is_refused(dataset, tmp_path):
    with pytest.raises(ValueError, match='home_page'):
        an_export(dataset, tmp_path / 'out', group_tables=GROUPS, home_page='nope')
    assert not (tmp_path / 'out').exists()


def test_the_page_script_spells_a_group_index_as_the_export_writes_it():
    app_js = (Path(positronic_server.__file__).parent / 'static' / 'app.js').read_text()
    params, file = (field.name for field in fields(GroupFile))

    assert f"const GROUP_INDEX_FILE = '{GROUP_INDEX_FILE}';" in app_js
    assert f"const ENTRY_PARAMS = '{params}';" in app_js and f"const ENTRY_FILE = '{file}';" in app_js


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
