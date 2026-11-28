from pathlib import Path

import pytest

from positronic.dagger import META_RECORD_UID, META_ROUND, write_training_data
from positronic.dataset.episode import META_UID
from positronic.dataset.local_dataset import DiskEpisodeWriter, LocalDataset
from positronic.policy.dagger import DaggerDataset
from positronic.policy.tests.test_dagger import (
    ARM,
    GRIP,
    OBSERVATIONS,
    POLICY,
    RULE,
    SHADOW_ARM,
    STATE,
    _policy,
    _record,
    _steps,
    _teleoperator,
    _values,
)

LABELS = [ARM, GRIP]


def _rounds(tmp_path, count: int) -> list[DaggerDataset]:
    return [
        DaggerDataset(LocalDataset(tmp_path / f'round{index}'), OBSERVATIONS, LABELS, RULE) for index in range(count)
    ]


def test_the_training_data_of_round_2_holds_the_teleoperator_steps_of_each_round_with_their_labels_and_shadow_actions(
    tmp_path,
):
    # Round 0: the teleoperator takes over at the first step. In later rounds, it takes over when the policy fails.
    rounds = [
        [_policy(), _teleoperator(), _teleoperator()],
        [_policy(), _teleoperator(), _teleoperator(shadow=()), _policy()],
        [_policy(), _policy(), _teleoperator()],
    ]
    uids = [_record(tmp_path / f'round{index}', steps).meta[META_UID] for index, steps in enumerate(rounds)]
    data = _rounds(tmp_path, 3)

    write_training_data(data, tmp_path / 'training')

    written = LocalDataset(tmp_path / 'training')
    assert [(episode.meta[META_ROUND], episode.meta[META_RECORD_UID]) for episode in written] == list(enumerate(uids))
    assert [_steps(episode[STATE]) for episode in written] == [[1, 2], [1, 2], [2]]
    assert [_steps(episode[ARM]) for episode in written] == [[1, 2], [1, 2], [2]]
    assert [_values(episode[ARM]) for episode in written] == [[11.0, 12.0], [11.0, 12.0], [12.0]]
    assert [list(episode[GRIP].values()) for episode in written] == [[1.0, 2.0], [1.0, 2.0], [2.0]]
    assert [_values(episode[SHADOW_ARM]) for episode in written] == [[21.0, 22.0], [21.0], [22.0]]
    assert [_steps(episode[SHADOW_ARM]) for episode in written] == [[1, 2], [1], [2]]
    assert [len(round_data) for round_data in data] == [1, 1, 1]
    assert [round_data.steps_without_shadow for round_data in data] == [{POLICY: 0}, {POLICY: 1}, {POLICY: 0}]


def test_a_record_that_fails_the_read_writes_no_training_data(tmp_path):
    _record(tmp_path / 'round0', [_teleoperator()] * 2)
    _record(tmp_path / 'round1', [_teleoperator()] * 2, grip_missing_at=1)

    with pytest.raises(ValueError, match='has no sample at the step'):
        write_training_data(_rounds(tmp_path, 2), tmp_path / 'training')
    assert sorted(path.name for path in tmp_path.iterdir()) == ['round0', 'round1']


def test_a_write_that_stops_in_a_later_round_writes_no_training_data(tmp_path, monkeypatch):
    _record(tmp_path / 'round0', [_teleoperator()] * 2)
    _record(tmp_path / 'round1', [_teleoperator()] * 2)
    append = DiskEpisodeWriter.append

    def disk_full_in_the_second_episode(writer, *args):
        if writer.path.name == f'{1:012d}':
            raise OSError('disk full')
        append(writer, *args)

    monkeypatch.setattr(DiskEpisodeWriter, 'append', disk_full_in_the_second_episode)
    with pytest.raises(OSError, match='disk full'):
        write_training_data(_rounds(tmp_path, 2), tmp_path / 'training')
    assert sorted(path.name for path in tmp_path.iterdir()) == ['round0', 'round1']


def test_the_training_data_folder_under_home_is_made_as_the_folder_of_the_records(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    _record(tmp_path / 'round0', [_teleoperator()] * 2)

    write_training_data(_rounds(tmp_path, 1), Path('~/training'))
    assert len(LocalDataset(tmp_path / 'training')) == 1
    assert (tmp_path / 'training').stat().st_mode == (tmp_path / 'round0').stat().st_mode


@pytest.mark.parametrize('episodes', [0, 1])
def test_the_training_data_does_not_go_to_a_folder_that_exists(tmp_path, episodes):
    _record(tmp_path / 'round0', [_teleoperator()] * 2)
    (tmp_path / 'training').mkdir()
    for _ in range(episodes):
        _record(tmp_path / 'training', [_teleoperator()] * 2)

    with pytest.raises(ValueError, match='exists'):
        write_training_data(_rounds(tmp_path, 1), tmp_path / 'training')
    assert len(LocalDataset(tmp_path / 'training')) == episodes
