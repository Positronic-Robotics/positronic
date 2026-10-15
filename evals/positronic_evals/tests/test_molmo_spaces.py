"""MolmoSpaces' committed episode lists."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest
from positronic_evals.molmo_spaces import BROKEN_EPISODES_PATH, RIGHT_VIEW_BLOCKED_PATH, read_episode_list


def test_a_list_reads_as_the_episodes_of_each_benchmark_path(tmp_path: Path):
    path = tmp_path / 'list.json'
    path.write_text(json.dumps({'suite/scenes/task/bench': [3, 1, 3], 'suite/scenes/task/other': []}))

    assert read_episode_list(path) == {
        PurePosixPath('suite/scenes/task/bench'): frozenset({1, 3}),
        PurePosixPath('suite/scenes/task/other'): frozenset(),
    }


@pytest.mark.parametrize('path', [RIGHT_VIEW_BLOCKED_PATH, BROKEN_EPISODES_PATH])
def test_each_list_ships_with_the_package(path):
    assert path.is_file()
    read_episode_list(path)
