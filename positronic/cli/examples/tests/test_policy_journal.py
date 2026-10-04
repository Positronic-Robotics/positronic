"""`policy_journal/record.py` and `policy_journal/replay.py`: one recorded episode, verified and branched offline."""

import runpy
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / 'policy_journal'


def test_the_replay_example_verifies_and_branches_what_the_record_example_journals(tmp_path, capsys):
    runpy.run_path(str(EXAMPLE / 'record.py'))['main'](tmp_path / 'rollout')
    recorded = capsys.readouterr().out
    assert 'the inference ran 2 times, at positions [0, 150]' in recorded
    assert "episode: {'eval.terminated': False}" in recorded

    runpy.run_path(str(EXAMPLE / 'replay.py'))['main'](tmp_path / 'rollout', tmp_path / 'branches')
    replayed = capsys.readouterr().out.splitlines()
    assert 'verified 5 turns without running the inference, complete: True' in replayed
    assert 'refused: Rerunning submission 0 needs allow_execution=True' in replayed
    assert not (tmp_path / 'branches' / 'refused').exists()
    replaced = replayed.index(f'replaced result: {tmp_path / "branches" / "replaced"}')
    assert replayed[replaced + 1 : replaced + 3] == [
        "  turn 1 at 4 ms: commands [{'motor': 1}] -> [{'motor': 0}]",
        "  turn 2 at 104 ms: commands [{'motor': 2}] -> [{'motor': 0}]",
    ]
    rerun = replayed.index(f'rerun inference: {tmp_path / "branches" / "rerun"}')
    assert replayed[rerun + 1 : rerun + 3] == [
        "  turn 1 at 4 ms: commands [{'motor': 1}] -> [{'motor': -1}]",
        "  turn 2 at 104 ms: commands [{'motor': 2}] -> [{'motor': -2}]",
    ]
    assert replayed[-2:] == [
        'changed request: stopped, because the source records no step_plan v2',
        'source unchanged: True',
    ]
