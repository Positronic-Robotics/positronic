import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_basedpyright_baseline_ratchet as ratchet  # noqa: E402


def _entry(line: int) -> dict[str, object]:
    return {'code': 'reportReturnType', 'range': {'startColumn': 1, 'endColumn': 2, 'lineCount': 1, 'line': line}}


def _baseline(counts: dict[str, int]) -> dict[str, object]:
    return {'files': {path: [_entry(i) for i in range(n)] for path, n in counts.items()}}


def test_passes_when_every_file_at_or_below_base():
    base = _baseline({'./a.py': 3, './b.py': 2})
    current = _baseline({'./a.py': 3, './b.py': 1})  # b shrank, a unchanged
    assert ratchet.grown_files(base, current) == []


def test_fails_when_a_file_grew():
    base = _baseline({'./a.py': 2})
    current = _baseline({'./a.py': 5})
    assert ratchet.grown_files(base, current) == [('./a.py', 2, 5)]


def test_fails_when_new_file_appears_with_entries():
    base = _baseline({'./a.py': 2})
    current = _baseline({'./a.py': 2, './new.py': 1})
    assert ratchet.grown_files(base, current) == [('./new.py', 0, 1)]


def test_passes_on_reanchor_same_count_different_lines():
    base = {'files': {'./a.py': [_entry(10), _entry(20)]}}
    current = {'files': {'./a.py': [_entry(99), _entry(120)]}}  # same count, shifted lines
    assert ratchet.grown_files(base, current) == []


def test_base_ref_arg_wins_then_env_then_default():
    assert ratchet.resolve_base_ref('abc123', 'def456') == 'abc123'  # --base wins over env
    assert ratchet.resolve_base_ref(None, 'def456') == 'def456'  # RATCHET_BASE env used
    assert ratchet.resolve_base_ref(None, None) == 'origin/main'  # local pre-commit default
    assert ratchet.resolve_base_ref(None, '') == 'origin/main'  # empty env falls through
