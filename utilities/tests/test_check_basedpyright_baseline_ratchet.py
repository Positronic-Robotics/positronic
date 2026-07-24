import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_basedpyright_baseline_ratchet as ratchet  # noqa: E402


def _entry(code: str, col: int = 1) -> dict[str, object]:
    return {'code': code, 'range': {'startColumn': col, 'endColumn': col + 1, 'lineCount': 1}}


def _baseline(files: dict[str, list[str]]) -> dict[str, object]:
    """Build a baseline from {path: [code, code, ...]}; columns are auto-assigned distinctly."""
    return {'files': {path: [_entry(c, i + 1) for i, c in enumerate(codes)] for path, codes in files.items()}}


def test_reanchor_same_codes_changed_columns_passes():
    base = {'files': {'./a.py': [_entry('reportReturnType', 1), _entry('reportArgumentType', 5)]}}
    current = {'files': {'./a.py': [_entry('reportReturnType', 40), _entry('reportArgumentType', 88)]}}
    assert ratchet.grown_files(base, current) == []


def test_cross_code_swap_same_total_fails():
    base = _baseline({'./a.py': ['reportReturnType', 'reportReturnType']})
    current = _baseline({'./a.py': ['reportReturnType', 'reportArgumentType']})  # dropped one, added a new code
    assert ratchet.grown_files(base, current) == [('./a.py', 'reportArgumentType', 0, 1)]


def test_genuine_fix_fewer_entries_passes():
    base = _baseline({'./a.py': ['reportReturnType', 'reportReturnType']})
    current = _baseline({'./a.py': ['reportReturnType']})
    assert ratchet.grown_files(base, current) == []


def test_new_file_with_entries_fails():
    base = _baseline({'./a.py': ['reportReturnType']})
    current = _baseline({'./a.py': ['reportReturnType'], './new.py': ['reportUnknownMemberType']})
    assert ratchet.grown_files(base, current) == [('./new.py', 'reportUnknownMemberType', 0, 1)]


def test_more_of_existing_code_fails():
    base = _baseline({'./a.py': ['reportReturnType']})
    current = _baseline({'./a.py': ['reportReturnType', 'reportReturnType']})
    assert ratchet.grown_files(base, current) == [('./a.py', 'reportReturnType', 1, 2)]


def test_base_ref_arg_wins_then_env_then_default():
    assert ratchet.resolve_base_ref('abc123', 'def456') == 'abc123'  # --base wins over env
    assert ratchet.resolve_base_ref(None, 'def456') == 'def456'  # RATCHET_BASE env used
    assert ratchet.resolve_base_ref(None, None) == 'origin/main'  # local pre-commit default
    assert ratchet.resolve_base_ref(None, '') == 'origin/main'  # empty env falls through
