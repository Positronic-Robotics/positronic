import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_basedpyright_baseline_ratchet as ratchet  # noqa: E402


def _git(repo: Path, *args: str) -> None:
    subprocess.run(['git', *args], cwd=repo, check=True, capture_output=True, text=True)


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
    assert ratchet.grown_files(base, current) == [('a.py', 'reportArgumentType', 0, 1)]


def test_genuine_fix_fewer_entries_passes():
    base = _baseline({'./a.py': ['reportReturnType', 'reportReturnType']})
    current = _baseline({'./a.py': ['reportReturnType']})
    assert ratchet.grown_files(base, current) == []


def test_new_file_with_entries_fails():
    base = _baseline({'./a.py': ['reportReturnType']})
    current = _baseline({'./a.py': ['reportReturnType'], './new.py': ['reportUnknownMemberType']})
    assert ratchet.grown_files(base, current) == [('new.py', 'reportUnknownMemberType', 0, 1)]


def test_more_of_existing_code_fails():
    base = _baseline({'./a.py': ['reportReturnType']})
    current = _baseline({'./a.py': ['reportReturnType', 'reportReturnType']})
    assert ratchet.grown_files(base, current) == [('a.py', 'reportReturnType', 1, 2)]


def test_pure_rename_with_identical_entries_passes():
    base = _baseline({'./positronic/old.py': ['reportReturnType', 'reportArgumentType']})
    current = _baseline({'./positronic/new.py': ['reportReturnType', 'reportArgumentType']})
    rename_map = {'positronic/new.py': 'positronic/old.py'}  # git-style, no leading ./
    assert ratchet.grown_files(base, current, rename_map) == []


def test_rename_that_also_grew_fails_on_new_path():
    base = _baseline({'./positronic/old.py': ['reportReturnType']})
    current = _baseline({'./positronic/new.py': ['reportReturnType', 'reportArgumentType']})  # renamed AND grew
    rename_map = {'positronic/new.py': 'positronic/old.py'}
    assert ratchet.grown_files(base, current, rename_map) == [('positronic/new.py', 'reportArgumentType', 0, 1)]


def test_rename_without_map_flags_every_entry_as_new():
    # A rename NOT reflected in the map (empty map) reads the moved file as brand-new — the exact
    # false-positive build_rename_map prevents; asserts the map is what suppresses it.
    base = _baseline({'./positronic/old.py': ['reportReturnType']})
    current = _baseline({'./positronic/new.py': ['reportReturnType']})
    assert ratchet.grown_files(base, current) == [('positronic/new.py', 'reportReturnType', 0, 1)]


def test_build_rename_map_includes_staged_rename(tmp_path, monkeypatch):
    # Under pre-commit the rename is staged (in the index), not yet in HEAD. build_rename_map must
    # still map it, or a moved baselined file reads as brand-new and the local commit is wrongly
    # blocked while CI (where HEAD holds the rename) passes.
    repo = tmp_path
    _git(repo, 'init', '-q')
    _git(repo, 'config', 'user.email', 't@t')
    _git(repo, 'config', 'user.name', 't')
    (repo / 'old.py').write_text('x = 1\n' * 20)
    _git(repo, 'add', 'old.py')
    _git(repo, 'commit', '-qm', 'base')
    base = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo, capture_output=True, text=True).stdout.strip()
    _git(repo, 'mv', 'old.py', 'new.py')  # staged, NOT committed
    monkeypatch.chdir(repo)
    assert ratchet.build_rename_map(base) == {'new.py': 'old.py'}


def test_parse_renames_maps_renames_not_copies():
    # A rename (R) moves entries to a new key and must map back; a copy (C) leaves the source in
    # place and duplicates its diagnostics into a genuinely new file, which must face the new-file
    # check rather than reuse the source's grandfathered allowances — so C must NOT be mapped.
    out = 'R100\told.py\tmoved.py\nC100\tkept.py\tcopy.py\n'
    assert ratchet._parse_renames(out) == {'moved.py': 'old.py'}


def test_base_ref_arg_wins_then_env_then_default():
    assert ratchet.resolve_base_ref('abc123', 'def456') == 'abc123'  # --base wins over env
    assert ratchet.resolve_base_ref(None, 'def456') == 'def456'  # RATCHET_BASE env used
    assert ratchet.resolve_base_ref(None, None) == 'origin/main'  # local pre-commit default
    assert ratchet.resolve_base_ref(None, '') == 'origin/main'  # empty env falls through
