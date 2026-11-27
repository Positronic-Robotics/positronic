import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pin_move  # noqa: E402

OLD_SHA = 'a' * 40
NEW_SHA = 'b' * 40


class Change:
    """One changed file of a pull request: its files-API entry and its text at the head commit."""

    def __init__(self, path, patch, head='', status=pin_move.MODIFIED):
        self.path, self.head = path, head
        self.entry = {pin_move.FILE_NAME: path, pin_move.FILE_STATUS: status, pin_move.FILE_PATCH: patch}


def is_pin_move(changes, *on_branch):
    heads = {c.path: c.head for c in changes}
    return pin_move.is_pin_move([c.entry for c in changes], lambda r: r in on_branch, heads.__getitem__)


def project(name, version, *more):
    return '\n'.join(['[project]', f'name = "{name}"', f'version = "{version}"', *more])


# The `[project]` header sits above the hunk, as in a real version bump.
CLIENT = Change(
    'client/pyproject.toml',
    '@@ -2,4 +2,4 @@\n name = "client"\n-version = "1.4.2"\n+version = "1.4.3"\n dependencies = [\n'
    '-    "evals==0.2.1",\n+    "evals==0.2.2",',
    project('client', '1.4.3', 'dependencies = [', '    "evals==0.2.2",', ']'),
)
REVISION = Change('evals/evals/evals.py', f"@@ -7 +7 @@\n-REVISION = '{OLD_SHA}'\n+REVISION = '{NEW_SHA}'")
EVALS = Change('evals/pyproject.toml', '@@ -3 +3 @@\n-version = "0.2.1"\n+version = "0.2.2"', project('evals', '0.2.2'))
ROOT = Change(
    'pyproject.toml',
    '@@ -5 +5 @@\n-    "client==1.4.2",\n+    "client==1.4.3",',
    project('root', '9.0', 'dependencies = [', '    "client==1.4.3",', ']'),
)
LOCK = Change(
    'uv.lock',
    '@@ -10,3 +10,3 @@\n [[package]]\n name = "evals"\n-version = "0.2.1"\n+version = "0.2.2"\n'
    '@@ -20,3 +20,3 @@\n [[package]]\n name = "client"\n-version = "1.4.2"\n+version = "1.4.3"',
)
A_PIN_MOVE = [CLIENT, REVISION, EVALS, ROOT, LOCK]


def code(old, new):
    return Change('evals/evals/evals.py', f'@@ -30 +30 @@\n-{old}\n+{new}')


def test_a_pin_move_to_a_commit_on_the_guarded_branch_passes():
    assert is_pin_move(A_PIN_MOVE, NEW_SHA)


def test_a_pin_move_to_a_commit_off_the_guarded_branch_is_refused():
    assert not is_pin_move(A_PIN_MOVE)


def test_a_changed_line_of_code_beside_a_pin_move_is_refused():
    assert not is_pin_move([*A_PIN_MOVE, code('    return run(task)', '    return run(task, retries=3)')], NEW_SHA)


def test_a_quoted_number_outside_a_version_field_is_not_a_pin():
    assert not is_pin_move([*A_PIN_MOVE, code("THRESHOLD = '0.5'", "THRESHOLD = '0.7'")], NEW_SHA)


def test_an_exact_pin_to_the_version_another_package_moves_to_is_refused():
    """A pin of a third-party package that matches this repository's new version brings code no human reviewed."""
    head = project('root', '2.0', 'dependencies = [', '    "torch==2.0",', ']')
    files = Change(
        'pyproject.toml',
        '@@ -3,3 +3,3 @@\n-version = "1.0"\n+version = "2.0"\n dependencies = [\n'
        '-    "torch==1.0",\n+    "torch==2.0",',
        head,
    )
    assert not is_pin_move([files])


def test_a_name_and_version_outside_the_project_table_name_no_package():
    """A `[tool.*]` table that names a third-party package cannot vouch for a pin of it."""
    head = '\n'.join([
        project('root', '9.0', 'dependencies = [', '    "torch==2.0",', ']'),
        '[tool.release]',
        'name = "torch"',
        'version = "2.0"',
    ])
    pin = '@@ -5 +5 @@\n-    "torch==1.0",\n+    "torch==2.0",'
    release = '@@ -8,2 +8,2 @@\n name = "torch"\n-version = "1.0"\n+version = "2.0"'
    assert not is_pin_move([Change('pyproject.toml', f'{pin}\n{release}', head)])


def test_an_exact_pin_outside_a_dependency_table_is_refused():
    head = '\n'.join([project('root', '9.0'), '[tool.extra]', 'pins = [', '    "evals==0.2.2",', ']'])
    other = Change('pyproject.toml', '@@ -6 +6 @@\n-    "evals==0.2.1",\n+    "evals==0.2.2",', head)
    assert not is_pin_move([CLIENT, REVISION, EVALS, LOCK, other], NEW_SHA)


def test_an_exact_pin_outside_a_project_file_is_refused():
    assert not is_pin_move([*A_PIN_MOVE, code('    "evals==0.2.1",', '    "evals==0.2.2",')], NEW_SHA)


def test_a_project_file_that_names_no_package_is_refused():
    nameless = Change(
        'evals/pyproject.toml', '@@ -2 +2 @@\n-version = "0.2.1"\n+version = "0.2.2"', '[project]\nversion = "0.2.2"'
    )
    assert not is_pin_move([nameless, ROOT, CLIENT])


def test_a_lock_version_the_pull_request_does_not_set_is_refused():
    lock = Change('uv.lock', '@@ -40,3 +40,3 @@\n [[package]]\n name = "torch"\n-version = "2.5.0"\n+version = "2.6.0"')
    assert not is_pin_move([*A_PIN_MOVE, lock], NEW_SHA)


def test_a_lock_version_away_from_its_package_header_is_refused():
    lock = Change(
        'uv.lock', '@@ -40,3 +40,3 @@\n [package.metadata]\n name = "evals"\n-version = "0.2.1"\n+version = "0.2.2"'
    )
    assert not is_pin_move([CLIENT, REVISION, EVALS, ROOT, lock], NEW_SHA)


def test_an_added_line_or_an_added_file_is_refused():
    one_sided = Change('evals/evals/evals.py', '@@ -1 +1,2 @@\n x = 1\n+y = 2')
    assert not is_pin_move([*A_PIN_MOVE, one_sided], NEW_SHA)
    added = Change('evals/new.py', '@@ -0,0 +1 @@\n+import os', status='added')
    assert not is_pin_move([*A_PIN_MOVE, added], NEW_SHA)


def test_a_file_without_a_patch_is_refused():
    binary = Change('assets/logo.png', '')
    del binary.entry[pin_move.FILE_PATCH]
    assert not is_pin_move([*A_PIN_MOVE, binary], NEW_SHA)


def test_version_fields_alone_move_no_pin():
    assert not is_pin_move([EVALS])


def test_a_line_that_already_holds_a_placeholder_is_not_a_pin_move():
    """A revision moved into the place of a literal placeholder changes which name holds the revision."""
    swap = code(f"SAFE = '<revision>'; REVISION = '{OLD_SHA}'", f"SAFE = '{NEW_SHA}'; REVISION = '<revision>'")
    assert not is_pin_move([swap], NEW_SHA)
