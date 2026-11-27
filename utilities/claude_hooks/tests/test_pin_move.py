import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pin_move  # noqa: E402

OLD_SHA = 'a' * 40
NEW_SHA = 'b' * 40


def changed(name, *pairs, status=pin_move.MODIFIED):
    """A files-API entry whose patch replaces each `old` line with its `new` line, one block each."""
    patch = '\n'.join(f'@@ -1 +1 @@\n context\n-{old}\n+{new}\n context' for old, new in pairs)
    return {pin_move.FILE_NAME: name, pin_move.FILE_STATUS: status, pin_move.FILE_PATCH: patch}


def a_pin_move():
    return [
        changed(
            'client/pyproject.toml',
            ('version = "1.4.2"', 'version = "1.4.3"'),
            ('    "evals==0.2.1",', '    "evals==0.2.2",'),
        ),
        changed('evals/evals/evals.py', (f"REVISION = '{OLD_SHA}'", f"REVISION = '{NEW_SHA}'")),
        changed('evals/pyproject.toml', ('version = "0.2.1"', 'version = "0.2.2"')),
        changed('pyproject.toml', ('    "client==1.4.2",', '    "client==1.4.3",')),
        changed('uv.lock', ('version = "0.2.1"', 'version = "0.2.2"'), ('version = "1.4.2"', 'version = "1.4.3"')),
    ]


def on_branch(*revisions):
    return lambda revision: revision in revisions


def test_a_pin_move_to_a_commit_on_the_guarded_branch_passes():
    assert pin_move.is_pin_move(a_pin_move(), on_branch(NEW_SHA))


def test_a_pin_move_to_a_commit_off_the_guarded_branch_is_refused():
    assert not pin_move.is_pin_move(a_pin_move(), on_branch())


def test_a_changed_line_of_code_beside_a_pin_move_is_refused():
    files = [
        *a_pin_move(),
        changed('evals/evals/evals.py', ('    return run(task)', '    return run(task, retries=3)')),
    ]
    assert not pin_move.is_pin_move(files, on_branch(NEW_SHA))


def test_a_quoted_number_outside_a_version_field_is_not_a_pin():
    files = [*a_pin_move(), changed('evals/evals/evals.py', ("THRESHOLD = '0.5'", "THRESHOLD = '0.7'"))]
    assert not pin_move.is_pin_move(files, on_branch(NEW_SHA))


def test_an_exact_pin_to_a_version_the_pull_request_does_not_set_is_refused():
    """A third-party version bump brings code that no human of this repository reviewed."""
    files = [changed('pyproject.toml', ('    "torch==2.5.0",', '    "torch==2.6.0",'))]
    assert not pin_move.is_pin_move(files, on_branch())


def test_a_lock_version_the_pull_request_does_not_set_is_refused():
    files = [*a_pin_move(), changed('uv.lock', ('version = "2.5.0"', 'version = "2.6.0"'))]
    assert not pin_move.is_pin_move(files, on_branch(NEW_SHA))


def test_an_added_line_or_an_added_file_is_refused():
    one_sided = {**changed('evals/pyproject.toml'), pin_move.FILE_PATCH: '@@ -1 +1,2 @@\n x = 1\n+y = 2'}
    assert not pin_move.is_pin_move([*a_pin_move(), one_sided], on_branch(NEW_SHA))
    added = changed('evals/new.py', ('', 'import os'), status='added')
    assert not pin_move.is_pin_move([*a_pin_move(), added], on_branch(NEW_SHA))


def test_a_file_without_a_patch_is_refused():
    binary = {pin_move.FILE_NAME: 'assets/logo.png', pin_move.FILE_STATUS: pin_move.MODIFIED}
    assert not pin_move.is_pin_move([*a_pin_move(), binary], on_branch(NEW_SHA))


def test_version_fields_alone_move_no_pin():
    files = [changed('evals/pyproject.toml', ('version = "0.2.1"', 'version = "0.2.2"'))]
    assert not pin_move.is_pin_move(files, on_branch())
