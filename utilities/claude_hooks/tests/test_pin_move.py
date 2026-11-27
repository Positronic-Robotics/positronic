import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pin_move  # noqa: E402

OLD_SHA = 'a' * 40
NEW_SHA = 'b' * 40


def changed(name, *blocks, status=pin_move.MODIFIED):
    """A files-API entry. Each block is (context above, [(old, new), ...]): one hunk that replaces each old line."""
    hunks = []
    for above, pairs in blocks:
        lines = [f' {line}' for line in above]
        lines += [f'-{old}' for old, _ in pairs] + [f'+{new}' for _, new in pairs]
        hunks.append('\n'.join(['@@ -1 +1 @@', *lines, ' description = "x"']))
    return {pin_move.FILE_NAME: name, pin_move.FILE_STATUS: status, pin_move.FILE_PATCH: '\n'.join(hunks)}


def version_bump(path, package, old, new):
    return changed(path, (['[project]', f'name = "{package}"'], [(f'version = "{old}"', f'version = "{new}"')]))


def pin(package, old, new):
    return (f'    "{package}=={old}",', f'    "{package}=={new}",')


def a_pin_move():
    return [
        version_bump('client/pyproject.toml', 'client', '1.4.2', '1.4.3'),
        changed('client/pyproject.toml', (['dependencies = ['], [pin('evals', '0.2.1', '0.2.2')])),
        changed('evals/evals/evals.py', ([], [(f"REVISION = '{OLD_SHA}'", f"REVISION = '{NEW_SHA}'")])),
        version_bump('evals/pyproject.toml', 'evals', '0.2.1', '0.2.2'),
        changed('pyproject.toml', (['dependencies = ['], [pin('client', '1.4.2', '1.4.3')])),
        changed(
            'uv.lock',
            (['[[package]]', 'name = "evals"'], [('version = "0.2.1"', 'version = "0.2.2"')]),
            (['[[package]]', 'name = "client"'], [('version = "1.4.2"', 'version = "1.4.3"')]),
        ),
    ]


def on_branch(*revisions):
    return lambda revision: revision in revisions


def test_a_pin_move_to_a_commit_on_the_guarded_branch_passes():
    assert pin_move.is_pin_move(a_pin_move(), on_branch(NEW_SHA))


def test_a_pin_move_to_a_commit_off_the_guarded_branch_is_refused():
    assert not pin_move.is_pin_move(a_pin_move(), on_branch())


def test_a_changed_line_of_code_beside_a_pin_move_is_refused():
    code = changed('evals/evals/evals.py', ([], [('    return run(task)', '    return run(task, retries=3)')]))
    assert not pin_move.is_pin_move([*a_pin_move(), code], on_branch(NEW_SHA))


def test_a_quoted_number_outside_a_version_field_is_not_a_pin():
    code = changed('evals/evals/evals.py', ([], [("THRESHOLD = '0.5'", "THRESHOLD = '0.7'")]))
    assert not pin_move.is_pin_move([*a_pin_move(), code], on_branch(NEW_SHA))


def test_an_exact_pin_to_the_version_another_package_moves_to_is_refused():
    """A pin of a third-party package that matches this repository's new version brings code no human reviewed."""
    files = [
        version_bump('pyproject.toml', 'evals', '1.0', '2.0'),
        changed('pyproject.toml', (['dependencies = ['], [pin('torch', '1.0', '2.0')])),
    ]
    assert not pin_move.is_pin_move(files, on_branch())


def test_an_exact_pin_outside_a_project_file_is_refused():
    code = changed('evals/evals/evals.py', ([], [pin('evals', '0.2.1', '0.2.2')]))
    assert not pin_move.is_pin_move([*a_pin_move(), code], on_branch(NEW_SHA))


def test_a_version_field_with_no_package_name_above_it_is_refused():
    nameless = changed('evals/pyproject.toml', (['[project]'], [('version = "0.2.1"', 'version = "0.2.2"')]))
    files = [nameless, changed('pyproject.toml', (['dependencies = ['], [pin('evals', '0.2.1', '0.2.2')]))]
    assert not pin_move.is_pin_move(files, on_branch())


def test_a_lock_version_the_pull_request_does_not_set_is_refused():
    lock = changed('uv.lock', (['[[package]]', 'name = "torch"'], [('version = "2.5.0"', 'version = "2.6.0"')]))
    assert not pin_move.is_pin_move([*a_pin_move(), lock], on_branch(NEW_SHA))


def test_an_added_line_or_an_added_file_is_refused():
    one_sided = {**changed('evals/pyproject.toml'), pin_move.FILE_PATCH: '@@ -1 +1,2 @@\n x = 1\n+y = 2'}
    assert not pin_move.is_pin_move([*a_pin_move(), one_sided], on_branch(NEW_SHA))
    added = changed('evals/new.py', ([], [('', 'import os')]), status='added')
    assert not pin_move.is_pin_move([*a_pin_move(), added], on_branch(NEW_SHA))


def test_a_file_without_a_patch_is_refused():
    binary = {pin_move.FILE_NAME: 'assets/logo.png', pin_move.FILE_STATUS: pin_move.MODIFIED}
    assert not pin_move.is_pin_move([*a_pin_move(), binary], on_branch(NEW_SHA))


def test_version_fields_alone_move_no_pin():
    assert not pin_move.is_pin_move([version_bump('evals/pyproject.toml', 'evals', '0.2.1', '0.2.2')], on_branch())
