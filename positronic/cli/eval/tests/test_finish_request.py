import json
import logging
import os
import threading
from pathlib import Path

import pytest

from positronic.cli.eval import finish_request


def write_request(path: Path, *, action: str = finish_request.Action.FINISH, run: str, **extra) -> None:
    """An ordinary fixture, built from the constants the module defines, so a rename moves with them."""
    path.write_text(json.dumps({finish_request.ACTION_KEY: action, finish_request.RUN_ID_KEY: run, **extra}))


def test_a_request_naming_this_run_is_granted(tmp_path):
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')
    assert finish_request.evaluate(path, 'batch-1')[0]


def test_an_absent_file_is_not_a_request(tmp_path):
    assert not finish_request.evaluate(tmp_path / 'nothing', 'batch-1')[0]


def test_a_request_naming_another_run_is_ignored(tmp_path):
    """The stale-request case: a previous run's file outliving it must not end the next run, which
    would otherwise stop early and look exactly like a short but successful round."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')
    assert not finish_request.evaluate(path, 'batch-2')[0]


def test_an_unparseable_request_leaves_the_run_going(tmp_path):
    path = tmp_path / 'finish'
    path.write_text('finish please')
    assert not finish_request.evaluate(path, 'batch-1')[0]


def test_a_json_scalar_is_not_a_request(tmp_path):
    """`json.loads` accepts a bare string, so parsing is not enough — the shape is checked too."""
    path = tmp_path / 'finish'
    path.write_text('"finish"')
    assert not finish_request.evaluate(path, 'batch-1')[0]


def test_an_unreadable_request_leaves_the_run_going(tmp_path):
    """The umask trap: a writer whose umask is 077 leaves a file this account cannot open. It must
    read as "keep running", never as a request, and it must not raise out of the poll loop."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')
    path.chmod(0o000)
    try:
        if os.geteuid() == 0:
            pytest.skip('root reads a mode-000 file, so the unreadable branch cannot be reached')
        assert not finish_request.evaluate(path, 'batch-1')[0]
    finally:
        path.chmod(0o644)


def test_an_unknown_action_is_ignored(tmp_path):
    """A second action would be a new value of this field, so an unrecognised one is a writer from a
    future this run does not implement — and acting on it would be guessing what it asked for."""
    path = tmp_path / 'finish'
    write_request(path, action='abort', run='batch-1')
    assert not finish_request.evaluate(path, 'batch-1')[0]


def test_extra_fields_do_not_break_a_request(tmp_path):
    """The writer records its own diagnostics in the same object; they are not this side's business."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1', requested_at_s=1.0, requested_by='someone')
    assert finish_request.evaluate(path, 'batch-1')[0]


def test_nothing_is_installed_without_a_run_id(monkeypatch):
    monkeypatch.delenv(finish_request.RUN_ID_ENV, raising=False)
    assert finish_request.from_env() is None


def test_an_empty_run_id_installs_nothing(monkeypatch):
    """An exported-but-empty variable is what a launcher produces from an unset one, and treating it
    as an identity would make every such run answer a request addressed to ''."""
    monkeypatch.setenv(finish_request.RUN_ID_ENV, '')
    assert finish_request.from_env() is None


def test_the_run_id_and_directory_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(finish_request.RUN_ID_ENV, 'batch-7')
    monkeypatch.setenv(finish_request.FINISH_REQUEST_DIR_ENV, str(tmp_path))
    cs = finish_request.from_env()
    assert cs is not None
    assert cs._run == 'batch-7'
    assert cs._path == tmp_path / f'{finish_request.FINISH_REQUEST_PREFIX}batch-7'


def test_the_path_names_the_run_so_two_runs_never_share_one(monkeypatch, tmp_path):
    """The property the whole mechanism rests on. One rig-wide path is a cell every run writes and
    only one account may replace, so a file left by an ended run is the next run's finish refused.
    Named per run, a leftover addresses a reader that will never exist again."""
    monkeypatch.setenv(finish_request.FINISH_REQUEST_DIR_ENV, str(tmp_path))
    assert finish_request.request_path('batch-1') != finish_request.request_path('batch-2')
    write_request(finish_request.request_path('batch-1'), run='batch-1')
    assert finish_request.evaluate(finish_request.request_path('batch-1'), 'batch-1')[0]
    assert not finish_request.evaluate(finish_request.request_path('batch-2'), 'batch-2')[0]


def test_the_default_path_is_absolute_and_on_tmpfs(monkeypatch):
    """Both properties are load-bearing and neither is visible at a call site: an account-relative
    path would be a different file for the writer and the run, and a persistent one would let a
    request outlive the reboot that was supposed to clear it."""
    monkeypatch.delenv(finish_request.FINISH_REQUEST_DIR_ENV, raising=False)
    path = finish_request.request_path('batch-1')
    assert path.is_absolute()
    assert str(path).startswith('/run/')


@pytest.mark.parametrize('override', ['requests', './requests', '../requests'])
def test_a_relative_request_directory_installs_nothing(monkeypatch, tmp_path, override):
    """A relative override resolves against each account's own working directory, so the writer and
    the run address different files from the same configuration."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(finish_request.RUN_ID_ENV, 'batch-1')
    monkeypatch.setenv(finish_request.FINISH_REQUEST_DIR_ENV, override)
    assert finish_request.from_env() is None


def test_an_absolute_request_directory_installs_the_poller(monkeypatch, tmp_path):
    """An absolute override installs the poller: what is refused is a relative path, not an override."""
    monkeypatch.setenv(finish_request.RUN_ID_ENV, 'batch-1')
    monkeypatch.setenv(finish_request.FINISH_REQUEST_DIR_ENV, str(tmp_path))
    assert finish_request.from_env() is not None


@pytest.mark.parametrize('run', ['a/b', '../elsewhere', '.', '..', ''])
def test_a_run_id_that_is_not_a_filename_installs_nothing(monkeypatch, run):
    """The path is built from the run id, so an id carrying a separator would poll a file under a
    directory nobody agreed on — or, for `..`, outside the request directory entirely. Such a run
    keeps running with no poller, which is this module's failure direction everywhere else."""
    monkeypatch.setenv(finish_request.RUN_ID_ENV, run)
    assert finish_request.from_env() is None


def test_the_object_the_writer_sends_is_granted(tmp_path):
    """The cross-repo contract, spelled as literals rather than through this module's own constants.

    The writer is the rollouts MCP in `Positronic-Robotics/platform`, which cannot import this module
    and which this module cannot import. So the only thing holding the two spellings together is a
    test on each side that names them: this one, and `rollouts/mcp/tests/test_ops.py`'s
    `test_a_written_request_is_the_object_the_run_requires`. Renaming a key here without renaming it
    there produces a run that ignores every request it is sent, silently, and the first sign of it is
    a rollout nobody can stop.
    """
    path = tmp_path / 'positronic_rollout_finish.batch_20260807-111935'
    # rules-allow: hardcoded-keys — the literals ARE the pin. Built from this module's constants it
    # would assert only that the module agrees with itself; spelled out, a rename on either side of
    # the contract fails one of the two tests that name it.
    path.write_text(
        '{"action": "finish", "run_id": "batch_20260807-111935", '
        '"requested_at_s": 1786101583.6, "requested_by": "rollouts-mcp"}'
    )
    assert finish_request.evaluate(path, 'batch_20260807-111935')[0]
    assert not finish_request.evaluate(path, 'batch_20260807-110748')[0]


def test_a_refusal_is_reported_once_and_again_only_when_it_changes(tmp_path, caplog):
    """A request addressed to a dead run stays on the rig for the life of this one, and the poll runs
    every couple of seconds — so a reason logged per poll fills the log of exactly the long run
    somebody will later need to read."""
    path = tmp_path / 'finish'
    write_request(path, run='someone-else')
    cs = finish_request.FinishRequest(path, 'batch-1', poll_interval_s=0.0)

    def polls(n: int) -> int:
        with caplog.at_level('ERROR'):
            caplog.clear()
            for _ in range(n):
                granted, reason = finish_request.evaluate(cs._path, cs._run)
                if not granted and reason != cs._reported:
                    cs._reported = reason
                    logging.getLogger(finish_request.__name__).error(reason)
        return len(caplog.records)

    assert polls(5) == 1  # reported when it appears...
    assert polls(5) == 0  # ...and not again while it says the same thing

    write_request(path, action='abort', run='batch-1')
    assert polls(5) == 1  # a different refusal is a different fact


def test_bytes_that_are_not_utf8_are_a_refusal_not_an_exception(tmp_path):
    """`UnicodeDecodeError` is a `ValueError`, not an `OSError`, so it does not fall out of the read
    guard on its own — and this control system runs in the foreground, where an escaping exception
    takes the World down in the middle of an episode. Every other malformed input is a logged
    refusal; so is this one."""
    path = tmp_path / 'finish'
    path.write_bytes(b'\xff\xfe not utf-8 at all')

    granted, reason = finish_request.evaluate(path, 'batch-1')

    assert not granted
    assert 'could not be read' in reason


def test_an_action_outside_the_closed_set_is_a_refusal(tmp_path):
    """The action names a member of `Action`; anything else is a writer from a future this run does
    not implement, and acting on it would be guessing what it asked for."""
    path = tmp_path / 'finish'
    path.write_text(json.dumps({finish_request.ACTION_KEY: 'discard', finish_request.RUN_ID_KEY: 'batch-1'}))
    assert not finish_request.evaluate(path, 'batch-1')[0]


def test_the_default_request_directory_is_already_a_path(monkeypatch):
    """The domain type is the module's, and the string is converted where it enters — so no consumer
    re-derives it and the env override is the only place a `str` appears."""
    monkeypatch.delenv(finish_request.FINISH_REQUEST_DIR_ENV, raising=False)
    assert isinstance(finish_request.DEFAULT_FINISH_REQUEST_DIR, Path)
    assert isinstance(finish_request.request_dir(), Path)


def test_a_grant_survives_the_world_it_was_read_in(tmp_path):
    """One object is built per invocation and handed to every World of a sweep, so the grant lives on
    the INSTANCE. Held in `run`'s frame it would reset at each World, and the run could only learn it
    had been asked by re-reading a file that may by then have been retired — while a request
    addresses the RUN, which is one run however many Worlds it raises."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')
    cs = finish_request.FinishRequest(path, 'batch-1', poll_interval_s=0.0)

    granted, _ = finish_request.evaluate(cs._path, cs._run)
    cs._granted = granted
    assert cs._granted

    path.unlink()  # the writer retires it once the run is over; the grant is not un-asked
    assert cs._granted


def test_a_sweep_can_read_the_grant_without_entering_a_world(tmp_path):
    """The grant is readable outside a World, which is where a sweep decides whether to raise one."""
    path = tmp_path / 'finish'
    cs = finish_request.FinishRequest(path, 'batch-1', poll_interval_s=0.0)
    assert cs.granted is False

    write_request(path, run='batch-1')
    cs._granted = finish_request.evaluate(cs._path, cs._run)[0]
    assert cs.granted is True


def test_the_action_arrives_as_the_enum_not_the_string_it_was_written_as(tmp_path):
    """A `StrEnum` member compares equal to its own string, so a bare comparison passes while leaving
    every later reader holding a `str` the contract calls an `Action`. The conversion is the check."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')
    assert finish_request.evaluate(path, 'batch-1')[0]
    assert (
        finish_request.Action(json.loads(path.read_text())[finish_request.ACTION_KEY]) is finish_request.Action.FINISH
    )

    path.write_text(json.dumps({finish_request.RUN_ID_KEY: 'batch-1'}))  # no action at all
    granted, reason = finish_request.evaluate(path, 'batch-1')
    assert not granted and 'not a string this run can read' in reason

    path.write_text(json.dumps({finish_request.ACTION_KEY: 'abort', finish_request.RUN_ID_KEY: 'batch-1'}))
    granted, reason = finish_request.evaluate(path, 'batch-1')
    assert not granted and 'does not implement' in reason


# Every way a file can fail to become a request, as a table rather than a test per discovery. The
# class is what matters, not the members: this reader runs in the FOREGROUND, so an exception it
# does not answer for ends the World in the middle of an episode. Each entry writes the file itself,
# because what is under test is the read, not a fixture's idea of one.
def _write_bytes(b: bytes):
    return lambda p: p.write_bytes(b)


def _symlink_to_a_real_request(path: Path) -> None:
    """A symlink whose target IS a request addressed to this run — so following it would GRANT.

    Any account may create the link, and the account it points at need not be the one that may write
    the request; refusing the link is what keeps the two the same decision."""
    target = path.with_name('a-real-request')
    write_request(target, run='batch-1')
    path.symlink_to(target)


_UNREADABLE_SHAPES = [
    (_write_bytes(b'\xff\xfe not utf-8'), 'bytes that are not UTF-8 (UnicodeDecodeError)'),
    (_write_bytes(b'finish please'), 'text that is not JSON (JSONDecodeError)'),
    (_write_bytes(b'"finish"'), 'a JSON scalar rather than an object'),
    (_write_bytes(b'[1, 2, 3]'), 'a JSON array rather than an object'),
    (_write_bytes(b'[' * 200_000), 'nesting past the parser limit (RecursionError)'),
    (
        lambda p: p.write_text(
            json.dumps({
                finish_request.ACTION_KEY: finish_request.Action.FINISH,
                finish_request.RUN_ID_KEY: 'batch-1',
                'pad': 'x' * (finish_request.MAX_REQUEST_BYTES + 10),
            })
        ),
        'a file past the size bound',
    ),
    (
        lambda p: p.write_text(
            json.dumps({finish_request.ACTION_KEY: 'discard', finish_request.RUN_ID_KEY: 'batch-1'})
        ),
        'an action outside the closed set',
    ),
    (lambda p: p.write_text(json.dumps({finish_request.RUN_ID_KEY: 'batch-1'})), 'no action at all'),
    (
        lambda p: p.write_text(
            json.dumps({
                finish_request.ACTION_KEY: finish_request.Action.FINISH,
                finish_request.RUN_ID_KEY: 'someone-else',
            })
        ),
        'a request addressed to another run',
    ),
    (
        lambda p: p.write_text(json.dumps({finish_request.ACTION_KEY: finish_request.Action.FINISH})),
        'no addressee at all',
    ),
    (
        lambda p: p.write_text(
            json.dumps({finish_request.ACTION_KEY: ['finish'], finish_request.RUN_ID_KEY: 'batch-1'})
        ),
        'an action that is a JSON array (unhashable)',
    ),
    (
        lambda p: p.write_text(
            json.dumps({finish_request.ACTION_KEY: {'do': 'finish'}, finish_request.RUN_ID_KEY: 'batch-1'})
        ),
        'an action that is a JSON object (unhashable)',
    ),
    (
        lambda p: p.write_text(json.dumps({finish_request.ACTION_KEY: 7, finish_request.RUN_ID_KEY: 'batch-1'})),
        'an action that is a number',
    ),
    # The request directory is world-writable by design, so what sits at the path is not necessarily
    # a file. A device belongs to this class too, and is absent only because creating one needs root.
    # The FIFO has a test of its own below: it is the member that hangs rather than fails.
    (_symlink_to_a_real_request, 'a symlink rather than a file'),
    (lambda p: p.mkdir(), 'a directory rather than a file'),
]


@pytest.mark.parametrize(('write', 'shape'), _UNREADABLE_SHAPES, ids=[s[1].split(' (')[0] for s in _UNREADABLE_SHAPES])
def test_no_unreadable_request_ever_raises_out_of_the_reader(tmp_path, write, shape):
    """Total: every one of these is a refusal that leaves the run going, and none is an exception."""
    path = tmp_path / 'finish'
    write(path)

    granted, reason = finish_request.evaluate(path, 'batch-1')

    assert not granted, shape
    assert reason, f'a refusal says why, for the log: {shape}'


def test_a_fifo_at_the_request_path_is_refused_rather_than_waited_on(tmp_path):
    """The one shape that does not fail but HANGS, which no refusal can reach.

    Opened by name, a FIFO with no writer holds the `open` until one arrives, and this poller runs in
    the foreground, so the run stops there mid-episode. Bounded in a thread so a regression fails the
    suite instead of hanging it."""
    path = tmp_path / 'finish'
    os.mkfifo(path)
    outcome: list[tuple[bool, str]] = []
    reader = threading.Thread(target=lambda: outcome.append(finish_request.evaluate(path, 'batch-1')), daemon=True)

    reader.start()
    reader.join(timeout=10.0)

    assert not reader.is_alive(), 'the read blocked on the FIFO instead of refusing it'
    granted, reason = outcome[0]
    assert not granted
    assert reason, 'a refusal says why, for the log'


def test_the_unreadable_boundary_answers_for_a_shape_nobody_listed(tmp_path, monkeypatch):
    """The list above is what is known; the boundary is what makes the read total. A parse that
    raises something new still refuses rather than ending the World."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')

    def _boom(_raw):
        raise RecursionError('maximum recursion depth exceeded')

    monkeypatch.setattr(finish_request.json, 'loads', _boom)
    granted, reason = finish_request.evaluate(path, 'batch-1')

    assert not granted and 'could not be read as a request' in reason
