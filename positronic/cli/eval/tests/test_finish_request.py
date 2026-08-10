import json
import os
import threading
from pathlib import Path

import pytest

import pimm
from positronic.cli.eval import finish_request


def write_request(path: Path, *, action: str = finish_request.Action.FINISH, run: str, **extra) -> None:
    """An ordinary fixture, built from the constants the module defines, so a rename moves with them."""
    path.write_text(json.dumps({finish_request.ACTION_KEY: action, finish_request.RUN_ID_KEY: run, **extra}))


def test_a_request_naming_this_run_is_granted(tmp_path):
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')
    assert finish_request.evaluate(path, 'batch-1')


def test_an_absent_file_is_the_ordinary_state(tmp_path):
    """The one negative: no file, no request, and nothing wrong."""
    assert not finish_request.evaluate(tmp_path / 'nothing', 'batch-1')


def test_a_request_naming_another_run_raises(tmp_path):
    """A file at THIS run's path addressed to another run is a writer that got the path wrong. It
    cannot be honoured and must not be ignored: nothing else would ever report the mistake."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')
    with pytest.raises(ValueError, match='names run'):
        finish_request.evaluate(path, 'batch-2')


def test_an_unparseable_request_raises(tmp_path):
    path = tmp_path / 'finish'
    path.write_text('finish please')
    with pytest.raises(ValueError):
        finish_request.evaluate(path, 'batch-1')


def test_a_json_scalar_raises(tmp_path):
    """`json.loads` accepts a bare string, so parsing is not enough — the shape is checked too."""
    path = tmp_path / 'finish'
    path.write_text('"finish"')
    with pytest.raises(ValueError, match='not an object'):
        finish_request.evaluate(path, 'batch-1')


def test_an_unreadable_request_raises(tmp_path):
    """The umask trap: a writer whose umask is 077 leaves a file this account cannot open. Silence
    there would be a request nobody could send and nobody could see failing."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')
    path.chmod(0o000)
    try:
        if os.geteuid() == 0:
            pytest.skip('root reads a mode-000 file, so the unreadable branch cannot be reached')
        with pytest.raises(PermissionError):
            finish_request.evaluate(path, 'batch-1')
    finally:
        path.chmod(0o644)


def test_an_unknown_action_raises(tmp_path):
    """An action this run does not implement is a deploy that shipped a writer ahead of its runs.
    Acting on it would be guessing; ignoring it would leave the mismatch to be found by hand."""
    path = tmp_path / 'finish'
    write_request(path, action='abort', run='batch-1')
    with pytest.raises(ValueError, match='does not implement'):
        finish_request.evaluate(path, 'batch-1')


def test_extra_fields_do_not_break_a_request(tmp_path):
    """The writer records its own diagnostics in the same object; they are not this side's business."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1', requested_at_s=1.0, requested_by='someone')
    assert finish_request.evaluate(path, 'batch-1')


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
    assert finish_request.evaluate(finish_request.request_path('batch-1'), 'batch-1')
    assert not finish_request.evaluate(finish_request.request_path('batch-2'), 'batch-2')


def test_the_default_path_is_absolute_and_on_tmpfs(monkeypatch):
    """Both properties are load-bearing and neither is visible at a call site: an account-relative
    path would be a different file for the writer and the run, and a persistent one would let a
    request outlive the reboot that was supposed to clear it."""
    monkeypatch.delenv(finish_request.FINISH_REQUEST_DIR_ENV, raising=False)
    path = finish_request.request_path('batch-1')
    assert path.is_absolute()
    assert str(path).startswith('/run/')


@pytest.mark.parametrize('override', ['requests', './requests', '../requests'])
def test_a_relative_request_directory_raises_at_launch(monkeypatch, tmp_path, override):
    """A relative override resolves against each account's own working directory, so the writer and
    the run address different files from the same configuration. Raised before the first episode,
    since a run nobody can address is a misconfigured deploy rather than a run to start."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(finish_request.RUN_ID_ENV, 'batch-1')
    monkeypatch.setenv(finish_request.FINISH_REQUEST_DIR_ENV, override)
    with pytest.raises(ValueError, match='not absolute'):
        finish_request.from_env()


def test_an_absolute_request_directory_installs_the_poller(monkeypatch, tmp_path):
    """An absolute override installs the poller: what is refused is a relative path, not an override."""
    monkeypatch.setenv(finish_request.RUN_ID_ENV, 'batch-1')
    monkeypatch.setenv(finish_request.FINISH_REQUEST_DIR_ENV, str(tmp_path))
    assert finish_request.from_env() is not None


@pytest.mark.parametrize('run', ['a/b', '../elsewhere', '.', '..'])
def test_a_run_id_that_is_not_a_filename_raises_at_launch(monkeypatch, run):
    """The path is built from the run id, so an id carrying a separator would poll a file under a
    directory nobody agreed on — or, for `..`, outside the request directory entirely."""
    monkeypatch.setenv(finish_request.RUN_ID_ENV, run)
    with pytest.raises(ValueError, match='single path segment'):
        finish_request.from_env()


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
    assert finish_request.evaluate(path, 'batch_20260807-111935')
    with pytest.raises(ValueError, match='names run'):
        finish_request.evaluate(path, 'batch_20260807-110748')


def test_bytes_that_are_not_utf8_raise(tmp_path):
    """`UnicodeDecodeError` is a `ValueError`; nothing catches it, so the decode failure reaches the
    World and the broken writer is visible."""
    path = tmp_path / 'finish'
    path.write_bytes(b'\xff\xfe not utf-8 at all')

    with pytest.raises(UnicodeDecodeError):
        finish_request.evaluate(path, 'batch-1')


def test_an_action_outside_the_closed_set_raises(tmp_path):
    """The action names a member of `Action`; anything else is a writer this run cannot serve."""
    path = tmp_path / 'finish'
    path.write_text(json.dumps({finish_request.ACTION_KEY: 'discard', finish_request.RUN_ID_KEY: 'batch-1'}))
    with pytest.raises(ValueError, match='does not implement'):
        finish_request.evaluate(path, 'batch-1')


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

    cs._granted = finish_request.evaluate(cs._path, cs._run)
    assert cs._granted

    path.unlink()  # the writer retires it once the run is over; the grant is not un-asked
    assert cs._granted


def test_a_sweep_can_read_the_grant_without_entering_a_world(tmp_path):
    """The grant is readable outside a World, which is where a sweep decides whether to raise one."""
    path = tmp_path / 'finish'
    cs = finish_request.FinishRequest(path, 'batch-1', poll_interval_s=0.0)
    assert cs.granted is False

    write_request(path, run='batch-1')
    cs._granted = finish_request.evaluate(cs._path, cs._run)
    assert cs.granted is True


def test_the_action_arrives_as_the_enum_not_the_string_it_was_written_as(tmp_path):
    """A `StrEnum` member compares equal to its own string, so a bare comparison passes while leaving
    every later reader holding a `str` the contract calls an `Action`. The conversion is the check."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')
    assert finish_request.evaluate(path, 'batch-1')
    assert (
        finish_request.Action(json.loads(path.read_text())[finish_request.ACTION_KEY]) is finish_request.Action.FINISH
    )

    path.write_text(json.dumps({finish_request.RUN_ID_KEY: 'batch-1'}))  # no action at all
    with pytest.raises(ValueError, match='not a string'):
        finish_request.evaluate(path, 'batch-1')

    path.write_text(json.dumps({finish_request.ACTION_KEY: 'abort', finish_request.RUN_ID_KEY: 'batch-1'}))
    with pytest.raises(ValueError, match='does not implement'):
        finish_request.evaluate(path, 'batch-1')


# Every way a file can fail to be the request the contract describes. Each raises: the writer and
# the run are one system, so a file at this path that is not a request is breakage. Each entry
# writes the file itself, because what is under test is the read, not a fixture's idea of one.
def _write_bytes(b: bytes):
    return lambda p: p.write_bytes(b)


def _symlink_to_a_real_request(path: Path) -> None:
    """A symlink whose target IS a request addressed to this run — so following it would GRANT.

    Any account may create the link, and the account it points at need not be the one that may write
    the request; refusing the link is what keeps the two the same decision."""
    target = path.with_name('a-real-request')
    write_request(target, run='batch-1')
    path.symlink_to(target)


_CONTRACT_VIOLATIONS = [
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


@pytest.mark.parametrize(
    ('write', 'shape'), _CONTRACT_VIOLATIONS, ids=[s[1].split(' (')[0] for s in _CONTRACT_VIOLATIONS]
)
def test_every_contract_violation_raises(tmp_path, write, shape):
    """None of these is quietly ignored: each ends the run with what was wrong at the path."""
    path = tmp_path / 'finish'
    write(path)

    with pytest.raises(Exception) as caught:  # noqa: B017 — the class is the point, not one member
        finish_request.evaluate(path, 'batch-1')

    assert str(caught.value), f'the failure says what was wrong: {shape}'


def test_a_fifo_at_the_request_path_raises_rather_than_waiting_on_it(tmp_path):
    """The one shape that does not fail but HANGS, which no exception can reach.

    Opened by name, a FIFO with no writer holds the `open` until one arrives, and this poller runs in
    the foreground, so the run stops there mid-episode. `O_NONBLOCK` is what turns it into a failure
    that can be raised. Bounded in a thread so a regression fails the suite instead of hanging it."""
    path = tmp_path / 'finish'
    os.mkfifo(path)
    outcome: list[BaseException] = []

    def read() -> None:
        try:
            finish_request.evaluate(path, 'batch-1')
        except BaseException as e:  # noqa: BLE001 — recorded so the assert below can name it
            outcome.append(e)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    reader.join(timeout=10.0)

    assert not reader.is_alive(), 'the read blocked on the FIFO instead of failing on it'
    assert outcome and str(outcome[0]), 'the failure says what was wrong at the path'


def test_a_shape_nobody_listed_reaches_the_world_too(tmp_path, monkeypatch):
    """The list above is what is known; nothing catches what is not on it, which is the point. A
    parse failure of a kind nobody anticipated ends the run rather than reading as no request."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')

    def _boom(_raw):
        raise RecursionError('maximum recursion depth exceeded')

    monkeypatch.setattr(finish_request.json, 'loads', _boom)

    with pytest.raises(RecursionError):
        finish_request.evaluate(path, 'batch-1')


def test_the_poll_is_paced_on_the_clock_the_run_is_measured_on(tmp_path):
    """A simulated sweep advances episodes on the virtual clock, which runs as fast as the machine
    allows. A poll paced on WALL time is incommensurable with that: whole episodes pass between two
    reads, so a request written during one is still unread when the harness decides whether to open
    the next. Paced on the world's clock the poll interval means the same thing in both modes.
    """
    path = tmp_path / 'finish'
    cs = finish_request.FinishRequest(path, 'batch-1', poll_interval_s=2.0)

    with pimm.World(virtual_time=True) as world:
        loop = world.interleave(cs.run)
        next(loop)  # the first poll, before the request exists
        write_request(path, run='batch-1')
        deadline = world.clock.now() + 300.0  # 300 virtual seconds; almost no wall time passes
        while world.clock.now() < deadline and not cs.granted:
            next(loop)

    assert cs.granted, 'the request went unread while 300 seconds of the run went by'


def test_a_defect_in_this_run_s_own_code_reaches_the_world(tmp_path, monkeypatch):
    """A defect in the checks is breakage like any other: it surfaces rather than reading as a file
    that was never a request, which would leave the run going and the writer unacknowledged."""
    path = tmp_path / 'finish'
    write_request(path, run='batch-1')

    class Broken:
        def __iter__(self):
            raise ValueError('a defect in the staged checks')

    monkeypatch.setattr(finish_request, 'Action', Broken())

    with pytest.raises(ValueError, match='a defect in the staged checks'):
        finish_request.evaluate(path, 'batch-1')


def test_a_missing_request_directory_raises_on_the_read(tmp_path):
    """`ENOENT` is the same for an absent request and an absent directory, and the two mean opposite
    things: the first is every run nobody has asked, the second is a run nothing can ever ask. Read
    as the ordinary state it would disable finishing for the life of the run, silently."""
    path = tmp_path / 'not-a-directory' / 'finish'
    with pytest.raises(ValueError, match='directory'):
        finish_request.evaluate(path, 'batch-1')


def test_a_missing_request_directory_raises_at_launch(monkeypatch, tmp_path):
    """A mistyped override or an absent mount is a deploy that cannot be finished; it fails before
    the first episode rather than at a poll nothing is watching."""
    monkeypatch.setenv(finish_request.RUN_ID_ENV, 'batch-1')
    monkeypatch.setenv(finish_request.FINISH_REQUEST_DIR_ENV, str(tmp_path / 'absent'))
    with pytest.raises(ValueError, match='does not exist'):
        finish_request.from_env()


def test_a_run_id_too_long_for_the_directory_raises_at_launch(monkeypatch, tmp_path):
    """`names_one_segment` certifies the id; what has to fit is the id PLUS the prefix, against the
    directory's own component limit. Unchecked, the id passes at launch and the first poll dies of
    `ENAMETOOLONG` after the World is up and the arm has homed."""
    monkeypatch.setenv(finish_request.FINISH_REQUEST_DIR_ENV, str(tmp_path))
    limit = os.pathconf(tmp_path, 'PC_NAME_MAX')
    monkeypatch.setenv(finish_request.RUN_ID_ENV, 'x' * (limit - len(finish_request.FINISH_REQUEST_PREFIX) + 1))
    with pytest.raises(ValueError, match='filename limit'):
        finish_request.from_env()


def test_a_run_id_that_just_fits_installs_the_poller(monkeypatch, tmp_path):
    """The bound is the filesystem's, not a guess: the longest id that fits is accepted."""
    monkeypatch.setenv(finish_request.FINISH_REQUEST_DIR_ENV, str(tmp_path))
    limit = os.pathconf(tmp_path, 'PC_NAME_MAX')
    monkeypatch.setenv(finish_request.RUN_ID_ENV, 'x' * (limit - len(finish_request.FINISH_REQUEST_PREFIX)))
    assert finish_request.from_env() is not None
