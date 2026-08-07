import json
import logging
import os
from pathlib import Path

import pytest

from positronic.cli.eval import finish_request


def write_request(path: Path, *, action: str = finish_request.FINISH_ACTION, run: str, **extra) -> None:
    """An ordinary fixture, built from the constants this module defines, so a rename moves with them.
    The cross-repository compatibility fixture at the bottom deliberately does not."""
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


def test_the_run_id_and_path_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(finish_request.RUN_ID_ENV, 'batch-7')
    monkeypatch.setenv(finish_request.FINISH_REQUEST_PATH_ENV, str(tmp_path / 'elsewhere'))
    cs = finish_request.from_env()
    assert cs is not None
    assert cs._run == 'batch-7'
    assert cs._path == tmp_path / 'elsewhere'


def test_the_default_path_is_absolute_and_on_tmpfs(monkeypatch):
    """Both properties are load-bearing and neither is visible at a call site: an account-relative
    path would be a different file for the writer and the run, and a persistent one would let a
    request outlive the reboot that was supposed to clear it."""
    monkeypatch.delenv(finish_request.FINISH_REQUEST_PATH_ENV, raising=False)
    path = finish_request.request_path()
    assert path.is_absolute()
    assert str(path).startswith('/run/')


def test_the_object_the_writer_sends_is_granted(tmp_path):
    """The cross-repo contract, spelled as literals rather than through this module's own constants.

    The writer is the rollouts MCP in `Positronic-Robotics/platform`, which cannot import this module
    and which this module cannot import. So the only thing holding the two spellings together is a
    test on each side that names them: this one, and `rollouts/mcp/tests/test_ops.py`'s
    `test_a_written_request_is_the_object_the_run_requires`. Renaming a key here without renaming it
    there produces a run that ignores every request it is sent, silently, and the first sign of it is
    a rollout nobody can stop.
    """
    path = tmp_path / 'positronic_rollout_finish'
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
