"""`serverless_eval.py`: what a run makes, what it prints, and that it deletes everything it made."""

from __future__ import annotations

import json
import os
import runpy
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'serverless_eval.py'
SCORE = 'score 0.600 over 5 episodes, 0 unscored'
POLICY_URL_HOST = 'port8000-abc.endpoints.example'
ARGS = ['--policy-image=docker.io/team/policy:v1', '--subnet-id=vpcsubnet-test']


@pytest.fixture(scope='module')
def script() -> dict[str, Any]:
    """The script's globals, loaded from its path: the examples directory is not a package."""
    return runpy.run_path(str(SCRIPT))


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeNebius:
    """The `nebius` CLI over an in-memory project. A job walks through `job_states` and stays at the last one.

    `interrupt_on_job_get` makes the first read of the job raise KeyboardInterrupt, as Ctrl-C does.
    `failing_lookup` makes the first `lookup_failures` lookups by name of that kind fail with an error other than
    NotFound.
    """

    def __init__(
        self,
        error: type[Exception],
        not_found: type[Exception],
        *,
        job_states: tuple[str, ...] = ('PROVISIONING', 'RUNNING', 'COMPLETED'),
        logs: str = f'Waiting for the policy server\n{SCORE}\n',
        failing_create: str | None = None,
        undeletable: str | None = None,
        failing_lookup: str | None = None,
        lookup_failures: int = 1,
        endpoint_states: tuple[str, ...] = ('RUNNING',),
        interrupt_on_job_get: bool = False,
    ):
        self.error = error
        self.not_found = not_found
        self.failing_lookup = failing_lookup
        self.lookup_failures = lookup_failures
        self.endpoint_states: Iterator[str] = iter(endpoint_states)
        self.endpoint_state = 'PROVISIONING'
        self.job_states: Iterator[str] = iter(job_states)
        self.job_state = 'PROVISIONING'
        self.logs = logs
        self.failing_create = failing_create
        self.undeletable = undeletable
        self.interrupt_on_job_get = interrupt_on_job_get
        self.resources: dict[str, dict[str, str]] = {'secret': {}, 'endpoint': {}, 'job': {}}
        self.calls: list[tuple[str, ...]] = []
        self.tokens: list[str] = []
        self.request_file_modes: list[int] = []
        self.request_files: list[Path] = []
        self.endpoint_polls = 0

    def __call__(self, *args: str, structured: bool = True) -> Any:
        self.calls.append(args)
        if args[:2] == ('config', 'get'):
            return 'project-test\n'
        kind, verb = args[1], args[2]
        handler = getattr(self, f'_{verb.replace("-", "_")}')
        return handler(kind, args)

    @staticmethod
    def _value(args: tuple[str, ...], flag: str) -> str:
        return args[args.index(flag) + 1]

    def _create(self, kind: str, args: tuple[str, ...]) -> dict[str, Any] | str:
        if '--dry-run' in args:
            return 'validation succeeded'
        if kind == 'secret':
            path = Path(self._value(args, '--file'))
            self.request_files.append(path)
            self.request_file_modes.append(stat.S_IMODE(os.stat(path).st_mode))
            request = json.loads(path.read_text())
            name = request['metadata']['name']
            self.tokens.append(request['spec']['secret_version']['payload'][0]['string_value'])
        else:
            name = self._value(args, '--name')
        self.resources[kind][name] = f'{kind}-{len(self.calls)}'
        if kind == self.failing_create:
            raise self.error(f'create {kind} failed after it made {name}')
        return {'resource_id': self.resources[kind][name]}

    def _get_by_name(self, kind: str, args: tuple[str, ...]) -> dict[str, Any]:
        name = self._value(args, '--name')
        if kind == self.failing_lookup and self.lookup_failures > 0:
            self.lookup_failures -= 1
            raise self.error(f'get-by-name {kind} timed out')
        if name not in self.resources[kind]:
            raise self.not_found(f'NotFound: {name}')
        return {'metadata': {'id': self.resources[kind][name], 'name': name}}

    def _get(self, kind: str, args: tuple[str, ...]) -> dict[str, Any]:
        if kind == 'endpoint':
            self.endpoint_polls += 1
            urls = ['10.0.0.1:8000'] + ([f'https://{POLICY_URL_HOST}'] if self.endpoint_polls > 1 else [])
            self.endpoint_state = next(self.endpoint_states, self.endpoint_state)
            return {'status': {'state': self.endpoint_state, 'public_endpoints': urls}}
        if self.interrupt_on_job_get:
            self.interrupt_on_job_get = False
            raise KeyboardInterrupt
        self.job_state = next(self.job_states, self.job_state)
        return {'status': {'state': self.job_state}}

    def _logs(self, kind: str, args: tuple[str, ...]) -> str:
        return self.logs

    def _cancel(self, kind: str, args: tuple[str, ...]) -> str:
        self.job_state = 'CANCELLED'
        return ''

    def _delete(self, kind: str, args: tuple[str, ...]) -> str:
        resource_id = self._value(args, '--id')
        if kind != self.undeletable:
            self.resources[kind] = {n: i for n, i in self.resources[kind].items() if i != resource_id}
        return ''

    def _list(self, kind: str, args: tuple[str, ...]) -> dict[str, Any]:
        return {'items': [{'metadata': {'id': i, 'name': name}} for name, i in self.resources[kind].items()]}

    def verbs(self, kind: str) -> list[str]:
        return [call[2] for call in self.calls if call[1:2] == (kind,)]


def _run(script: dict[str, Any], fake: FakeNebius, *extra: str) -> None:
    script['main']([*ARGS, *extra], nebius=fake, clock=FakeClock())


def _fake(script: dict[str, Any], **kwargs: Any) -> FakeNebius:
    return FakeNebius(script['NebiusError'], script['NebiusNotFound'], **kwargs)


def test_a_run_prints_the_score_last_and_leaves_nothing(script, capsys):
    fake = _fake(script)
    _run(script, fake)
    assert capsys.readouterr().out.splitlines()[-1] == SCORE
    assert fake.resources == {'secret': {}, 'endpoint': {}, 'job': {}}


def test_the_token_reaches_nebius_in_a_private_file_and_in_no_command_line(script):
    fake = _fake(script)
    _run(script, fake)
    (token,) = fake.tokens
    assert fake.request_file_modes == [0o600]
    assert not fake.request_files[0].exists()
    assert not [call for call in fake.calls if any(token in arg for arg in call)]


def test_both_workloads_read_the_token_from_the_secret_and_the_job_dials_the_https_host(script):
    fake = _fake(script)
    _run(script, fake)
    creates = {call[1]: call for call in fake.calls if call[2] == 'create' and call[1] != 'secret'}
    secret_name = next(iter(c for c in fake.calls if c[2] == 'get-by-name' and c[1] == 'secret'))[-1]
    for call in creates.values():
        assert f'AUTH_TOKEN={secret_name}' in call
    assert f'POLICY_HOST={POLICY_URL_HOST}' in creates['job']
    assert creates['job'][creates['job'].index('--timeout') + 1] == '120m'


def test_a_failed_job_exits_with_its_log_and_leaves_nothing(script):
    fake = _fake(script, job_states=('RUNNING', 'FAILED'), logs='Traceback\nRuntimeError: the sim died\n')
    with pytest.raises(SystemExit) as exit_info:
        _run(script, fake)
    assert 'ended FAILED' in str(exit_info.value)
    assert 'the sim died' in str(exit_info.value)
    assert fake.resources == {'secret': {}, 'endpoint': {}, 'job': {}}


def test_a_completed_job_with_no_score_line_is_a_failure(script):
    fake = _fake(script, logs='the eval printed nothing\n')
    with pytest.raises(SystemExit) as exit_info:
        _run(script, fake)
    assert 'ended COMPLETED with no score' in str(exit_info.value)


def test_a_create_that_fails_after_it_made_the_endpoint_still_deletes_it(script):
    fake = _fake(script, failing_create='endpoint')
    with pytest.raises(SystemExit) as exit_info:
        _run(script, fake)
    assert 'create endpoint failed' in str(exit_info.value)
    assert 'delete' in fake.verbs('endpoint')
    assert fake.resources == {'secret': {}, 'endpoint': {}, 'job': {}}


def test_a_create_whose_lookup_fails_is_found_by_name_and_deleted(script):
    fake = _fake(script, failing_lookup='endpoint')
    with pytest.raises(SystemExit) as exit_info:
        _run(script, fake)
    assert 'get-by-name endpoint timed out' in str(exit_info.value)
    assert 'delete' in fake.verbs('endpoint')
    assert fake.resources == {'secret': {}, 'endpoint': {}, 'job': {}}


def test_a_lookup_that_fails_in_the_cleanup_is_reported(script, capsys):
    fake = _fake(script, failing_lookup='job', lookup_failures=2)
    with pytest.raises(SystemExit) as exit_info:
        _run(script, fake)
    assert 'get-by-name job timed out' in capsys.readouterr().err
    assert 'nebius ai job list' in str(exit_info.value)


def test_an_endpoint_that_stops_while_the_job_runs_ends_the_run_and_leaves_nothing(script):
    fake = _fake(script, endpoint_states=('RUNNING', 'RUNNING', 'STOPPED'), job_states=('RUNNING',))
    with pytest.raises(SystemExit) as exit_info:
        _run(script, fake)
    assert 'The policy endpoint is STOPPED while the eval runs' in str(exit_info.value)
    assert fake.verbs('job')[-3:] == ['cancel', 'delete', 'list']
    assert fake.resources == {'secret': {}, 'endpoint': {}, 'job': {}}


def test_an_interrupt_while_the_job_runs_cancels_it_and_leaves_nothing(script):
    fake = _fake(script, interrupt_on_job_get=True)
    with pytest.raises(SystemExit) as exit_info:
        _run(script, fake)
    assert 'Stopped before the end' in str(exit_info.value)
    assert fake.resources == {'secret': {}, 'endpoint': {}, 'job': {}}


def test_a_job_that_outlives_its_timeout_is_cancelled_and_deleted(script):
    fake = _fake(script, job_states=('RUNNING',))
    with pytest.raises(SystemExit) as exit_info:
        _run(script, fake, '--timeout-min=60')
    assert 'did not end in 60 min' in str(exit_info.value)
    assert fake.verbs('job')[-3:] == ['cancel', 'delete', 'list']
    assert fake.resources['job'] == {}


def test_a_resource_still_listed_after_its_delete_is_reported(script, capsys):
    fake = _fake(script, undeletable='endpoint')
    with pytest.raises(SystemExit) as exit_info:
        _run(script, fake)
    assert 'nebius ai endpoint list' in str(exit_info.value)
    assert '-policy is still there' in capsys.readouterr().err


def test_the_delete_commands_it_prints_name_the_profile_of_the_run(script, capsys):
    fake = _fake(script, undeletable='endpoint')
    with pytest.raises(SystemExit) as exit_info:
        _run(script, fake, '--profile=team', '--parent-id=project-team')
    assert 'nebius --profile team ai endpoint list --parent-id project-team' in str(exit_info.value)
    assert 'If this command stops: nebius --profile team ai endpoint delete --id' in capsys.readouterr().out


def test_a_dry_run_validates_both_requests_and_makes_nothing(script, capsys):
    fake = _fake(script)
    _run(script, fake, '--dry-run')
    creates = [call for call in fake.calls if call[2] == 'create']
    assert [call[1] for call in creates] == ['endpoint', 'job']
    assert all('--dry-run' in call for call in creates)
    assert fake.resources == {'secret': {}, 'endpoint': {}, 'job': {}}
    assert 'Nothing was made' in capsys.readouterr().out


def test_an_unknown_eval_is_refused_before_any_call(script):
    fake = _fake(script)
    with pytest.raises(SystemExit) as exit_info:
        _run(script, fake, '--eval=molmo.nope')
    assert 'not a public eval' in str(exit_info.value)
    assert fake.calls == []


@pytest.mark.parametrize(
    ('urls', 'host'),
    [
        (['10.0.0.1:8000', 'https://port8000-a.example'], 'port8000-a.example'),
        (['https://port9000-b.example', 'https://port8000-a.example/'], 'port8000-a.example'),
        (['https://lone.example'], 'lone.example'),
        (['https://one.example', 'https://two.example'], None),
        (['10.0.0.1:8000'], None),
        ([], None),
    ],
)
def test_the_policy_host_is_the_https_url_that_fronts_the_policy_port(script, urls, host):
    assert script['https_host'](urls) == host
