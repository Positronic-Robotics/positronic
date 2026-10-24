"""Score a policy image on a public MolmoSpaces eval on Nebius Serverless, with no GPU of your own.

    uv run positronic/cli/examples/nebius_competition/serverless_eval.py \\
        --policy-image=docker.io/<you>/policy:v1 --subnet-id=<vpcsubnet-...>
    uv run positronic/cli/examples/nebius_competition/serverless_eval.py ... --dry-run   # validate, create nothing

It serves the image as an Endpoint and runs the eval as a Job against it. It prints the score, deletes both,
and checks that neither is left. Nebius bills each of them until it is deleted.

It needs the `nebius` CLI with a profile. Everything runs in the project of the profile, or of `--parent-id`,
and in the region of that project. The bearer token of the endpoint goes through a MysteryBox secret that the
command makes and deletes, so no command line holds the token.
"""

from __future__ import annotations

import argparse
import functools
import json
import secrets
import signal
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from platform_client.policy_container import AUTH_TOKEN_ENV, POLICY_PORT, PROVISIONING_DEADLINE_S
from positronic_evals.evals import EvalRef, public_eval

from positronic.cli.eval.run import SCORE_LINE_PREFIX
from positronic.utils import nebius as nebius_cli
from positronic.utils.nebius import NebiusError

DEFAULT_SIM_IMAGE = 'docker.io/positro/molmospaces:main'
DEFAULT_EVAL = 'molmo.franka_pick_mini_smoke'
# The cheapest Serverless GPU that runs the sim. eu-north1 has it.
DEFAULT_PLATFORM = 'gpu-l40s-a'
DEFAULT_PRESET = '1gpu-8vcpu-32gb'
DEFAULT_TIMEOUT_MIN = 120

# What `docker/run_molmospaces_eval.sh` reads. No constant spans both languages.
POLICY_HOST_ENV = 'POLICY_HOST'
EVAL_ENV = 'EVAL'
POLICY_WAIT_ENV = 'POLICY_WAIT_S'

JOB_TERMINAL_STATES = frozenset({'COMPLETED', 'FAILED', 'CANCELLED', 'ERROR'})
ENDPOINT_FAILED_STATES = frozenset({'ERROR'})
URL_WAIT_S = 600.0
# The job's own timeout ends it first. This margin covers the time that Nebius takes to stop it.
JOB_GRACE_S = 600.0
POLL_S = 30.0
DELETE_WAIT_S = 120.0
LOG_TAIL_LINES = 40
LOG_ATTEMPTS = 3
RETRY_S = 10.0
CLI_TIMEOUT_S = 300.0


class RunFailed(RuntimeError):
    """The run ended with no score."""


@dataclass(frozen=True)
class Machine:
    platform: str
    preset: str


@dataclass(frozen=True)
class Run:
    """One run: an endpoint for the policy, a job for the sim, and the secret that holds the token."""

    name: str
    parent_id: str
    subnet_id: str
    policy_image: str
    sim_image: str
    eval: EvalRef
    policy: Machine
    sim: Machine
    timeout_min: int
    registry_secret: str | None = None
    profile: str | None = None

    @property
    def cli(self) -> str:
        """The `nebius` command that reaches the project of the run."""
        return f'nebius --profile {self.profile}' if self.profile else 'nebius'

    @property
    def endpoint_name(self) -> str:
        return f'{self.name}-policy'

    @property
    def job_name(self) -> str:
        return f'{self.name}-sim'

    @property
    def secret_name(self) -> str:
        return f'{self.name}-token'


class Kind(Enum):
    """What a run makes, as the `nebius` service and resource that make it."""

    SECRET = ('mysterybox', 'secret')
    ENDPOINT = ('ai', 'endpoint')
    JOB = ('ai', 'job')


@dataclass(frozen=True)
class Clock:
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep


def endpoint_create_args(run: Run, token_env: list[str]) -> list[str]:
    """The arguments after `nebius ai endpoint create`."""
    registry = ['--registry-secret', run.registry_secret] if run.registry_secret else []
    return [
        *('--parent-id', run.parent_id, '--name', run.endpoint_name),
        *('--image', run.policy_image, '--container-port', str(POLICY_PORT)),
        *('--platform', run.policy.platform, '--preset', run.policy.preset, '--subnet-id', run.subnet_id),
        *token_env,
        *registry,
    ]


def job_create_args(run: Run, policy_host: str, token_env: list[str]) -> list[str]:
    """The arguments after `nebius ai job create`."""
    return [
        *('--parent-id', run.parent_id, '--name', run.job_name, '--image', run.sim_image),
        *('--platform', run.sim.platform, '--preset', run.sim.preset, '--subnet-id', run.subnet_id),
        *('--timeout', f'{run.timeout_min}m', '--restart-policy', 'never'),
        *('--env', f'{POLICY_HOST_ENV}={policy_host}', '--env', f'{EVAL_ENV}={run.eval}'),
        *('--env', f'{POLICY_WAIT_ENV}={PROVISIONING_DEADLINE_S:.0f}'),
        *token_env,
    ]


def https_host(public_endpoints: list[str]) -> str | None:
    """The host of the managed https:// URL that fronts the policy port, or None while it has none."""
    https = [url for url in public_endpoints if url.startswith('https://')]
    fronting = [url for url in https if url.startswith(f'https://port{POLICY_PORT}-')]
    # The endpoint declares one port, so a lone https URL fronts it whatever its name.
    chosen = fronting or (https if len(https) == 1 else [])
    return chosen[0].removeprefix('https://').rstrip('/') if chosen else None


def last_score_line(logs: str) -> str | None:
    """The last score line that the eval printed."""
    lines = [line.strip() for line in logs.splitlines() if line.strip().startswith(SCORE_LINE_PREFIX)]
    return lines[-1] if lines else None


class ServerlessEval:
    """Makes the resources of one run, waits for the score, and deletes what it made."""

    def __init__(self, nebius: Callable[..., Any], run: Run, clock: Clock | None = None):
        self.nebius = nebius
        self.run = run
        self.clock = clock or Clock()
        # The ids of what the run made, so that the cleanup deletes exactly that.
        self.made: dict[Kind, str] = {}

    def dry_run(self) -> None:
        """Validate both create requests. The token is a placeholder: the secret it would name does not exist."""
        token_env = ['--env', f'{AUTH_TOKEN_ENV}=dry-run']
        endpoint_args = endpoint_create_args(self.run, token_env)
        self.nebius(*Kind.ENDPOINT.value, 'create', *endpoint_args, '--dry-run', structured=False)
        job_args = job_create_args(self.run, 'dry-run.invalid', token_env)
        self.nebius(*Kind.JOB.value, 'create', *job_args, '--dry-run', structured=False)

    def score(self) -> str:
        """The score line of the eval. Raises `RunFailed` or `NebiusError` when the run ends with none."""
        token_env = ['--env-secret', f'{AUTH_TOKEN_ENV}={self.run.secret_name}']
        self._make_token_secret()
        endpoint_args = [*endpoint_create_args(self.run, token_env), '--async']
        endpoint = self._create(Kind.ENDPOINT, self.run.endpoint_name, endpoint_args)
        host = self._wait_for_host(endpoint)
        print(f'The policy endpoint is at https://{host}')
        job = self._create(Kind.JOB, self.run.job_name, [*job_create_args(self.run, host, token_env), '--async'])
        state = self._wait_for_job(job, endpoint)
        logs = self._logs(job)
        found = last_score_line(logs)
        if state != 'COMPLETED' or found is None:
            tail = '\n'.join(logs.splitlines()[-LOG_TAIL_LINES:])
            raise RunFailed(f'The eval job ended {state} with no score. The end of its log:\n{tail}')
        return found

    def _make_token_secret(self) -> None:
        """The request goes in a file, so no command line holds the token."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'secret.json'
            path.touch(mode=0o600)
            payload = [{'key': AUTH_TOKEN_ENV, 'string_value': secrets.token_urlsafe(32)}]
            request = {
                'metadata': {'parent_id': self.run.parent_id, 'name': self.run.secret_name},
                'spec': {'secret_version': {'payload': payload}},
            }
            path.write_text(json.dumps(request))
            self._create(Kind.SECRET, self.run.secret_name, ['--file', str(path)])

    def _create(self, kind: Kind, name: str, args: list[str]) -> str:
        """Create, then read the id by name: a create that fails can still leave a resource to delete."""
        try:
            self.nebius(*kind.value, 'create', *args)
        finally:
            resource_id = self._id(kind, name)
            if resource_id is not None:
                self.made[kind] = resource_id
                delete = f'{self.run.cli} {" ".join(kind.value)} delete --id {resource_id}'
                print(f'Made {name}. If this command stops: {delete}')
        if resource_id is None:
            raise RunFailed(f'Nebius lists no {" ".join(kind.value)} named {name} after its create')
        return resource_id

    def _id(self, kind: Kind, name: str) -> str | None:
        try:
            found = self.nebius(*kind.value, 'get-by-name', '--parent-id', self.run.parent_id, '--name', name)
        except NebiusError:
            return None
        return found['metadata']['id']

    def _state(self, kind: Kind, resource_id: str) -> dict[str, Any]:
        return self.nebius(*kind.value, 'get', '--id', resource_id).get('status', {})

    def _wait_for_host(self, endpoint: str) -> str:
        deadline = self.clock.monotonic() + URL_WAIT_S
        while True:
            status = self._state(Kind.ENDPOINT, endpoint)
            if status.get('state') in ENDPOINT_FAILED_STATES:
                raise RunFailed(f'The policy endpoint is {status["state"]}: {status}')
            host = https_host(status.get('public_endpoints', []))
            if host is not None:
                return host
            if self.clock.monotonic() >= deadline:
                raise RunFailed(f'The policy endpoint got no https:// URL in {URL_WAIT_S:.0f}s')
            self.clock.sleep(POLL_S)

    def _wait_for_job(self, job: str, endpoint: str) -> str:
        deadline = self.clock.monotonic() + self.run.timeout_min * 60 + JOB_GRACE_S
        shown = None
        while True:
            state = self._state(Kind.JOB, job).get('state')
            if state != shown:
                print(f'The eval job is {state}')
                shown = state
            if state in JOB_TERMINAL_STATES:
                return state
            status = self._state(Kind.ENDPOINT, endpoint)
            if status.get('state') in ENDPOINT_FAILED_STATES:
                raise RunFailed(f'The policy endpoint is {status["state"]} while the eval runs: {status}')
            if self.clock.monotonic() >= deadline:
                raise RunFailed(f'The eval job did not end in {self.run.timeout_min} min')
            self.clock.sleep(POLL_S)

    def _logs(self, job: str) -> str:
        since = f'{self.run.timeout_min * 60 + JOB_GRACE_S:.0f}s'
        logs = ''
        for _attempt in range(LOG_ATTEMPTS):  # The log store can lag the end of the job.
            logs = self.nebius(*Kind.JOB.value, 'logs', job, '--since', since, structured=False)
            if last_score_line(logs) is not None:
                break
            self.clock.sleep(RETRY_S)
        return logs

    def cleanup(self) -> list[str]:
        """Delete what the run made. Return each delete that failed and each resource that is left."""
        problems = []
        if Kind.JOB in self.made:
            problems += self._attempt(self._stop_job, self.made[Kind.JOB])
        # The job first, then the endpoint it dials, then the secret both read.
        for kind in (Kind.JOB, Kind.ENDPOINT, Kind.SECRET):
            if kind in self.made:
                problems += self._attempt(self.nebius, *kind.value, 'delete', '--id', self.made[kind], structured=False)
        return problems + self._left()

    def _stop_job(self, job: str) -> None:
        if self._state(Kind.JOB, job).get('state') not in JOB_TERMINAL_STATES:
            self.nebius(*Kind.JOB.value, 'cancel', '--id', job, structured=False)

    @staticmethod
    def _attempt(call: Callable[..., Any], *args: Any, **kwargs: Any) -> list[str]:
        try:
            call(*args, **kwargs)
        except NebiusError as exc:
            return [str(exc)]
        return []

    def _left(self) -> list[str]:
        """The endpoint and the job of this run that Nebius still lists, after a wait for the deletes to end."""
        names = {self.run.endpoint_name: Kind.ENDPOINT, self.run.job_name: Kind.JOB}
        deadline = self.clock.monotonic() + DELETE_WAIT_S
        while True:
            try:
                listed = [
                    item['metadata']['name']
                    for kind in (Kind.ENDPOINT, Kind.JOB)
                    for item in self.nebius(*kind.value, 'list', '--parent-id', self.run.parent_id).get('items', [])
                ]
            except NebiusError as exc:
                return [f'Could not check what is left: {exc}']
            left = [f'The {" ".join(names[name].value)} {name} is still there' for name in listed if name in names]
            if not left or self.clock.monotonic() >= deadline:
                return left
            self.clock.sleep(RETRY_S)


def _interrupt(signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt(f'signal {signum}')


@contextmanager
def _signals(handlers: dict[signal.Signals, Any]) -> Iterator[None]:
    previous = {signum: signal.signal(signum, handler) for signum, handler in handlers.items()}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _score_then_delete(session: ServerlessEval) -> str:
    """The score line. The deletes run whatever happens before them, and a stop signal does not stop them."""
    failure = None
    found = None
    with _signals({signal.SIGTERM: _interrupt, signal.SIGHUP: _interrupt}):
        try:
            found = session.score()
        except (NebiusError, RunFailed) as exc:
            failure = str(exc)
        except KeyboardInterrupt:
            failure = 'Stopped before the end'
        finally:
            print('Deleting the endpoint, the job and the token secret')
            with _signals({signal.SIGINT: signal.SIG_IGN, signal.SIGTERM: signal.SIG_IGN}):
                problems = session.cleanup()
    if problems:
        listing, parent = f'{session.run.cli} ai', f'--parent-id {session.run.parent_id}'
        print('\n'.join(problems), file=sys.stderr)
        failure = (
            f'{failure}\n' if failure else ''
        ) + f'Delete what is left: `{listing} endpoint list {parent}`, `{listing} job list {parent}`'
    else:
        print('No endpoint or job of this run is left')
    if failure is not None or found is None:
        raise SystemExit(failure)
    return found


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--policy-image', required=True, help='the policy image that the endpoint serves')
    parser.add_argument('--subnet-id', required=True, help='a VPC subnet of the project: `nebius vpc subnet list`')
    parser.add_argument('--eval', default=DEFAULT_EVAL, help=f'a public eval, default {DEFAULT_EVAL}')
    parser.add_argument('--parent-id', default=None, help='the project, default the project of the profile')
    parser.add_argument('--profile', default=None, help='a `nebius` profile other than the default one')
    parser.add_argument('--sim-image', default=DEFAULT_SIM_IMAGE, help=f'default {DEFAULT_SIM_IMAGE}')
    parser.add_argument('--policy-platform', default=DEFAULT_PLATFORM)
    parser.add_argument('--policy-preset', default=DEFAULT_PRESET)
    parser.add_argument('--sim-platform', default=DEFAULT_PLATFORM)
    parser.add_argument('--sim-preset', default=DEFAULT_PRESET)
    parser.add_argument(
        '--timeout-min', type=int, default=DEFAULT_TIMEOUT_MIN, help='the job timeout and the wait, in minutes'
    )
    parser.add_argument('--registry-secret', default=None, help='a MysteryBox secret to pull a private policy image')
    parser.add_argument('--dry-run', action='store_true', help='validate both create requests; create nothing')
    return parser


def main(argv: list[str] | None = None, nebius: Callable[..., Any] | None = None, clock: Clock | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        public_eval(EvalRef(args.eval))
    except LookupError as exc:
        raise SystemExit(str(exc)) from None
    nebius = nebius or functools.partial(nebius_cli.call, profile=args.profile, timeout=CLI_TIMEOUT_S)
    try:
        parent_id = args.parent_id or nebius('config', 'get', 'parent-id', structured=False).strip()
    except NebiusError as exc:
        raise SystemExit(str(exc)) from None
    run = Run(
        name=f'molmo-eval-{datetime.now(UTC):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}',
        parent_id=parent_id,
        subnet_id=args.subnet_id,
        policy_image=args.policy_image,
        sim_image=args.sim_image,
        eval=EvalRef(args.eval),
        policy=Machine(args.policy_platform, args.policy_preset),
        sim=Machine(args.sim_platform, args.sim_preset),
        timeout_min=args.timeout_min,
        registry_secret=args.registry_secret,
        profile=args.profile,
    )
    session = ServerlessEval(nebius, run, clock)
    if args.dry_run:
        try:
            session.dry_run()
        except NebiusError as exc:
            raise SystemExit(str(exc)) from None
        print(f'Nebius accepts the endpoint and the job of {run.name}. Nothing was made.')
        return

    print(_score_then_delete(session))


if __name__ == '__main__':
    main()
