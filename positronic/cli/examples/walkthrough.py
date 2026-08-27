"""Drive one submission from registration to a board, against any platform.

    POSITRONIC_PLATFORM_CREDENTIAL=<token> uv run positronic/cli/examples/walkthrough.py

`uv run` builds the environment this needs from the checkout, so nothing has to be installed first.
The credential is read from the environment rather than taken as an argument: a command line is
readable by every process on the box and lands in shell history.

Every call goes through `PlatformClient`, so each response is a typed model rather than a dict. The
same flow from the command line is `positronic account register`, `positronic eval run` and
`positronic eval status`.
"""

from __future__ import annotations

import argparse
import os
import time

from platform_client.client import CREDENTIAL_ENV, PlatformClient
from platform_client.enums import NO_RESULT_STATUSES, TERMINAL_STATUSES, KeyStatus
from platform_client.errors import PlatformError
from platform_client.evals import EvalRef
from platform_client.ids import SubmissionId
from platform_client.policy_images import PolicyImage
from platform_client.requests import RegisterRequest, SubmissionCreateRequest
from platform_client.responses import ErroredSubmissionView, FinishedSubmissionView, SubmissionView


def authenticate(client: PlatformClient, *, credential: str, alias: str) -> None:
    """Leave the client holding a usable key, rotating if this identity is already registered.

    `register` keeps whatever key it is given. A key's plaintext is stored only as a hash, so a
    second registration reports `existing` and carries none; rotating issues a fresh one and
    retires the old.
    """
    registration = client.register(RegisterRequest(credential=credential, alias=alias))
    print(f'   user {registration.user_id} ({registration.key_status.name})')
    if registration.api_key is None:
        registration = client.register(RegisterRequest(credential=credential, alias=alias, rotate=True))
        print(f'   rotated ({registration.key_status.name})')
    assert client.api_key is not None, f'expected a key after {KeyStatus.rotated.name}'


def poll_until_terminal(
    client: PlatformClient, submission_id: SubmissionId, *, timeout_s: float, poll_s: float = 0.2
) -> SubmissionView:
    """Poll one submission to a terminal status, printing each distinct status on the way."""
    deadline = time.monotonic() + timeout_s
    seen: list[str] = []
    while time.monotonic() < deadline:
        view = client.get_submission(submission_id)
        if not seen or seen[-1] != view.status.name:
            seen.append(view.status.name)
            print(f'   status: {view.status.name}')
        if view.status in TERMINAL_STATUSES:
            return view
        time.sleep(poll_s)
    raise TimeoutError(f'submission {submission_id} never reached a terminal status (saw {seen})')


def print_quota(client: PlatformClient) -> None:
    """What the caller's plan allows, and what is left of it right now."""
    me = client.me()
    print(f'   {me.tenant} on {me.plan}')
    for limit in me.quota:
        remaining, allowed = limit.remaining / limit.scale, limit.limit / limit.scale
        print(f'   {limit.key} ({limit.window}): {remaining:g} of {allowed:g} {limit.unit} left')


def walkthrough(
    client: PlatformClient,
    *,
    credential: str,
    alias: str,
    eval_ref: EvalRef,
    policy_image: PolicyImage,
    timeout_s: float,
) -> None:
    print('1. register')
    authenticate(client, credential=credential, alias=alias)

    print('2. submit')
    # The eval is the whole of the choice: it names the embodiment its tasks run on, and asking for
    # one the platform does not offer comes back with the names it does, under `PlatformError.evals`.
    submission = client.create_submission(SubmissionCreateRequest(policy_image=policy_image, eval=eval_ref))
    print(f'   submission {submission.submission_id} ({submission.status.name})')
    if submission.status in NO_RESULT_STATUSES:
        reason = submission.reason_code.name if submission.reason_code else submission.status.name
        raise SystemExit(f'   terminal at the door: {reason}')

    print('3. quota')
    print_quota(client)

    print('4. poll until terminal')
    view = poll_until_terminal(client, submission.submission_id, timeout_s=timeout_s)
    if isinstance(view, ErroredSubmissionView):
        raise SystemExit(f'   failed: {view.reason_code.name if view.reason_code else "unknown"} — {view.reason}')
    # A terminal view that is not finished carries no result, so there is no score to report.
    if not isinstance(view, FinishedSubmissionView):
        raise SystemExit(f'   {view.status.name}, with no result to read')
    print(f'   primary {view.scores.primary}')
    print(f'   result  {view.artifacts.result}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--platform-url', default=None, help='a platform other than the default one')
    parser.add_argument('--alias', default='demo', help='the name a board displays you by')
    parser.add_argument('--eval', default='fake.smoke', help='the eval to run; the platform lists the ones it offers')
    parser.add_argument('--policy-image', default='org/policy:v1', help='the image the platform pulls and runs')
    parser.add_argument('--timeout', type=float, default=60.0, help='seconds to wait for a terminal status')
    args = parser.parse_args()

    credential = os.environ.get(CREDENTIAL_ENV)
    if not credential:
        raise SystemExit(f'set {CREDENTIAL_ENV} to the token the platform verifies you by')

    with PlatformClient(args.platform_url) as client:
        try:
            walkthrough(
                client,
                credential=credential,
                alias=args.alias,
                eval_ref=EvalRef(args.eval),
                policy_image=PolicyImage(args.policy_image),
                timeout_s=args.timeout,
            )
        except PlatformError as exc:
            offered = f'\nevals on offer: {", ".join(exc.evals)}' if exc.evals is not None else ''
            raise SystemExit(f'{exc.code.name}: {exc.message}{offered}') from exc


if __name__ == '__main__':
    main()
