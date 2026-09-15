"""Drive one submission from registration to a board, against any platform.

    uv run positronic/cli/examples/walkthrough.py --eval=<name> --policy-image=<reference>

`uv run` builds the environment this needs from the checkout, so nothing has to be installed first.
The credential is read from the environment rather than taken as an argument: a command line is
readable by every process on the box and lands in shell history.

`--eval` names the eval to run. The platform owns the list: with no `--eval`, the script prints the
public boards and the eval each one ranks, and stops. A name the platform does not offer is refused,
and the refusal names the evals on offer.

The key comes from POSITRONIC_PLATFORM_API_KEY, which `platform-register` prints. With no key, the
script registers with the GitHub token in POSITRONIC_PLATFORM_CREDENTIAL, which the platform's own
OAuth app must have minted.

Every call goes through `PlatformClient`, so each response is a typed model rather than a dict. The
same flow from the command line is `positronic account register`, `positronic eval run` and
`positronic eval status`.
"""

from __future__ import annotations

import argparse
import os
import time

import httpx
from platform_client.client import API_KEY_ENV, CREDENTIAL_ENV, PlatformClient
from platform_client.enums import NO_RESULT_STATUSES, TERMINAL_STATUSES, KeyStatus
from platform_client.errors import PlatformError
from platform_client.eval_plan import plan_of_image
from platform_client.evals import EvalRef
from platform_client.ids import SubmissionId
from platform_client.policy_images import PolicyImage
from platform_client.requests import RegisterRequest
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


def anonymous_client(platform_url: str | None = None, *, client: httpx.Client | None = None) -> PlatformClient:
    """A client that sends no key, whatever the environment holds: a public board is readable by anyone."""
    anonymous = PlatformClient(platform_url, client=client)
    anonymous.api_key = None
    return anonymous


def walkthrough(
    client: PlatformClient,
    *,
    credential: str | None,
    alias: str,
    eval_ref: EvalRef,
    policy_image: PolicyImage,
    timeout_s: float,
) -> None:
    print('1. register')
    if client.api_key is not None:
        print(f'   key from {API_KEY_ENV}; nothing to register')
    elif credential is not None:
        authenticate(client, credential=credential, alias=alias)
    else:
        raise SystemExit(f'   set {API_KEY_ENV} to the key `platform-register` prints')

    print('2. submit')
    # The eval is the whole of the choice: it names the embodiment its tasks run on, and asking for
    # one the platform does not offer comes back with the names it does, under `PlatformError.evals`.
    submission = client.create_submission(plan_of_image(policy_image, eval_ref))
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

    print('5. board')
    with anonymous_client(client.base_url) as public:
        boards = [board for board in public.list_boards().boards if board.eval == eval_ref]
        if not boards:
            print(f'   no public board ranks {eval_ref}')
        for board in boards:
            print(f'   {board.board}')
            for row in public.rankings(board=board.board).rankings:
                print(f'   {row.rank:>4}  {row.display_name}#{row.tag}  {row.scores.primary}  {row.submission_id}')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--platform-url', default=None, help='a platform other than the default one')
    parser.add_argument('--alias', default='demo', help='the name a board displays you by')
    parser.add_argument(
        '--eval', default=None, help='the eval to run; with none, the public boards name the ones on offer'
    )
    parser.add_argument(
        '--policy-image', default=None, help='the image the platform pulls and runs; needed with --eval'
    )
    parser.add_argument('--timeout', type=float, default=60.0, help='seconds to wait for a terminal status')
    args = parser.parse_args(argv)
    if (args.eval is None) != (args.policy_image is None):
        parser.error('--eval and --policy-image go together: pass both, or neither to list the boards')
    # Every value the wire types refuse — a name, a reference, a platform URL — is refused here,
    # before any request.
    try:
        eval_ref = EvalRef(args.eval) if args.eval is not None else None
        policy_image = PolicyImage(args.policy_image) if args.policy_image is not None else None
        client = anonymous_client(args.platform_url) if eval_ref is None else PlatformClient(args.platform_url)
    except ValueError as exc:
        parser.error(str(exc))

    if eval_ref is None or policy_image is None:
        with client as public:
            print('pass --eval=<name>; the public boards rank these evals:')
            for board in public.list_boards().boards:
                print(f'   {board.board}: ranks {board.eval} by {board.primary_metric}')
        return

    with client:
        try:
            walkthrough(
                client,
                credential=os.environ.get(CREDENTIAL_ENV) or None,
                alias=args.alias,
                eval_ref=eval_ref,
                policy_image=policy_image,
                timeout_s=args.timeout,
            )
        except PlatformError as exc:
            offered = f'\nevals on offer: {", ".join(exc.evals)}' if exc.evals is not None else ''
            raise SystemExit(f'{exc.code.name}: {exc.message}{offered}') from exc


if __name__ == '__main__':
    main()
