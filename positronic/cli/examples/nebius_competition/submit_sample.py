"""Submit a sample policy to one eval and print what it scored.

    uv run positronic/cli/examples/nebius_competition/submit_sample.py \
        --eval=<name> --policy-image=<registry>/<you>/policy@sha256:... --policy-wire=websocket

The platform owns the list of evals. `standings.py` prints the public leaderboards and the eval
each one ranks; a submission with a name the platform does not offer is refused, and the refusal
names the evals on offer.

`uv run` builds the environment this needs from the checkout, so nothing has to be installed first.
The key comes from POSITRONIC_PLATFORM_API_KEY: a secret passed as an argument lands in your shell
history and in every process listing on the box.

Assumes a key you already hold; `../walkthrough.py` covers registration. Re-running with the same
`--transaction-key` returns the original submission rather than spending quota twice.

The command-line equivalent is `positronic eval run --eval=<name> --policy-image=... --policy-wire=...`, then
`positronic eval status --id=...`.
"""

from __future__ import annotations

import argparse
import os
import time

from platform_client.client import API_KEY_ENV, PlatformClient
from platform_client.enums import NO_RESULT_STATUSES, TERMINAL_STATUSES, ReasonCode, Wire
from platform_client.errors import PlatformError
from platform_client.eval_plan import EvalPlan, plan_of_image
from platform_client.evals import EvalRef
from platform_client.ids import ApiKey, SubmissionId, TransactionKey
from platform_client.policy_images import PolicyImage
from platform_client.responses import FinishedSubmissionView, SubmissionCreateResponse, SubmissionView
from platform_client.slug import members_by_slug


def submit(client: PlatformClient, plan: EvalPlan) -> SubmissionCreateResponse:
    """Create the submission, and report the exact image it was pinned to."""
    submission = client.create_submission(plan)
    print(f'submission {submission.submission_id} — {submission.status.name}')
    print(f'pinned image {submission.policy_image_digest} against eval {plan.eval}')
    return submission


def poll_until_terminal(
    client: PlatformClient, submission_id: SubmissionId, *, timeout_s: float, poll_s: float = 5.0
) -> SubmissionView:
    """Poll one submission until it is decided, or give up and say how to follow it by hand."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        view = client.get_submission(submission_id)
        if view.status in TERMINAL_STATUSES:
            return view
        time.sleep(poll_s)
    raise SystemExit(f'still running after {timeout_s:.0f}s — `positronic eval status` follows it')


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--platform-url', default=None, help='a platform other than the default one')
    parser.add_argument('--eval', required=True, help='the eval to run; `standings.py` lists the ones with a board')
    parser.add_argument('--policy-image', required=True, help='a digest-pinned reference the platform can pull')
    parser.add_argument(
        '--policy-wire', required=True, choices=list(members_by_slug(Wire)), help='the wire the image serves'
    )
    parser.add_argument('--alias', default=None, help='a per-submission label; the board shows your user alias')
    parser.add_argument('--transaction-key', default=None, help='reuse it to retry without a second charge')
    parser.add_argument('--timeout', type=float, default=3600.0, help='seconds to wait for a terminal status')
    args = parser.parse_args(argv)
    key = os.environ.get(API_KEY_ENV)
    if not key:
        # `positronic account register` saves the key in its record, which this script does not read.
        # `platform-register` prints the export line, so it is the one that helps here.
        raise SystemExit(f'set {API_KEY_ENV} to the key `platform-register` prints')
    # Every value the wire types refuse — a name, a reference, a transaction key, a platform URL — is
    # refused here, before any request. A pydantic `ValidationError` is a `ValueError`.
    try:
        transaction_key = TransactionKey(args.transaction_key) if args.transaction_key is not None else None
        # The eval names the embodiment it runs on, so it is the whole of what a submission chooses.
        plan = plan_of_image(
            PolicyImage(args.policy_image),
            EvalRef(args.eval),
            members_by_slug(Wire)[args.policy_wire],
            alias=args.alias,
            transaction_key=transaction_key,
        )
        client = PlatformClient(args.platform_url, api_key=ApiKey(key))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    with client:
        try:
            submission = submit(client, plan)
            if submission.status in NO_RESULT_STATUSES:
                reason = submission.reason_code.name if submission.reason_code else submission.status.name
                if submission.reason_code is ReasonCode.image_unpullable:
                    reason += ' — check the reference and its visibility'
                raise SystemExit(f'terminal at the door: {reason}')
            view = poll_until_terminal(client, submission.submission_id, timeout_s=args.timeout)
            # A terminal view that is not finished carries no result, so there is no score to report.
            if not isinstance(view, FinishedSubmissionView):
                raise SystemExit(f'finished as {view.status.name}, with no result to read')
            print(f'finished as {view.status.name}')
            print(f'primary {view.scores.primary}')
        except PlatformError as exc:
            offered = f'\nevals on offer: {", ".join(exc.evals)}' if exc.evals is not None else ''
            raise SystemExit(f'{exc.code.name}: {exc.message}{offered}') from exc


if __name__ == '__main__':
    main()
