"""Submit a sample policy to the competition's eval and print what it scored.

    uv run positronic/cli/examples/nebius_competition/submit_sample.py \
        --policy-image=<registry>/<you>/policy@sha256:...

`uv run` builds the environment this needs from the checkout, so nothing has to be installed first.
The key comes from POSITRONIC_PLATFORM_API_KEY: a secret passed as an argument lands in your shell
history and in every process listing on the box.

Assumes a key you already hold; `../walkthrough.py` covers registration. Re-running with the same
`--transaction-key` returns the original submission rather than spending quota twice.

The command-line equivalent is `positronic eval run --eval=robolab.public_subset
--policy-image=...`, then `positronic eval status --submission-id=...`.
"""

from __future__ import annotations

import argparse
import os
import time

from platform_client.client import API_KEY_ENV, PlatformClient
from platform_client.enums import NO_RESULT_STATUSES, TERMINAL_STATUSES, ReasonCode
from platform_client.errors import PlatformError
from platform_client.evals import EvalRef
from platform_client.ids import ApiKey, SubmissionId, TransactionKey
from platform_client.policy_images import PolicyImage
from platform_client.requests import SubmissionCreateRequest
from platform_client.responses import FinishedSubmissionView, SubmissionCreateResponse, SubmissionView

# The eval names the embodiment it runs on, so it is the whole of what a submission chooses.
EVAL = EvalRef('robolab.public_subset')


def submit(
    client: PlatformClient, *, policy_image: PolicyImage, alias: str | None, transaction_key: TransactionKey | None
) -> SubmissionCreateResponse:
    """Create the submission, and report the exact image it was pinned to."""
    submission = client.create_submission(
        SubmissionCreateRequest(policy_image=policy_image, eval=EVAL, alias=alias, transaction_key=transaction_key)
    )
    print(f'submission {submission.submission_id} — {submission.status.name}')
    print(f'pinned image {submission.policy_image_digest} against eval {EVAL}')
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--platform-url', default=None, help='a platform other than the default one')
    parser.add_argument('--policy-image', required=True, help='a digest-pinned reference the platform can pull')
    parser.add_argument('--alias', default=None, help='a per-submission label; the board shows your user alias')
    parser.add_argument('--transaction-key', default=None, help='reuse it to retry without a second charge')
    parser.add_argument('--timeout', type=float, default=3600.0, help='seconds to wait for a terminal status')
    args = parser.parse_args()

    key = os.environ.get(API_KEY_ENV)
    if not key:
        raise SystemExit(f'set {API_KEY_ENV} to the key `positronic account register` printed')

    with PlatformClient(args.platform_url, api_key=ApiKey(key)) as client:
        try:
            submission = submit(
                client,
                policy_image=PolicyImage(args.policy_image),
                alias=args.alias,
                transaction_key=TransactionKey(args.transaction_key) if args.transaction_key else None,
            )
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
