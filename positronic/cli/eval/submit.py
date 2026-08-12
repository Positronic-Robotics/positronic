"""The half of `positronic eval run` that hands the run to the platform instead of running it here.

Not a command of its own: running an eval is one act, and where it runs is an argument to it.
"""

from platform_client.enums import SubmissionStatus
from platform_client.evals import EvalRef
from platform_client.ids import TransactionKey
from platform_client.images import ImageRef
from platform_client.requests import SubmissionCreateRequest

from positronic.cli.account.gateway import gateway


def submit(
    eval_name: str,
    policy_image: str,
    *,
    alias: str | None = None,
    transaction_key: str | None = None,
    platform_url: str | None = None,
) -> None:
    """Submit one policy image against one eval, and print what came back.

    The eval names the embodiment it runs on, so it is the whole of the choice; naming one the
    platform does not offer answers with the ones it does. An image pinned by digest runs the bytes
    you tested, while a mutable tag is resolved at submission time. Repeating a submission under one
    `transaction_key` returns the original instead of spending another day's quota.
    """
    request = SubmissionCreateRequest(
        policy_image=ImageRef(policy_image),
        eval=EvalRef(eval_name),
        alias=alias,
        transaction_key=TransactionKey(transaction_key) if transaction_key is not None else None,
    )
    with gateway(platform_url) as client:
        created = client.create_submission(request)
    print(f'submission {created.submission_id} ({created.status.name})')
    if created.policy_image_digest is not None:
        print(f'digest {created.policy_image_digest}')
    # Terminal at the door, and charged. A zero exit would let a script treat a run that never
    # happened as one that did.
    if created.status is SubmissionStatus.errored:
        reason = created.reason_code.name if created.reason_code is not None else 'unknown'
        raise SystemExit(f'rejected: {reason}')
