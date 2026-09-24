"""The part of `positronic eval run` that hands the run to the platform.

Not a command of its own: running an eval is one act, and where it runs is an argument to it.
"""

from pathlib import Path

from platform_client.enums import NO_RESULT_STATUSES
from platform_client.eval_plan import RegistryCredential, password_from_file, plan_of_image
from platform_client.evals import EvalRef
from platform_client.ids import OrgSlug, TransactionKey
from platform_client.policy_images import PolicyImage
from platform_client.responses import SubmissionCreateResponse
from pydantic import SecretStr

from positronic.cli.account.gateway import gateway, refusing_bad_input


def _credential(username: str | None, password_file: str | None) -> RegistryCredential | None:
    """The credential that opens a private registry, from the two flags that state it."""
    if (username is None) != (password_file is None):
        raise SystemExit('--registry-username and --registry-password-file state one credential: pass both')
    if username is None or password_file is None:
        return None
    # configuronic hands `1` or `[]` through as an int or a list, and `Path` raises on either.
    if not isinstance(password_file, str):
        raise SystemExit(f'--registry-password-file={password_file!r} names no file: pass its path')
    try:
        password = password_from_file(Path(password_file))
    except ValueError as exc:
        raise SystemExit(f'--registry-password-file names no readable password: {exc}') from exc
    return RegistryCredential(username=username, password=SecretStr(password))


def submit(
    eval_name: str,
    policy_image: str,
    *,
    alias: str | None = None,
    transaction_key: str | None = None,
    platform_url: str | None = None,
    org: str | None = None,
    registry_username: str | None = None,
    registry_password_file: str | None = None,
) -> SubmissionCreateResponse:
    """Submit one policy image against one eval, print what came back, and return it.

    A submission is a plan with one image endpoint: the eval names the tasks and the embodiment
    that runs them, and the catalogue expands that name. Naming one the platform does not offer
    answers with the ones it does. An image pinned by digest runs the bytes you tested, while a
    mutable tag is resolved at submission time. Repeating a submission under one `transaction_key`
    returns the original instead of spending another day's quota. `org` runs it as a private
    request for that organisation instead of a `nebius_competition` one. An image the platform
    cannot pull anonymously takes a credential: `registry_username` and `registry_password_file`,
    the file the password is in.
    """
    with refusing_bad_input():
        plan = plan_of_image(
            PolicyImage(policy_image),
            EvalRef(eval_name),
            alias=alias,
            transaction_key=TransactionKey(transaction_key) if transaction_key is not None else None,
            credential=_credential(registry_username, registry_password_file),
            org=OrgSlug(org) if org is not None else None,
        )
    with gateway(platform_url) as client:
        submission = client.create_submission(plan)
    print(f'submission {submission.submission_id} ({submission.status.name})')
    if submission.policy_image_digest is not None:
        print(f'digest {submission.policy_image_digest}')
    # These terminal statuses can never carry a result, so the submission fails rather than returns.
    if submission.status in NO_RESULT_STATUSES:
        reason = submission.reason_code.name if submission.reason_code is not None else submission.status.name
        raise SystemExit(f'rejected: {reason}')
    return submission
