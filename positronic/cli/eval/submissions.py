"""`positronic eval status|list|cancel` — what the runs you sent to the platform are doing."""

import configuronic as cfn
from platform_client.requests import CancelRequest
from platform_client.responses import ID_FIELD, STATUS_FIELD

from positronic.cli.account.gateway import gateway, parse_submission_id

# What `status` prints on its header line, so the body below it does not repeat them. Field names
# rather than literals, so a model rename cannot leave this excluding a field that no longer exists.
_HEADER_FIELDS = frozenset({ID_FIELD, STATUS_FIELD})


@cfn.config()
def status(submission_id: str, platform_url: str | None = None):
    """Report what one submission is doing, and what it produced once it is done."""
    wanted = parse_submission_id(submission_id)
    with gateway(platform_url) as client:
        view = client.get_submission(wanted)
    print(f'{view.id} {view.status.name}')
    for name, value in view.model_dump(mode='json', exclude=set(_HEADER_FIELDS)).items():
        print(f'  {name}: {value}')


@cfn.config()
def list_submissions(platform_url: str | None = None):
    """List the submissions this API key can see."""
    with gateway(platform_url) as client:
        response = client.list_submissions()
    for row in response.submissions:
        alias = f' {row.alias}' if row.alias else ''
        print(f'{row.id} {row.received_at:%Y-%m-%d %H:%M} {row.status.name} {row.eval}{alias}')


@cfn.config()
def cancel(submission_id: str, platform_url: str | None = None):
    """Cancel a submission that has not reached a terminal status."""
    request = CancelRequest(id=parse_submission_id(submission_id))
    with gateway(platform_url) as client:
        result = client.cancel_submission(request)
    print(f'{result.status.name}, quota {"refunded" if result.refunded else "charged"}')
