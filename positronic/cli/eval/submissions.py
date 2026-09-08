"""`positronic eval status|list|cancel|catalog` — what the runs you sent to the platform are doing."""

import sys

import configuronic as cfn
from platform_client.client import PlatformClient
from platform_client.enums import ErrorCode
from platform_client.errors import PlatformError
from platform_client.ids import SubmissionId
from platform_client.requests import CancelRequest
from platform_client.responses import ID_FIELD, STATUS_FIELD, SubmissionListRow, SubmissionView

from positronic.cli.account.gateway import gateway, parse_id, refusing_bad_input

# What `status` prints on its header line, so the body below it does not repeat them. Field names
# rather than literals, so a model rename cannot leave this excluding a field that no longer exists.
_HEADER_FIELDS = frozenset({ID_FIELD, STATUS_FIELD})


@cfn.config()
def status(id: str, platform_url: str | None = None):
    """Report what one run is doing, and what it produced once it is done."""
    with gateway(platform_url) as client:
        view: SubmissionView = client.get_submission(parse_id(id, SubmissionId))
    print(f'submission {view.id} {view.status.name}')
    for name, value in view.model_dump(mode='json', exclude=set(_HEADER_FIELDS)).items():
        print(f'  {name}: {value}')


def _every_submission(client: PlatformClient) -> list[SubmissionListRow]:
    """Every page of `submissions.list`, oldest first."""
    rows: list[SubmissionListRow] = []
    cursor: SubmissionId | None = None
    while True:
        page = client.list_submissions(after=cursor)
        rows += page.submissions
        if page.next is None:
            return rows
        cursor = page.next


@cfn.config()
def list_submissions(platform_url: str | None = None):
    """List the runs this API key can see, oldest first."""
    with gateway(platform_url) as client:
        rows = _every_submission(client)
    for row in rows:
        named = f' {row.eval}' if row.eval else ''
        alias = f' {row.alias}' if row.alias else ''
        # A run the platform executes itself reports no episode count, so the tail is the rig's.
        episodes = f' {row.episodes.done}/{row.episodes.total} episodes' if row.episodes.total else ''
        print(f'submission {row.id} {row.received_at:%Y-%m-%d %H:%M} {row.status.name}{named}{alias}{episodes}')


@cfn.config()
def cancel(id: str, platform_url: str | None = None):
    """Cancel a run that has not reached a terminal status."""
    with refusing_bad_input():
        request = CancelRequest(id=parse_id(id, SubmissionId))
    with gateway(platform_url) as client:
        result = client.cancel_submission(request)
    print(f'{result.status.name}, quota {"refunded" if result.refunded else "charged"}')


@cfn.config()
def catalog(platform_url: str | None = None):
    """Print what this key may name: the evals a plan names, and the tasks a plan composes.

    A key with no customer grant composes no plan, so `catalog.tasks` refuses it; the evals print, and
    the refusal goes to stderr.
    """
    with gateway(platform_url) as client:
        evals = client.catalog_evals()
        try:
            tasks = client.catalog_tasks().model_dump_json(indent=2)
        except PlatformError as exc:
            if exc.code is not ErrorCode.forbidden:
                raise
            tasks = None
            print(f'tasks: {exc.code.name}: {exc.message}', file=sys.stderr)
    print(evals.model_dump_json(indent=2))
    if tasks is not None:
        print(tasks)
