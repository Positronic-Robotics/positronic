"""`positronic eval status|list|cancel|catalog` — what the runs you sent to the platform are doing."""

import sys

import configuronic as cfn
from platform_client.catalog import EvalListResponse, TaskListResponse, TaskSummary
from platform_client.client import PlatformClient
from platform_client.enums import ErrorCode
from platform_client.errors import PlatformError
from platform_client.ids import SubmissionId
from platform_client.requests import CancelRequest
from platform_client.responses import ID_FIELD, STATUS_FIELD, SubmissionListResponse, SubmissionListRow, SubmissionView

from positronic.cli.account.gateway import gateway, parse_id, refusing_bad_input

# What `status` prints on its header line, so the body below it does not repeat them. Field names
# rather than literals, so a model rename cannot leave this excluding a field that no longer exists.
_HEADER_FIELDS = frozenset({ID_FIELD, STATUS_FIELD})


@cfn.config()
def status(id: str, platform_url: str | None = None, json: bool = False):
    """Report what one run is doing, and what it produced once it is done. `--json` prints the view as JSON."""
    with gateway(platform_url, refusal_as_json=json) as client:
        view: SubmissionView = client.get_submission(parse_id(id, SubmissionId))
    if json:
        print(view.model_dump_json(indent=2))
        return
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
def list_submissions(platform_url: str | None = None, json: bool = False):
    """List the runs this API key can see, oldest first. `--json` prints every page as one JSON page."""
    with gateway(platform_url, refusal_as_json=json) as client:
        rows = _every_submission(client)
    if json:
        print(SubmissionListResponse(submissions=rows).model_dump_json(indent=2))
        return
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


class CatalogView(EvalListResponse):
    """What `catalog --json` prints: the evals, and the tasks, or null where the key may compose no plan."""

    tasks: list[TaskSummary] | None


def _tasks_or_none(client: PlatformClient) -> TaskListResponse | None:
    """`catalog.tasks`, or None where the key has no customer grant. The refusal goes to stderr."""
    try:
        return client.catalog_tasks()
    except PlatformError as exc:
        if exc.code is not ErrorCode.forbidden:
            raise
        print(f'tasks: {exc.code.name}: {exc.message}', file=sys.stderr)
        return None


@cfn.config()
def catalog(platform_url: str | None = None, json: bool = False):
    """Print what this key may name: the evals a plan names, and the tasks a plan composes.

    A key with no customer grant composes no plan, so `catalog.tasks` refuses it; the evals print, and
    the refusal goes to stderr. Both print as JSON. `--json` prints them as one `CatalogView`, and a
    refusal of the evals as JSON too.
    """
    with gateway(platform_url, refusal_as_json=json) as client:
        evals = client.catalog_evals()
        tasks = _tasks_or_none(client)
    if json:
        view = CatalogView(evals=evals.evals, tasks=None if tasks is None else tasks.tasks)
        print(view.model_dump_json(indent=2))
        return
    print(evals.model_dump_json(indent=2))
    if tasks is not None:
        print(tasks.model_dump_json(indent=2))
