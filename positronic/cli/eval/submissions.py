"""`positronic eval status|list|cancel|catalog` — what the runs you sent to the platform are doing."""

import configuronic as cfn
from platform_client.enums import ErrorCode
from platform_client.errors import PlatformError
from platform_client.ids import PlanId, SubmissionId
from platform_client.requests import CancelRequest
from platform_client.responses import ID_FIELD, PLAN_ID_FIELD, STATUS_FIELD, PlanView, SubmissionView

from positronic.cli.account.gateway import gateway, parse_id, refusing_bad_input

# What `status` prints on its header line, so the body below it does not repeat them. Field names
# rather than literals, so a model rename cannot leave this excluding a field that no longer exists.
_HEADER_FIELDS = frozenset({ID_FIELD, PLAN_ID_FIELD, STATUS_FIELD})


def _both_ids(token: object) -> tuple[SubmissionId, PlanId]:
    """One id read as each kind. Both are bare hex on the wire, so an id does not say which it names."""
    return parse_id(token, SubmissionId), parse_id(token, PlanId)


def _show(header: str, view: SubmissionView | PlanView) -> None:
    print(header)
    for name, value in view.model_dump(mode='json', exclude=set(_HEADER_FIELDS)).items():
        print(f'  {name}: {value}')


@cfn.config()
def status(id: str, platform_url: str | None = None):
    """Report what one submission or one eval plan is doing, and what it produced once it is done.

    The id says which of the two it names, so this reads the submission first and the plan where the
    platform knows no such submission.
    """
    submission_id, plan_id = _both_ids(id)
    with gateway(platform_url) as client:
        try:
            submission = client.get_submission(submission_id)
        except PlatformError as exc:
            if exc.code is not ErrorCode.not_found:
                raise
            plan = client.get_eval(plan_id)
            _show(f'plan {plan.plan_id} {plan.status.name}', plan)
            return
        _show(f'submission {submission.id} {submission.status.name}', submission)


@cfn.config()
def list_runs(platform_url: str | None = None):
    """List the submissions and the eval plans this API key can see."""
    with gateway(platform_url) as client:
        submissions = client.list_submissions()
        plans = client.list_evals()
    for row in submissions.submissions:
        alias = f' {row.alias}' if row.alias else ''
        print(f'submission {row.id} {row.received_at:%Y-%m-%d %H:%M} {row.status.name} {row.eval}{alias}')
    for row in plans.plans:
        print(f'plan {row.plan_id} {row.status.name} {row.episodes.done}/{row.episodes.total} episodes')


@cfn.config()
def cancel(id: str, platform_url: str | None = None):
    """Cancel a submission that has not reached a terminal status."""
    submission_id, plan_id = _both_ids(id)
    with refusing_bad_input():
        request = CancelRequest(id=submission_id)
    with gateway(platform_url) as client:
        try:
            result = client.cancel_submission(request)
        except PlatformError as exc:
            if exc.code is not ErrorCode.not_found:
                raise
            client.get_eval(plan_id)
            raise SystemExit(f'{plan_id} is an eval plan, and the platform cancels no plan yet') from exc
        print(f'{result.status.name}, quota {"refunded" if result.refunded else "charged"}')


@cfn.config()
def catalog(platform_url: str | None = None):
    """Print what this key may name: the evals `eval run` takes by name, and the tasks a plan composes."""
    with gateway(platform_url) as client:
        evals = client.catalog_evals()
        tasks = client.catalog_tasks()
    print(evals.model_dump_json(indent=2))
    print(tasks.model_dump_json(indent=2))
