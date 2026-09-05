"""The envelope every endpoint fails with, and the exception a client raises for it.

Agents parse `code`, humans read `message`. `details` is open-ended per error, with four fixed
keys: `reason_code` behind a caller-fault rejection, `quota` behind `quota_exceeded`, and `evals`
or `tasks` carrying the names on offer when the one asked for is not among them.
"""

from __future__ import annotations

from typing import Any, ClassVar

from platform_client.enums import ErrorCode, ReasonCode
from platform_client.evals import EvalRef
from platform_client.responses import QuotaLimit
from platform_client.slug import Slugged, members_by_slug
from platform_client.tasks import TaskRef
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

# The `details` keys the gateway writes and this module reads.
REASON_CODE_DETAIL = 'reason_code'
QUOTA_DETAIL = 'quota'
EVALS_DETAIL = 'evals'
TASKS_DETAIL = 'tasks'
# Behind a scene the task cannot be set up in: the mounts and the tote sides that task offers.
MOUNTS_DETAIL = 'mounts'
TOTE_SIDES_DETAIL = 'tote_sides'


class ApiErrorBody(BaseModel):
    code: Slugged[ErrorCode]
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorEnvelope(BaseModel):
    error: ApiErrorBody


class PlatformError(Exception):
    """A non-2xx response from the platform, carrying the parsed envelope."""

    def __init__(self, envelope: ErrorEnvelope, *, http_status: int) -> None:
        super().__init__(f'{envelope.error.code.name}: {envelope.error.message}')
        self.envelope = envelope
        self.http_status = http_status

    @property
    def code(self) -> ErrorCode:
        return self.envelope.error.code

    @property
    def message(self) -> str:
        return self.envelope.error.message

    @property
    def details(self) -> dict[str, Any]:
        return self.envelope.error.details

    @property
    def reason_code(self) -> ReasonCode | None:
        """The terminal reason behind a caller fault, when the failure carries one.

        `None` means the failure carries no reason. A reason this client cannot read is
        `ReasonCode.INVALID` instead, so a gateway sending a code from a newer taxonomy — or a
        malformed one — is distinguishable from one sending none at all.
        """
        if REASON_CODE_DETAIL not in self.details:
            return None
        raw = self.details[REASON_CODE_DETAIL]
        return members_by_slug(ReasonCode).get(raw, ReasonCode.INVALID) if isinstance(raw, str) else ReasonCode.INVALID

    @property
    def quota(self) -> QuotaLimit | None:
        """The rule that refused the request, when the failure carries one.

        Absent means absent. A present value that is not a rule is a gateway defect, so it raises
        rather than reading as no rule at all.
        """
        if QUOTA_DETAIL not in self.details:
            return None
        return QuotaLimit.model_validate(self.details[QUOTA_DETAIL])

    # A list is not a model, so it validates through an adapter. Built once, because building one
    # costs more than the validation it then does.
    _EVAL_LIST: ClassVar[TypeAdapter[list[EvalRef]]] = TypeAdapter(list[EvalRef])

    @property
    def evals(self) -> list[EvalRef] | None:
        """The evals this platform offers, when the failure is that the one asked for is not one.

        The set lives on the server, so a caller naming an eval it does not have learns the real
        ones from the refusal itself rather than from a list this client would have to keep current.
        A present value that is not a list of names raises: a short list is worse than none, since a
        caller would pick from it believing it whole.
        """
        if EVALS_DETAIL not in self.details:
            return None
        return self._EVAL_LIST.validate_python(self.details[EVALS_DETAIL])

    _TASK_LIST: ClassVar[TypeAdapter[list[TaskRef]]] = TypeAdapter(list[TaskRef])

    @property
    def tasks(self) -> list[TaskRef] | None:
        """The task ids the rollouts catalogue holds, when the failure is that the one asked for is not one.

        Absent means absent. A present value that is not a list of ids raises, for the reason
        `evals` gives.
        """
        if TASKS_DETAIL not in self.details:
            return None
        return self._TASK_LIST.validate_python(self.details[TASKS_DETAIL])

    @classmethod
    def from_payload(cls, http_status: int, payload: object) -> PlatformError:
        """Build from a decoded response body, synthesizing an envelope for anything unparseable.

        A proxy or load balancer between the caller and the gateway answers in its own shape, so an
        unparseable body must still raise the same exception rather than a decode error.
        """
        try:
            envelope = ErrorEnvelope.model_validate(payload)
        except ValidationError:
            envelope = ErrorEnvelope(
                error=ApiErrorBody(
                    code=ErrorCode.internal_error,
                    message=f'unparseable error response (HTTP {http_status})',
                    details={'body': payload},
                )
            )
        return cls(envelope, http_status=http_status)
