"""What a caller sends: the POST bodies, and the query models the GET endpoints take.

Unknown fields are rejected, so a typo'd field is a 422 rather than a silently dropped input that
would change what the submission means.
"""

from __future__ import annotations

from typing import Self

from platform_client.boards import BoardRef
from platform_client.enums import CameraVantage, EndpointKind, Placement
from platform_client.evals import EvalRef
from platform_client.ids import RequestId, SubmissionId, TransactionKey
from platform_client.policy_images import PolicyImage
from platform_client.slug import Slugged
from platform_client.tasks import TaskRef
from pydantic import BaseModel, ConfigDict, Field, model_validator

_FORBID_EXTRA = ConfigDict(extra='forbid')


class RegisterRequest(BaseModel):
    """`users.register` — create-or-return, keyed on the external identity behind `credential`."""

    model_config = _FORBID_EXTRA

    credential: str
    alias: str | None = None
    rotate: bool = False


class SubmissionCreateRequest(BaseModel):
    """`submissions.create` — the run-defining fields, exactly as sent."""

    model_config = _FORBID_EXTRA

    # A `PolicyImage`, so a reference the registry could never resolve is refused in the caller's own
    # process instead of spending a round trip to learn it.
    policy_image: PolicyImage
    # The whole of what the caller chooses: the eval names its own embodiment, and the platform
    # answers an unknown name with the ones it offers.
    eval: EvalRef
    alias: str | None = None
    # A present key must be non-empty: an empty string is a client bug, not "no key", and accepting
    # it would silently drop the caller's dedup guarantee.
    transaction_key: TransactionKey | None = Field(default=None, min_length=1)


class CancelRequest(BaseModel):
    """`submissions.cancel`."""

    model_config = _FORBID_EXTRA

    id: SubmissionId


class SubmissionGetQuery(BaseModel):
    """`submissions.get` — the id travels in the query string, in its hex wire form."""

    model_config = _FORBID_EXTRA

    id: SubmissionId


class RankingsQuery(BaseModel):
    """`rankings.get` — one board by slug."""

    model_config = _FORBID_EXTRA

    board: BoardRef


def _require_unique_names(names: list[str], whose: str) -> None:
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f'{whose} names {", ".join(repeated)} more than once; each entry names one')


class EndpointAsk(BaseModel):
    """One policy a request runs, and where it comes from.

    A `remote` endpoint is an address the caller holds up, and its `url` may follow later. A
    `served` endpoint names the bring-up (`provider`) and what it serves (`spec`), and carries no
    `url`: the platform brings it up and records the address.
    """

    model_config = _FORBID_EXTRA

    name: str = Field(min_length=1)
    kind: Slugged[EndpointKind] = EndpointKind.remote
    url: str | None = Field(default=None, min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    spec: str | None = Field(default=None, min_length=1)

    @model_validator(mode='after')
    def _the_kind_carries_its_own_locator(self) -> Self:
        if self.kind is EndpointKind.served:
            if self.provider is None or self.spec is None:
                raise ValueError(f'served endpoint {self.name!r} names no provider or no spec')
            if self.url is not None:
                raise ValueError(
                    f'served endpoint {self.name!r} names a url; the platform records the address it serves at'
                )
        elif self.provider is not None or self.spec is not None:
            raise ValueError(
                f'remote endpoint {self.name!r} names a provider or a spec, which only a served endpoint carries'
            )
        return self

    @property
    def names_a_locator(self) -> bool:
        """Whether this entry says where its policy comes from, or only names one the request defines."""
        return self.kind is EndpointKind.served or self.url is not None


class SceneAsk(BaseModel):
    """How the rig is laid out for a task: each piece's side, and how steeply the external camera looks.

    A field left unset takes the nearest level that states it, then the task's own choices.
    `external_cameras` is keyed by the mount name the task gives each camera.
    """

    model_config = _FORBID_EXTRA

    tote_placement: Slugged[Placement] | None = None
    camera_vantage: Slugged[CameraVantage] | None = None
    external_cameras: dict[str, Slugged[Placement]] = Field(default_factory=dict)


class TaskAsk(BaseModel):
    """One task of a request, by its catalogue id, and what this request changes for it.

    A field left unset takes the request's own value. `endpoints`, when given, replaces the
    request's list for this task; an entry in it that names no locator names one of the request's
    endpoints by label.
    """

    model_config = _FORBID_EXTRA

    task_id: TaskRef
    episodes_per_endpoint: int | None = Field(default=None, ge=1)
    cap_per_episode_sec: int | None = Field(default=None, ge=1)
    policy_preset: str | None = Field(default=None, min_length=1)
    scene: SceneAsk | None = None
    endpoints: list[EndpointAsk] | None = Field(default=None, min_length=1)

    @model_validator(mode='after')
    def _each_endpoint_is_named_once(self) -> Self:
        _require_unique_names([entry.name for entry in self.endpoints or []], f'task {self.task_id!r}')
        return self


class RequestCreate(BaseModel):
    """`requests.create` — one round: the tasks, the endpoints each task runs, and the count per endpoint.

    A task level overrides the request's own values for itself. The request carries no client:
    the key names the customer, and the gateway reads the client off their grant.
    """

    model_config = _FORBID_EXTRA

    tasks: list[TaskAsk] = Field(min_length=1)
    endpoints: list[EndpointAsk] = Field(default_factory=list)
    episodes_per_endpoint: int = Field(ge=1)
    cap_per_episode_sec: int | None = Field(default=None, ge=1)
    policy_preset: str | None = Field(default=None, min_length=1)
    scene: SceneAsk | None = None
    # A short name for the request. The coordinator builds the request's own slug around it.
    slug: str | None = Field(default=None, min_length=1)
    # A present key must be non-empty, as on `submissions.create`.
    transaction_key: TransactionKey | None = Field(default=None, min_length=1)

    @model_validator(mode='after')
    def _each_task_appears_once(self) -> Self:
        _require_unique_names([task.task_id for task in self.tasks], 'the request')
        return self

    @model_validator(mode='after')
    def _each_endpoint_is_named_once(self) -> Self:
        _require_unique_names([entry.name for entry in self.endpoints], 'the request')
        return self

    @model_validator(mode='after')
    def _every_task_runs_on_a_defined_endpoint(self) -> Self:
        defined = {entry.name for entry in self.endpoints}
        for task in self.tasks:
            if task.endpoints is None and not self.endpoints:
                raise ValueError(
                    f'task {task.task_id!r} runs on no endpoint: the request defines none and the task names none'
                )
            unknown = sorted(
                entry.name for entry in task.endpoints or [] if not entry.names_a_locator and entry.name not in defined
            )
            if unknown:
                raise ValueError(
                    f'task {task.task_id!r} names {", ".join(unknown)}, which state no locator and name no endpoint '
                    'the request defines'
                )
        return self

    def task_endpoints(self, task: TaskAsk) -> list[EndpointAsk]:
        """The endpoints `task` runs on: its own list, else the request's."""
        return self.endpoints if task.endpoints is None else task.endpoints

    @property
    def episodes_total(self) -> int:
        """Every episode this request asks for: per task, its endpoints times its count per endpoint."""
        return sum(
            len(self.task_endpoints(task)) * (task.episodes_per_endpoint or self.episodes_per_endpoint)
            for task in self.tasks
        )


class RequestGetQuery(BaseModel):
    """`requests.get` — the id travels in the query string, in its hex wire form."""

    model_config = _FORBID_EXTRA

    id: RequestId


class RequestListQuery(BaseModel):
    """`requests.list` — the page after the last id seen. A `limit` above the gateway's cap is clamped to it."""

    model_config = _FORBID_EXTRA

    after: RequestId | None = None
    limit: int | None = Field(default=None, gt=0)
