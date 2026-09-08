"""`EvalPlan`: an eval to run — the tasks, the policies that run them, and the count per policy per task.

The plan is the one definition of that shape. The rollouts coordinator's own record extends it with
its bookkeeping (id, client, requester, channel, approval, runs) and adds no field of its own to
the ask. Unknown fields are rejected: a typo'd field is a 422.
"""

from __future__ import annotations

from typing import Self

import httpx
from platform_client.enums import CameraVantage, EndpointKind, Placement
from platform_client.ids import TransactionKey
from platform_client.slug import Slugged
from platform_client.tasks import TaskRef
from pydantic import BaseModel, ConfigDict, Field, model_validator

_FORBID_EXTRA = ConfigDict(extra='forbid')

# The properties an endpoint may not state: a run lays out one scene, one cap and one preset for
# its whole sample, so they are per task, and an endpoint overrides only its own count.
_PER_TASK_PROPERTIES = ('cap_per_episode_sec', 'policy_preset', 'camera_vantage', 'tote_placement', 'clutter')


def _require_unique_names(names: list[str], whose: str) -> None:
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f'{whose} names {", ".join(repeated)} more than once; each entry names one')


class Clutter(BaseModel):
    """How much of the rest of the object kit a run draws onto the table.

    The draw picks a count between the two bounds per run; at most `large_cap` large objects and
    `medium_cap` medium ones, and the rest small.
    """

    model_config = _FORBID_EXTRA

    count_min: int = Field(default=4, ge=0)
    count_max: int = Field(default=8, ge=0)
    large_cap: int = Field(default=1, ge=0)
    medium_cap: int = Field(default=2, ge=0)

    @model_validator(mode='after')
    def _bounds_in_order(self) -> Self:
        if self.count_max < self.count_min:
            raise ValueError(f'clutter draws between {self.count_min} and {self.count_max}, which is no range')
        return self


class Cascade(BaseModel):
    """The properties any level of a plan may state: the plan, one of its tasks, one endpoint.

    A field left unset takes the nearest level above that states it, and below the plan the task's
    own catalogue entry. A stated value pins it for this level and everything under it;
    `random` draws a side here whatever a level above says.
    """

    model_config = _FORBID_EXTRA

    # Episodes each endpoint of each task under this level takes.
    episodes_per_endpoint: int | None = Field(default=None, ge=1)
    cap_per_episode_sec: int | None = Field(default=None, ge=1)
    policy_preset: str | None = Field(default=None, min_length=1)
    tote_placement: Slugged[Placement] | None = None
    camera_vantage: Slugged[CameraVantage] | None = None
    # Per external camera the task defines, keyed by the mount name the task gives it.
    external_cameras: dict[str, Slugged[Placement]] = Field(default_factory=dict)
    clutter: Clutter | None = None


class Endpoint(Cascade):
    """One policy to run, and where it comes from.

    A `remote` endpoint is an address the caller holds up. A `served` endpoint names the bring-up
    (`provider`) and what it serves (`spec`), and carries no `url`: the platform brings it up and
    records the address. A bare label on a task names one of the plan's endpoints.
    """

    name: str = Field(min_length=1)
    kind: Slugged[EndpointKind] = EndpointKind.remote
    url: str | None = Field(default=None, min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    spec: str | None = Field(default=None, min_length=1)

    @model_validator(mode='before')
    @classmethod
    def _accept_bare_label(cls, value: object) -> object:
        return {'name': value} if isinstance(value, str) else value

    @model_validator(mode='after')
    def _the_kind_carries_its_own_locator(self) -> Self:
        if self.url is not None and not httpx.URL(self.url).is_absolute_url:
            # The scheme is the platform's to judge; an address with no host reaches nothing.
            raise ValueError(f'endpoint {self.name!r} names {self.url!r}, which has no host: give an absolute URL')
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

    @model_validator(mode='after')
    def _overrides_only_the_count(self) -> Self:
        stated = [name for name in _PER_TASK_PROPERTIES if getattr(self, name) is not None]
        if self.external_cameras:
            stated.append('external_cameras')
        if stated:
            raise ValueError(
                f'endpoint {self.name!r} states {", ".join(stated)}, which are per-task properties: '
                'an endpoint overrides only episodes_per_endpoint'
            )
        return self

    @property
    def names_a_locator(self) -> bool:
        """Whether this entry says where its policy comes from, or only names one the plan defines."""
        return self.kind is EndpointKind.served or self.url is not None


class TaskNode(Cascade):
    """One task of a plan, by its catalogue id, and what this plan changes for it.

    `endpoints`, when given, replaces the plan's list for this task; an entry in it that names no
    locator names one of the plan's endpoints by label. A bare id states nothing for the task.
    """

    task_id: TaskRef
    endpoints: list[Endpoint] | None = Field(default=None, min_length=1)

    @model_validator(mode='before')
    @classmethod
    def _accept_bare_id(cls, value: object) -> object:
        return {'task_id': value} if isinstance(value, str) else value

    @model_validator(mode='after')
    def _each_endpoint_is_named_once(self) -> Self:
        _require_unique_names([entry.name for entry in self.endpoints or []], f'task {self.task_id!r}')
        return self


class EvalPlan(Cascade):
    """`evals.run` — one eval to run: the tasks, the endpoints each task runs, and the count per endpoint.

    The plan states the count once; a task overrides it for itself, and an endpoint for itself. The
    plan carries no client: the key names the customer, and the gateway reads the client off their
    grant. A named eval the platform offers is a plan the registry holds; this is the one a caller
    composes.
    """

    tasks: list[TaskNode] = Field(min_length=1)
    endpoints: list[Endpoint] = Field(default_factory=list)
    # The checksum: stated, it must equal the sum over the resolved leaves; absent, the platform fills it.
    episodes_total: int | None = Field(default=None, ge=1)
    # A ceiling every leaf's window sits under. A cap typo costs real minutes at the rig.
    max_cap_per_episode_sec: int | None = Field(default=None, ge=1)
    # A present key must be non-empty: an empty string is a client bug.
    transaction_key: TransactionKey | None = Field(default=None, min_length=1)

    @model_validator(mode='after')
    def _states_a_count(self) -> Self:
        if self.episodes_per_endpoint is None:
            raise ValueError('a plan states episodes_per_endpoint; a task or an endpoint overrides it')
        return self

    @model_validator(mode='after')
    def _each_task_appears_once(self) -> Self:
        _require_unique_names([task.task_id for task in self.tasks], 'the plan')
        return self

    @model_validator(mode='after')
    def _each_endpoint_is_named_once(self) -> Self:
        _require_unique_names([entry.name for entry in self.endpoints], 'the plan')
        return self

    @model_validator(mode='after')
    def _every_task_runs_on_a_defined_endpoint(self) -> Self:
        defined = {entry.name for entry in self.endpoints}
        for task in self.tasks:
            if task.endpoints is None and not self.endpoints:
                raise ValueError(
                    f'task {task.task_id!r} runs on no endpoint: the plan defines none and the task names none'
                )
            unknown = sorted(
                entry.name for entry in task.endpoints or [] if not entry.names_a_locator and entry.name not in defined
            )
            if unknown:
                raise ValueError(
                    f'task {task.task_id!r} names {", ".join(unknown)}, which state no locator and name no endpoint '
                    'the plan defines'
                )
        return self

    @model_validator(mode='after')
    def _every_cap_sits_under_the_ceiling(self) -> Self:
        ceiling = self.max_cap_per_episode_sec
        if ceiling is None:
            return self
        for task in self.tasks:
            cap = task.cap_per_episode_sec if task.cap_per_episode_sec is not None else self.cap_per_episode_sec
            if cap is not None and cap > ceiling:
                raise ValueError(
                    f'task {task.task_id!r} takes {cap} s per episode, over the plan ceiling of {ceiling} s'
                )
        return self

    @model_validator(mode='after')
    def _the_checksum_matches(self) -> Self:
        if self.episodes_total is not None and self.episodes_total != self.resolved_episodes_total:
            raise ValueError(
                f'episodes_total states {self.episodes_total}, and the leaves sum to {self.resolved_episodes_total}'
            )
        return self

    def task_endpoints(self, task: TaskNode) -> list[Endpoint]:
        """The endpoints `task` runs on: its own list, else the plan's."""
        return self.endpoints if task.endpoints is None else task.endpoints

    def episodes_on(self, task: TaskNode, entry: Endpoint) -> int:
        """The episodes `entry` takes for `task`.

        Its own count, else its definition's, else the task's, else the plan's. The definition is
        the plan's entry of that name, for an entry that names no locator.
        """
        definition = None
        if not entry.names_a_locator:
            definition = next((defined for defined in self.endpoints if defined.name == entry.name), None)
        stated = (level.episodes_per_endpoint for level in (entry, definition, task) if level is not None)
        count = next((count for count in stated if count is not None), self.episodes_per_endpoint)
        assert count is not None  # `_states_a_count` refused a plan with none
        return count

    @property
    def resolved_episodes_total(self) -> int:
        """Every episode this plan asks for: per task, the count of each endpoint it runs on."""
        return sum(self.episodes_on(task, entry) for task in self.tasks for entry in self.task_endpoints(task))
