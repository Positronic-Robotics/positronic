"""`EvalPlan`: an eval to run — the tasks, the policies that run them, and the count per policy per task.

Unknown fields are rejected, so a misspelled field is a 422.
"""

from __future__ import annotations

from collections import Counter
from typing import Self

import httpx
from platform_client.enums import CameraVantage, EndpointKind, Placement
from platform_client.evals import EvalRef
from platform_client.ids import TransactionKey
from platform_client.policy_images import PolicyImage
from platform_client.slug import Slugged
from platform_client.tasks import TaskRef
from pydantic import BaseModel, ConfigDict, Field, model_validator

_FORBID_EXTRA = ConfigDict(extra='forbid')


def _require_unique_names(names: list[str], whose: str) -> None:
    repeated = sorted(name for name, seen in Counter(names).items() if seen > 1)
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

    An unset field takes the value of the nearest level above that states it. Below the plan, the
    task's catalogue entry supplies the default. A stated value applies to this level and every
    level under it. `random` draws a side at this level even when a level above states one.
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


def _absolute_url(url: str, whose: str) -> None:
    """Refuse an address that is not absolute. `httpx.InvalidURL` is not a `ValueError`, so this
    converts it to one for the model to report."""
    try:
        absolute = httpx.URL(url).is_absolute_url
    except httpx.InvalidURL as e:
        raise ValueError(f'endpoint {whose!r} names {url!r}, which is not a URL: {e}') from e
    if not absolute:
        # The platform judges the scheme; this refuses only an address with no host.
        raise ValueError(f'endpoint {whose!r} names {url!r}, which has no host: give an absolute URL')


# A field added to `Cascade` later is refused on an endpoint rather than silently accepted there.
_ENDPOINT_MAY_STATE = frozenset({'name', 'kind', 'url', 'provider', 'spec', 'image', 'episodes_per_endpoint'})


class Endpoint(Cascade):
    """One policy to run, and where it comes from.

    A `remote` endpoint is an address the caller provides. A `served` endpoint names the checkpoint
    it serves (`spec`) and has no `url`: the platform starts it and records the address. `provider`
    names what starts it, and the platform derives one from `spec` when the entry names none. An
    `image` endpoint names the container image the platform runs the policy from. An entry on a task
    carrying no locator at all names one of the plan's endpoints.
    """

    name: str = Field(min_length=1)
    kind: Slugged[EndpointKind] = EndpointKind.remote
    url: str | None = Field(default=None, min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    spec: str | None = Field(default=None, min_length=1)
    # A `PolicyImage`, so a reference the registry could never resolve is refused in the caller's own
    # process instead of spending a round trip to learn it.
    image: PolicyImage | None = None

    @model_validator(mode='before')
    @classmethod
    def _accept_bare_label(cls, value: object) -> object:
        return {'name': value} if isinstance(value, str) else value

    @model_validator(mode='after')
    def _the_kind_carries_its_own_locator(self) -> Self:
        if self.url is not None:
            _absolute_url(self.url, self.name)
        if self.kind is EndpointKind.served:
            if self.url is not None:
                raise ValueError(
                    f'served endpoint {self.name!r} names a url; the platform records the address it serves at'
                )
            if self.image is not None:
                raise ValueError(f'served endpoint {self.name!r} names an image, which only an image endpoint carries')
        elif self.kind is EndpointKind.image:
            if self.url is not None or self.provider is not None or self.spec is not None:
                raise ValueError(
                    f'image endpoint {self.name!r} names a url, a provider or a spec; the platform runs the image'
                )
        elif self.provider is not None or self.spec is not None or self.image is not None:
            raise ValueError(
                f'remote endpoint {self.name!r} names a provider, a spec or an image, which only a served or an '
                'image endpoint carries'
            )
        return self

    @model_validator(mode='after')
    def _overrides_only_the_count(self) -> Self:
        # A dump carries every field, and a plan read back from its own JSON therefore sets them
        # all, so what is refused is a per-task field carrying a value rather than one a dump names.
        fields = type(self).model_fields
        stated = sorted(
            name
            for name in self.model_fields_set - _ENDPOINT_MAY_STATE
            if getattr(self, name) != fields[name].get_default(call_default_factory=True)
        )
        if stated:
            raise ValueError(
                f'endpoint {self.name!r} states {", ".join(stated)}, which are per-task properties: '
                'an endpoint overrides only episodes_per_endpoint'
            )
        return self

    @property
    def names_a_locator(self) -> bool:
        """Whether this entry says where its policy comes from: a `url`, the `spec` a served one
        names, or the `image` the platform runs."""
        return self.url is not None or self.spec is not None or self.image is not None


class TaskNode(Cascade):
    """One task of a plan, by its catalogue id, and what this plan changes for it.

    `endpoints`, when given, replaces the plan's list for this task; an entry with no locator refers
    to a plan endpoint by its name. A bare id takes every value from the plan.
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
    """`submissions.create` — one eval to run: the tasks, the endpoints each task runs, and the
    count per endpoint.

    The plan states the count once. A task may override it for that task, and an endpoint for that
    endpoint. It either states its own tasks or names an eval the platform offers, and the catalogue
    expands that name into the same tasks. The plan carries no client field: the gateway reads the
    client from the key's grant.
    """

    tasks: list[TaskNode] = Field(default_factory=list)
    # The eval whose tasks this plan runs. The catalogue expands it, so a plan states `tasks` or
    # names an eval, and both arrive at the same set.
    eval: EvalRef | None = None
    endpoints: list[Endpoint] = Field(default_factory=list)
    # What a reader calls this run. It names nothing and identifies nothing.
    alias: str | None = None
    # A checksum. When stated, it must equal the sum over the leaves; when absent, the platform fills it in.
    episodes_total: int | None = Field(default=None, ge=1)
    # The upper bound on every leaf's cap: a mistyped cap costs minutes at the rig.
    max_cap_per_episode_sec: int | None = Field(default=None, ge=1)
    # A present key must be non-empty: an empty string is a client bug.
    transaction_key: TransactionKey | None = Field(default=None, min_length=1)

    @property
    def names_an_eval(self) -> bool:
        """Whether the catalogue supplies this plan's tasks, rather than the plan itself.

        Nothing here counts the leaves of such a plan: the expansion happens at the platform.
        """
        return self.eval is not None

    @model_validator(mode='after')
    def _names_a_task(self) -> Self:
        if self.tasks and self.names_an_eval:
            raise ValueError(f'a plan states tasks and names the eval {str(self.eval)!r}: it takes its tasks from one')
        if not self.tasks and not self.names_an_eval:
            raise ValueError('a plan names at least one task, or the eval whose tasks it runs')
        return self

    @model_validator(mode='after')
    def _states_a_count(self) -> Self:
        if self.episodes_per_endpoint is None and not self.names_an_eval:
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
        bare = sorted(entry.name for entry in self.endpoints if not entry.names_a_locator)
        if bare:
            raise ValueError(
                f'the plan defines {", ".join(bare)} with no url and no spec: an endpoint of the plan states '
                'where its policy comes from, and a bare label on a task names one'
            )
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
    def _a_named_eval_states_what_runs_it(self) -> Self:
        """A named eval states no task, so every check that iterates `tasks` reads nothing.

        Anything the catalogue's tasks take from the plan is checked here, or at the level that
        states it. Today that is the policy: the catalogue supplies the tasks, the plan the endpoint.
        """
        if self.names_an_eval and not self.tasks and not self.endpoints:
            raise ValueError(
                f'the plan names {self.eval} and defines no endpoint: the catalogue supplies its tasks, '
                'and the plan supplies the policy that runs them'
            )
        return self

    @model_validator(mode='after')
    def _every_cap_sits_under_the_ceiling(self) -> Self:
        ceiling = self.max_cap_per_episode_sec
        if ceiling is None:
            return self
        # The plan's own cap is checked here rather than through a task that inherits it: a named
        # eval states no task, and a loop over `tasks` would read nothing and pass.
        if self.cap_per_episode_sec is not None and self.cap_per_episode_sec > ceiling:
            raise ValueError(
                f'the plan takes {self.cap_per_episode_sec} s per episode, over its own ceiling of {ceiling} s'
            )
        for task in self.tasks:
            cap = task.cap_per_episode_sec
            if cap is not None and cap > ceiling:
                raise ValueError(
                    f'task {task.task_id!r} takes {cap} s per episode, over the plan ceiling of {ceiling} s'
                )
        return self

    @model_validator(mode='after')
    def _the_checksum_matches(self) -> Self:
        if self.names_an_eval:
            return self
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

        The count comes from `entry`, then from its definition, then from `task`, then from the plan.
        The definition is the plan endpoint of the same name; only an entry with no locator has one.
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
        """Every episode this plan asks for: per task, the count of each endpoint it runs on.

        A plan that names an eval states no task here, so this counts nothing until the catalogue
        expands the name.
        """
        return sum(self.episodes_on(task, entry) for task in self.tasks for entry in self.task_endpoints(task))


# The name the one endpoint of an image run carries. Such a run serves one policy, so nothing picks
# it out by name, and the model asks every endpoint for one.
IMAGE_ENDPOINT_NAME = 'policy'


def plan_of_image(
    image: PolicyImage, eval_name: EvalRef, *, alias: str | None = None, transaction_key: TransactionKey | None = None
) -> EvalPlan:
    """The plan a policy image runs as: one image endpoint, and the eval naming the tasks.

    The catalogue expands the name into tasks and the count each takes, so such a plan states
    neither.
    """
    return EvalPlan(
        eval=eval_name,
        endpoints=[Endpoint(name=IMAGE_ENDPOINT_NAME, kind=EndpointKind.image, image=image)],
        alias=alias,
        transaction_key=transaction_key,
    )
