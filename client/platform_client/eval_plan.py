"""`EvalPlan`: an eval to run — the tasks, the policies that run them, and the count per policy per task.

Unknown fields are rejected, so a misspelled field is a 422.
"""

from __future__ import annotations

import ipaddress
import re
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal, Self

from platform_client.enums import CameraVantage, EndpointKind, Placement, RequestType, Wire
from platform_client.evals import EvalRef
from platform_client.ids import OrgSlug, TransactionKey
from platform_client.policy_images import PolicyImage
from platform_client.slug import Slugged, members_by_slug, slug_of
from platform_client.tasks import TaskRef
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

_FORBID_EXTRA = ConfigDict(extra='forbid')


def _require_unique_names(names: list[str], whose: str) -> None:
    repeated = sorted(name for name, seen in Counter(names).items() if seen > 1)
    if repeated:
        raise ValueError(f'{whose} names {", ".join(repeated)} more than once; each entry names one')


class Clutter(BaseModel):
    """How much of the rest of the object kit a run draws onto the table.

    The draw picks a count between the two bounds once per plan, and every run of the plan lays out
    the same table: at most `large_cap` large objects and `medium_cap` medium ones, and the rest small.
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


# Each address field holds what the table in the client README states, and no other value.
_HOSTNAME = re.compile(r'[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*\.?')
_VISIBLE_ASCII = re.compile(r'[!-~]*')


def _a_bare_host(host: str) -> str:
    if ':' in host:
        try:
            literal = ipaddress.IPv6Address(host)
        except ValueError:
            literal = None
        if literal is not None:
            if literal.scope_id is not None:
                raise ValueError(f'host {host!r} carries an IPv6 zone index: write the address without one')
            return host
    elif _HOSTNAME.fullmatch(host):
        return host
    raise ValueError(
        f'host {host!r} is no hostname and no IP address: write the host alone, with no scheme, port, path, '
        'userinfo or brackets; name the wire in `wire` and the port in `port`'
    )


def _no_fragment_and_visible_ascii(field: str, value: str) -> str:
    if '#' in value:
        raise ValueError(f'{field} {value!r} carries `#`, which starts a URL fragment: write it as `%23`')
    if not _VISIBLE_ASCII.fullmatch(value):
        raise ValueError(f'{field} {value!r} holds a space or a character outside visible ASCII: percent-encode it')
    return value


def _a_session_path(path: str) -> str:
    if not path.startswith('/'):
        raise ValueError(f'path {path!r} is no session route: write the route alone, from its leading `/`')
    if '?' in path:
        raise ValueError(f'path {path!r} carries `?`: write the params in `query`')
    return _no_fragment_and_visible_ascii('path', path)


def _a_bare_query(query: str) -> str:
    if query.startswith('?'):
        raise ValueError(f'query {query!r} starts with `?`: write the params alone')
    return _no_fragment_and_visible_ascii('query', query)


def _an_absolute_path(uds: Path) -> Path:
    if '\0' in str(uds):
        raise ValueError(f'uds {str(uds)!r} holds a NUL byte, which no socket path holds')
    # A relative path is resolved against the directory each process was started from.
    if not uds.is_absolute():
        raise ValueError(f'uds {str(uds)!r} is a relative socket path; name an absolute one')
    return uds


Host = Annotated[str, AfterValidator(_a_bare_host)]
Port = Annotated[int, Field(ge=1, le=65535)]
# `session_path(model)` in `positronic_wire.wire`: the route a session on one model opens on.
SessionPath = Annotated[str, AfterValidator(_a_session_path)]
# The session params as written: the server reads each value as a JSON literal.
SessionQuery = Annotated[str, AfterValidator(_a_bare_query)]
SocketPath = Annotated[Path, AfterValidator(_an_absolute_path)]


class HostPortAddress(BaseModel):
    """A session on a server reached over the network."""

    model_config = _FORBID_EXTRA

    host: Host
    port: Port
    path: SessionPath
    query: SessionQuery = ''


class UnixSocketAddress(BaseModel):
    """A session on a server on the same machine, opened on the socket it bound."""

    model_config = _FORBID_EXTRA

    uds: SocketPath
    path: SessionPath
    query: SessionQuery = ''


class RoboarenaAddress(BaseModel):
    """A session on a roboarena server, at the root of the port the partner published."""

    model_config = _FORBID_EXTRA

    host: Host
    port: Port


EndpointAddress = HostPortAddress | UnixSocketAddress | RoboarenaAddress

# The fields each wire dials: `ClientWire.ADDRESS` of the wire the registry names.
ADDRESS_OF_WIRE: dict[Wire, type[EndpointAddress]] = {
    Wire.websocket: HostPortAddress,
    Wire.websocket_tls: HostPortAddress,
    Wire.websocket_unix: UnixSocketAddress,
    Wire.grpc: HostPortAddress,
    Wire.grpc_tls: HostPortAddress,
    Wire.roboarena: RoboarenaAddress,
}


# A field added to `Cascade` later is refused on an endpoint rather than silently accepted there.
_ENDPOINT_MAY_STATE = frozenset({
    'name',
    'kind',
    'wire',
    'address',
    'provider',
    'spec',
    'image',
    'episodes_per_endpoint',
})


class Endpoint(Cascade):
    """One policy to run, where it comes from, and the wire a session with it runs over.

    A `remote` endpoint is a server the caller provides: `address` carries the fields its `wire`
    dials. A `served` endpoint names the checkpoint it serves (`spec`) and carries no address: the
    platform starts it and records one. `provider` names what starts it, and the platform derives one
    from `spec` when the entry names none. An `image` endpoint names the container image the platform
    runs the policy from. Every kind names its wire. An entry on a task carrying no locator at all
    names one of the plan's endpoints.
    """

    name: str = Field(min_length=1)
    kind: Slugged[EndpointKind] = EndpointKind.remote
    wire: Slugged[Wire] | None = None
    address: EndpointAddress | None = None
    provider: str | None = Field(default=None, min_length=1)
    spec: str | None = Field(default=None, min_length=1)
    # A `PolicyImage`, so a reference the registry could never resolve is refused in the caller's own
    # process instead of spending a round trip to learn it.
    image: PolicyImage | None = None

    @model_validator(mode='before')
    @classmethod
    def _accept_bare_label(cls, value: object) -> object:
        if isinstance(value, str):
            return {'name': value}
        if isinstance(value, dict) and 'url' in value:
            fields = '; '.join(
                f'{slug_of(wire)}: {", ".join(address.model_fields)}' for wire, address in ADDRESS_OF_WIRE.items()
            )
            raise ValueError(
                f"endpoint {value.get('name')!r} names a url; an endpoint names its `wire` and that wire's "
                f'`address` fields instead ({fields})'
            )
        return value

    @model_validator(mode='after')
    def _the_kind_carries_its_own_locator(self) -> Self:
        if self.wire is None and self.names_a_locator:
            raise ValueError(f'endpoint {self.name!r} names no wire; name one of {", ".join(members_by_slug(Wire))}')
        if self.wire is not None and not self.names_a_locator:
            raise ValueError(f'endpoint {self.name!r} names a wire and nothing that runs on it')
        if self.wire is not None and self.address is not None:
            dialled = ADDRESS_OF_WIRE[self.wire]
            if not isinstance(self.address, dialled):
                carried = ', '.join(type(self.address).model_fields)
                raise ValueError(
                    f'endpoint {self.name!r} names the {slug_of(self.wire)} wire, which dials '
                    f'{", ".join(dialled.model_fields)}; the address carries {carried}'
                )
        if self.kind is EndpointKind.served:
            if self.address is not None:
                raise ValueError(
                    f'served endpoint {self.name!r} names an address; the platform records the address it serves at'
                )
            if self.image is not None:
                raise ValueError(f'served endpoint {self.name!r} names an image, which only an image endpoint carries')
        elif self.kind is EndpointKind.image:
            if self.address is not None or self.provider is not None or self.spec is not None:
                raise ValueError(
                    f'image endpoint {self.name!r} names an address, a provider or a spec; the platform runs the image'
                )
            if self.wire is not None and self.wire is not Wire.websocket:
                raise ValueError(
                    f'image endpoint {self.name!r} names the {slug_of(self.wire)} wire; an image endpoint takes the '
                    f'{slug_of(Wire.websocket)} wire, which the platform opens every image session on'
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
        """Whether this entry says where its policy comes from: the `address` a remote one dials, the
        `spec` a served one names, or the `image` the platform runs."""
        return self.address is not None or self.spec is not None or self.image is not None


class NebiusCompetition(BaseModel):
    """A public run: one of the named public evals, one image, the daily quota, and a board."""

    model_config = _FORBID_EXTRA

    type: Literal['nebius_competition'] = 'nebius_competition'

    @property
    def kind(self) -> RequestType:
        return RequestType.nebius_competition


class PrivateEval(BaseModel):
    """A run for one organisation: its approved evals, tasks and endpoint kinds, and no board."""

    model_config = _FORBID_EXTRA

    type: Literal['private_eval'] = 'private_eval'
    # The caller must be a member of this org.
    org: OrgSlug = Field(min_length=1)

    @property
    def kind(self) -> RequestType:
        return RequestType.private_eval


# Tagged by `type`. A later request type is one more member.
PlanRequestType = Annotated[NebiusCompetition | PrivateEval, Field(discriminator='type')]


class TaskNode(Cascade):
    """One task of a plan, by its catalogue id, and what this plan changes for it.

    `endpoints`, when given, replaces the plan's list for this task; an entry with no locator refers
    to a plan endpoint by its name. A bare id takes every value from the plan. A node is identified
    by its place in the plan's list, not by `task_id`: a plan may carry two nodes of one task.
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

    # The rules, the approvals and the board this plan runs under.
    request_type: PlanRequestType
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
    def _a_competition_run_names_an_eval(self) -> Self:
        if isinstance(self.request_type, NebiusCompetition) and not self.names_an_eval:
            raise ValueError('a nebius_competition plan names an eval and states no tasks')
        return self

    @model_validator(mode='after')
    def _states_a_count(self) -> Self:
        if self.episodes_per_endpoint is None and not self.names_an_eval:
            raise ValueError('a plan states episodes_per_endpoint; a task or an endpoint overrides it')
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
                f'the plan defines {", ".join(bare)} with no address, no spec and no image: an endpoint of the plan '
                'states where its policy comes from, and a bare label on a task names one'
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
    image: PolicyImage,
    eval_name: EvalRef,
    *,
    alias: str | None = None,
    transaction_key: TransactionKey | None = None,
    org: OrgSlug | None = None,
) -> EvalPlan:
    """The plan a policy image runs as: one image endpoint, and the eval naming the tasks.

    The endpoint names the websocket wire: the platform opens every image session over the websocket.
    The catalogue expands the eval name into tasks and the count each takes, so such a plan states
    neither. The plan is a `nebius_competition` run, or a private run for `org` where one is given.
    """
    return EvalPlan(
        request_type=NebiusCompetition() if org is None else PrivateEval(org=org),
        eval=eval_name,
        endpoints=[Endpoint(name=IMAGE_ENDPOINT_NAME, kind=EndpointKind.image, wire=Wire.websocket, image=image)],
        alias=alias,
        transaction_key=transaction_key,
    )
