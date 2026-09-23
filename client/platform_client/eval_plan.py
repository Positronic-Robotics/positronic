"""`EvalPlan`: an eval to run — the tasks, the policies that run them, and the count per policy per task.

Unknown fields are rejected, so a misspelled field is a 422.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Generic, Self

import httpx
from platform_client.enums import CameraVantage, EndpointKind, Placement
from platform_client.evals import EvalRef
from platform_client.ids import TransactionKey
from platform_client.model_config import INPUT_MODEL_CONFIG
from platform_client.policy_images import PolicyImage
from platform_client.slug import Slugged
from platform_client.tasks import TaskRef
from pydantic import BaseModel, Field, SecretStr, SerializationInfo, model_serializer, model_validator
from typing_extensions import TypeVar


def _require_unique_names(names: list[str], whose: str) -> None:
    repeated = sorted(name for name, seen in Counter(names).items() if seen > 1)
    if repeated:
        raise ValueError(f'{whose} names {", ".join(repeated)} more than once; each entry names one')


class Clutter(BaseModel):
    """How much of the rest of the object kit a run draws onto the table.

    The draw picks a count between the two bounds per run; at most `large_cap` large objects and
    `medium_cap` medium ones, and the rest small.
    """

    model_config = INPUT_MODEL_CONFIG

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

    model_config = INPUT_MODEL_CONFIG

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


# The serialisation context key under which a registry password dumps as plaintext. The client's
# send path alone sets it.
REVEAL_REGISTRY_PASSWORD = 'reveal_registry_password'


class RegistryCredential(BaseModel):
    """The username and password that open the registry one image endpoint names.

    A request carries this model. It holds the password as a value, and no field of it names a path.
    """

    model_config = INPUT_MODEL_CONFIG

    username: str = Field(min_length=1)
    password: SecretStr = Field(min_length=1)

    def plaintext_password(self) -> str:
        return self.password.get_secret_value()

    @model_serializer(mode='plain', when_used='json')
    def _dump(self, info: SerializationInfo) -> dict[str, str]:
        reveal = (info.context or {}).get(REVEAL_REGISTRY_PASSWORD)
        return {'username': self.username, 'password': self.plaintext_password() if reveal else str(self.password)}


class RegistryCredentialFile(BaseModel):
    """A registry credential as a plan file states it: the username, and the file the password is in.

    The gateway never validates this model. `plan_with_passwords_read` turns a plan of it into the
    plan a request carries.
    """

    model_config = INPUT_MODEL_CONFIG

    username: str = Field(min_length=1)
    password_file: Path

    @property
    def password(self) -> SecretStr:
        """Read from the file on each access. `RegistryCredential` validates from these attributes."""
        return SecretStr(password_from_file(self.password_file))


def password_from_file(password_file: Path) -> str:
    """The registry password a caller states as a path, read from the file it names.

    The file's last line ending comes off. A path that is mistyped, names a directory, cannot be
    read, or holds only whitespace raises `ValueError`, which every caller of this reports as a
    refusal.
    """
    try:
        path = password_file.expanduser()
    except RuntimeError as exc:  # `~name` for a user this machine does not have
        raise ValueError(f'{password_file} names no home directory: {exc}') from exc
    if not path.is_file():
        raise ValueError(f'{path} is not a file; password_file names the file the registry password is in')
    try:
        held = path.read_text()
    except OSError as exc:
        raise ValueError(f'{path} cannot be read: {exc.strerror}') from exc
    password = held.removesuffix('\n').removesuffix('\r')
    if not password.strip():
        raise ValueError(f'{path} holds no password')
    return password


def credential_from_file(username: str, password_file: Path) -> RegistryCredential:
    """The credential a caller states as a username and the path of the file its password is in."""
    return RegistryCredential(username=username, password=SecretStr(password_from_file(password_file)))


# The credential an image endpoint carries. A request carries a `RegistryCredential`, and so does a
# plan with no parameter. A plan file carries a `RegistryCredentialFile`.
Credential = TypeVar('Credential', RegistryCredential, RegistryCredentialFile, default=RegistryCredential)


# An endpoint overrides one cascading property; every other property of `Cascade` is per task.
_ENDPOINT_OVERRIDES = 'episodes_per_endpoint'
_PER_TASK_ONLY = frozenset(Cascade.model_fields) - {_ENDPOINT_OVERRIDES}


class Endpoint(Cascade, Generic[Credential]):
    """One policy to run, and where it comes from.

    * `remote` — the caller provides the address, as `url`.
    * `served` — `spec` names the checkpoint, and the platform starts it and records the address.
      `provider` names what starts it, and the platform derives one from `spec` when the entry
      names none.
    * `image` — `image` names the container the platform runs the policy from, and
      `image_credential` opens the registry that serves it to no anonymous caller.
    * no locator at all — the entry sits on a task and names one of the plan's endpoints.
    """

    name: str = Field(min_length=1)
    kind: Slugged[EndpointKind] = EndpointKind.remote
    url: str | None = Field(default=None, min_length=1)
    provider: str | None = Field(default=None, min_length=1)
    spec: str | None = Field(default=None, min_length=1)
    # A `PolicyImage`, so a reference the registry could never resolve is refused in the caller's own
    # process instead of spending a round trip to learn it.
    image: PolicyImage | None = None
    image_credential: Credential | None = None

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
    def _a_credential_opens_the_image_this_entry_names(self) -> Self:
        """Refuse a credential on an entry that names no image. A bare label runs a plan endpoint,
        which carries its own credential."""
        if self.image_credential is not None and self.image is None:
            raise ValueError(
                f'endpoint {self.name!r} states image_credential and names no image; a credential opens the '
                'image the entry that states it names'
            )
        return self

    @model_validator(mode='after')
    def _overrides_only_the_count(self) -> Self:
        # A dump carries every field, and a plan read back from its own JSON therefore sets them
        # all, so what is refused is a per-task field carrying a value rather than one a dump names.
        fields = type(self).model_fields
        stated = sorted(
            name
            for name in self.model_fields_set & _PER_TASK_ONLY
            if getattr(self, name) != fields[name].get_default(call_default_factory=True)
        )
        if stated:
            raise ValueError(
                f'endpoint {self.name!r} states {", ".join(stated)}, which are per-task properties: '
                f'an endpoint overrides only {_ENDPOINT_OVERRIDES}'
            )
        return self

    @property
    def names_a_locator(self) -> bool:
        """Whether this entry says where its policy comes from: a `url`, the `spec` a served one
        names, or the `image` the platform runs."""
        return self.url is not None or self.spec is not None or self.image is not None


class TaskNode(Cascade, Generic[Credential]):
    """One task of a plan, by its catalogue id, and what this plan changes for it.

    `endpoints`, when given, replaces the plan's list for this task; an entry with no locator refers
    to a plan endpoint by its name. A bare id takes every value from the plan. A node is identified
    by its place in the plan's list, not by `task_id`: a plan may carry two nodes of one task.
    """

    task_id: TaskRef
    endpoints: list[Endpoint[Credential]] | None = Field(default=None, min_length=1)

    @model_validator(mode='before')
    @classmethod
    def _accept_bare_id(cls, value: object) -> object:
        return {'task_id': value} if isinstance(value, str) else value

    @model_validator(mode='after')
    def _each_endpoint_is_named_once(self) -> Self:
        _require_unique_names([entry.name for entry in self.endpoints or []], f'task {self.task_id!r}')
        return self


class EvalPlan(Cascade, Generic[Credential]):
    """`submissions.create` — one eval to run: the tasks, the endpoints each task runs, and the
    count per endpoint.

    The plan states the count once. A task may override it for that task, and an endpoint for that
    endpoint. It either states its own tasks or names an eval the platform offers, and the catalogue
    expands that name into the same tasks. The plan carries no client field: the gateway reads the
    client from the key's grant.
    """

    tasks: list[TaskNode[Credential]] = Field(default_factory=list)
    # The eval whose tasks this plan runs. The catalogue expands it, so a plan states `tasks` or
    # names an eval, and both arrive at the same set.
    eval: EvalRef | None = None
    endpoints: list[Endpoint[Credential]] = Field(default_factory=list)
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

    def task_endpoints(self, task: TaskNode[Credential]) -> list[Endpoint[Credential]]:
        """The endpoints `task` runs on: its own list, else the plan's."""
        return self.endpoints if task.endpoints is None else task.endpoints

    def episodes_on(self, task: TaskNode[Credential], entry: Endpoint[Credential]) -> int:
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


def plan_with_passwords_read(plan: EvalPlan[RegistryCredentialFile]) -> EvalPlan:
    """The plan a request carries: `plan`, with the password of each credential read from its file.

    A file that gives no password raises `ValidationError` at the place of its credential in the plan.
    """
    return EvalPlan[RegistryCredential].model_validate(plan, from_attributes=True)


# The name the one endpoint of an image run carries. Such a run serves one policy, so nothing picks
# it out by name, and the model asks every endpoint for one.
IMAGE_ENDPOINT_NAME = 'policy'


def plan_of_image(
    image: PolicyImage,
    eval_name: EvalRef,
    *,
    alias: str | None = None,
    transaction_key: TransactionKey | None = None,
    credential: RegistryCredential | None = None,
) -> EvalPlan:
    """The plan a policy image runs as: one image endpoint, and the eval naming the tasks.

    The catalogue expands the name into tasks and the count each takes, so such a plan states
    neither. `credential` opens the registry when `image` is not public.
    """
    return EvalPlan(
        eval=eval_name,
        endpoints=[
            Endpoint(name=IMAGE_ENDPOINT_NAME, kind=EndpointKind.image, image=image, image_credential=credential)
        ],
        alias=alias,
        transaction_key=transaction_key,
    )
