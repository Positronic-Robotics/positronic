"""``RemoteEnvControlSystem``: a remote env server driven as one pimm control system.

A dumb translator: it owns the pimm ports (command receivers, observation + privileged emitters,
``robot_meta``, the eval's ``done``), the trial lifecycle, and the lifetime of the ``serve`` server it talks
to, but no command logic. Each control period it hands the latest command messages to the ``EnvAdapter``,
round-trips the raw action it returns over the wire, and re-emits the canonical signals the adapter maps
back — so only raw arrays cross the boundary and the World's virtual clock advances by the env's
``control_dt`` per step. The adapter owns holding and the canonical<->raw mappings;
``control_dt`` is whatever the latest observation reports (``reset`` and every ``step``).
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack
from typing import Any

import pimm
from positronic import keys, telemetry, telemetry_keys
from positronic.dataset.serializers import Serializers
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Observation
from positronic.eval import keys as eval_keys
from positronic.simulator.env_server.adapter import EnvAdapter
from positronic.simulator.env_server.client import EnvConnection

# Pacing before the first reset, when the env's ``control_dt`` is still unknown. Only sets the instant the
# first reset lands at, then the env's reported ``control_dt`` takes over.
_IDLE_DT = 0.1


class RemoteEnvControlSystem(pimm.ControlSystem):
    def __init__(self, adapter: EnvAdapter, serve: AbstractContextManager[tuple[str, int]]):
        self._adapter = adapter
        # The server this proxy talks to: a context manager yielding its ``(host, port)`` and owning its lifetime
        # (a launched subprocess, or an already-running server whose address it just hands back). Entered when the
        # proxy connects, exited after the socket closes — so the server outlives every request and dies last.
        self._serve = serve
        self._cleanup = ExitStack()
        self._conn: EnvConnection | None = None

        self.commands: pimm.ReceiverDict = pimm.ReceiverDict(self)
        self.observations: pimm.EmitterDict = pimm.EmitterDict(self)
        self.privileged: pimm.EmitterDict = pimm.EmitterDict(self)
        self.robot_meta = pimm.ControlSystemEmitter[dict](self)
        self.done = pimm.ControlSystemEmitter[dict](self)
        self.env_reset = pimm.calls.ControlSystemHandler[Any, None](self)

        # A trial is live between reset and the env's done. The proxy steps only then — not before the
        # first reset (Gym envs reject step-before-reset), not after done. It sleeps every turn regardless.
        self._active = False
        # The latest env frame (``obs`` + ``control_dt``), refreshed by ``reset`` and each ``step``.
        self._frame: dict[str, Any] | None = None
        # The scene meta the env reports at ``reset`` (the task/prompt, scene ids) — constant for the trial; read
        # by the client's ``Task`` for its live instruction. ``step`` omits it.
        self._meta: dict[str, Any] | None = None

    @property
    def meta(self) -> dict[str, Any]:
        """The env's scene meta from the latest ``reset`` (suite, task, …); a client reads its task from here."""
        assert self._meta is not None, 'meta read before the first reset'
        return self._meta

    def _connect(self) -> EnvConnection:
        """The connection to the env server, started and opened on the first call.

        Deferred so positronic can wire the World before the subprocess spawns. The connection closes before
        the server (registered last), so a rollout never races the server's teardown.
        """
        if self._conn is None:
            host, port = self._cleanup.enter_context(self._serve)
            self._conn = EnvConnection(host, port)
            self._cleanup.callback(self._conn.close)
        return self._conn

    def tasks(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """The tasks the env has for ``spec``, as the trial params an eval builds its sweep from."""
        try:
            params = self._adapter.task_params(self._connect().tasks(spec))
            if not params:
                # A sweep of no trials writes nothing and ends at once, which reads as a run that succeeded.
                raise ValueError(f'the env has no task for {spec!r}')
            return params
        except BaseException:
            # An eval lists its tasks before the scheduler enters ``run``, whose teardown is the only other place
            # that closes the server. So a listing that raises stops the server itself.
            self._cleanup.close()
            raise

    def reset(self, params: dict[str, Any]) -> None:
        """Re-randomize the env from the trial's params and publish the scene it draws.

        Resets the remote env (acquiring the fresh frame and its ``control_dt``), then publishes the robot
        model, a full observation payload and a non-terminal ``done``. Stale commands queued while inactive
        are dropped so the first step doesn't apply them.
        """
        conn = self._connect()
        for _, receiver in self.commands.items():
            receiver.read()
        self._frame = conn.reset(self._adapter.reset_token(params))
        self._meta = self._frame['meta']
        self._active = True
        self.robot_meta.emit(self._frame['robot_meta'])
        self._emit_payload(self._frame['obs'])
        # An empty payload clears the wire: a terminal the previous trial reached would end this one at once.
        self.done.emit({})

    def _emit_payload(self, raw_obs: dict[str, Any]) -> None:
        for name, value in self._adapter.observations(raw_obs).items():
            self.observations[name].emit(value)
        for name, value in self._adapter.privileged(raw_obs).items():
            self.privileged[name].emit(value)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Command]:
        try:
            while not should_stop.value:
                # The proxy is the eval's sole time-master: it sleeps one control period every turn —
                # stepping, resetting, or idle between trials alike. Before the first reset the env's
                # ``control_dt`` is unknown, so it paces at ``_IDLE_DT`` until reset reports the real one.
                yield pimm.Sleep(self._frame['control_dt'] if self._frame is not None else _IDLE_DT)
                if (call := next(self.env_reset.incoming(), None)) is not None:
                    with pimm.calls.raise_to(call):
                        self.reset(dict(call.request or {}))
                        call.set_result(None)
                elif self._active:
                    # ``env.step`` spans the whole client-observed step; ``materialize`` nests the client-side
                    # observation assembly (shared-memory image allocation + camera copies) inside it, so the
                    # reduce can split materialisation out of the wire cost.
                    with telemetry.span(telemetry_keys.SPAN_ENV_STEP):
                        self._frame = self._step_env()
                        with telemetry.span(telemetry_keys.SPAN_MATERIALIZE):
                            self._emit_payload(self._frame['obs'])
        finally:
            # Closes the connection then the server, in that order (reverse of acquisition); a no-op if nothing
            # ever connected.
            self._cleanup.close()

    def _step_env(self) -> dict[str, Any]:
        reads = ((name, receiver.read()) for name, receiver in self.commands.items())
        commands = {name: msg for name, msg in reads if msg is not None}
        result = self._conn.step(self._adapter.action(commands))
        payload = self._adapter.terminal(result)
        if payload:  # truthy-valued done: a non-empty payload ends the trial, an empty/``None`` one continues
            self.done.emit(payload)
            self._active = False
        return result


def remote_franka_embodiment(
    proxy: RemoteEnvControlSystem,
    camera_dict: dict[str, str],
    *,
    descriptor: str,
    static_meta: dict[str, Any] | None = None,
) -> Embodiment:
    """The canonical Franka embodiment over a remote env proxy.

    Every remote benchmark exposes the same channels — ``robot_state``/``grip``/one image per ``camera_dict``
    entry, ``robot_command``/``target_grip`` — so their wiring lives here; ``static_meta`` adds the
    embodiment's robot-model payload on top of the canonical signal map (supplied client-side when the env
    server cannot import positronic to emit it via ``robot_meta``).
    """
    observations = {
        keys.ROBOT_STATE: Observation(proxy.observations[keys.ROBOT_STATE], Serializers.robot_state),
        keys.GRIP: Observation(proxy.observations[keys.GRIP], None),
        **{logical: Observation(proxy.observations[logical], Serializers.camera_images) for logical in camera_dict},
    }
    commands = {
        keys.ROBOT_COMMAND: Command(proxy.commands[keys.ROBOT_COMMAND], Serializers.robot_command),
        keys.TARGET_GRIP: Command(proxy.commands[keys.TARGET_GRIP], None),
    }
    return Embodiment(
        descriptor=descriptor,
        observations=observations,
        commands=commands,
        # A remote env readies its own robot when it draws the scene; the proxy has no lever of its own
        prepare_handlers={eval_keys.SCENE: proxy.env_reset},
        static_meta={**ROBOT_STATIC_META, **(static_meta or {})},
        meta_source=proxy.robot_meta,
        control_systems=(proxy,),
        simulated=True,
    )
