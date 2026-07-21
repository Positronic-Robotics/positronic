"""``RemoteEnvControlSystem``: a remote env server driven as one pimm control system.

A dumb translator: it owns the pimm ports (command receivers, observation + privileged emitters,
``robot_meta``, the Task's ``done``), the trial lifecycle, and the lifetime of the ``serve`` server it talks
to, but no command logic. Each control period it hands the latest command messages to the ``EnvAdapter``,
round-trips the raw action it returns over the wire, and re-emits the canonical signals the adapter maps
back — so only raw arrays cross the boundary and the World's virtual clock advances by the env's
``control_dt`` per step. The adapter owns trajectory playing, holding, and the canonical<->raw mappings;
``control_dt`` is whatever the latest observation reports (``reset`` and every ``step``).
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, ExitStack
from typing import Any

import pimm
from positronic import eval_timing
from positronic.dataset.serializers import Serializers
from positronic.drivers.roboarm import command as roboarm_command
from positronic.eval import ROBOT_STATIC_META, Command, Embodiment, Observation
from positronic.eval_timing import Phase
from positronic.simulator.env_server.adapter import EnvAdapter
from positronic.simulator.env_server.client import EnvConnection

# Pacing before the first reset, when the env's ``control_dt`` is still unknown. Only sets the instant
# frame-0 lands at, then the env's reported ``control_dt`` takes over.
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

        self.commands: pimm.ReceiverDict = pimm.ReceiverDict(self, default=None)
        self.observations: pimm.EmitterDict = pimm.EmitterDict(self)
        self.privileged: pimm.EmitterDict = pimm.EmitterDict(self)
        self.robot_meta: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)
        self.done: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)

        # A trial is live between reset and the env's done. The proxy steps only then — not before the
        # first reset (Gym envs reject step-before-reset), not after done. It sleeps every turn regardless.
        self._active = False
        # The latest env frame (``obs`` + ``control_dt``), refreshed by ``reset`` and each ``step``.
        self._frame: dict[str, Any] | None = None
        # The scene meta the env reports at ``reset`` (the task/prompt, scene ids) — constant for the trial; read
        # by the client's ``Task`` for its live instruction. ``step`` omits it.
        self._meta: dict[str, Any] | None = None
        # The robot model identity the env reports at ``reset`` (URDF / joint names) — emitted on the
        # ``robot_meta`` port into the episode; distinct from the scene ``meta`` above.
        self._robot_meta: dict[str, Any] | None = None
        # Set by ``reset``; the run loop publishes frame-0 (instead of stepping) on its next turn and clears it.
        self._reset_pending = False

    @property
    def meta(self) -> dict[str, Any]:
        """The env's scene meta from the latest ``reset`` (suite, task, …); a client reads its task from here."""
        assert self._meta is not None, 'meta read before the first reset'
        return self._meta

    def reset(self, context: dict[str, Any]) -> None:
        """Re-randomize the env from the trial context and arm frame-0 publication for the next turn (the ``RUN`` hook).

        Resets the remote env (acquiring the fresh frame and its ``control_dt``), then flags the run loop
        to publish the scene meta, a full observation payload (frame-0) and a non-terminal ``done`` on its
        next turn — in sequence, so the recorder samples frame-0 before any step. Stale commands queued
        while inactive (e.g. the inter-episode home) are dropped so the first step doesn't apply them.
        """
        if self._conn is None:
            # Start the server and connect on the first reset, not at construction, so positronic can wire the
            # World before the subprocess spawns. The connection closes before the server (registered last), so a
            # rollout never races the server's teardown.
            host, port = self._cleanup.enter_context(self._serve)
            self._conn = EnvConnection(host, port)
            self._cleanup.callback(self._conn.close)
        for _, receiver in self.commands.items():
            receiver.read()
        self._frame = self._conn.reset(self._adapter.reset_token(context))
        self._meta = self._frame['meta']
        self._robot_meta = self._frame['robot_meta']
        self._reset_pending = True
        self._active = True
        # Clear any terminal the previous trial left on the wire: the env can reach ``done`` while the proxy
        # free-runs between trials, and the reset-publish turn re-clears it only later — the harness, which runs
        # before producers, would otherwise sample that stale success as this trial's terminal.
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
                # stepping, publishing frame-0, or idle between trials alike. Before the first reset the
                # env's ``control_dt`` is unknown, so it paces at ``_IDLE_DT`` until reset reports the real one.
                yield pimm.Sleep(self._frame['control_dt'] if self._frame is not None else _IDLE_DT)
                if self._reset_pending:
                    # The reset is this turn's step: publish the env's frame-0 (no step) and clear the prior
                    # terminal, so the recorder samples it before any step advances the env.
                    self._reset_pending = False
                    self.robot_meta.emit(self._robot_meta)
                    # Frame-0 materialisation (allocating shared-memory image buffers and copying each camera
                    # frame) is part of the reset cost, like ``_conn.reset``'s server-side render already under
                    # ``reset_s`` — time it there so it isn't billed to ``overhead_s``.
                    with eval_timing.timed(Phase.RESET):
                        self._emit_payload(self._frame['obs'])
                    self.done.emit({})
                elif self._active:
                    self._frame = self._step_env(clock)
                    # The step's observation is materialised client-side here (the adapter allocates
                    # shared-memory image buffers and copies each camera frame). It is part of the env step —
                    # matching the native path, so image-heavy remote runs don't bill it to ``overhead_s`` — but
                    # tracked apart (``env_client_s``) so the reduce charges it to materialisation, not wire.
                    with eval_timing.timed(Phase.MATERIALIZE):
                        self._emit_payload(self._frame['obs'])
        finally:
            # Closes the connection then the server, in that order (reverse of acquisition); a no-op if no reset
            # ever connected.
            self._cleanup.close()

    def _step_env(self, clock: pimm.Clock) -> dict[str, Any]:
        commands = {name: receiver.read() for name, receiver in self.commands.items()}
        with eval_timing.timed(Phase.ENV_STEP):
            result = self._conn.step(self._adapter.action(commands, clock.now_ns()))
        # An env server that decomposes its own step cost reports it in the response; record it against the
        # client-observed ``env_step_s`` so the reduce can split wire, physics, render and server plumbing.
        timing = result.get('timing')
        if timing is not None:
            eval_timing.record_env_phases(timing['physics_s'], timing['render_s'], timing['wall_s'])
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
        'robot_state': Observation(proxy.observations['robot_state'], Serializers.robot_state),
        'grip': Observation(proxy.observations['grip'], None),
        **{logical: Observation(proxy.observations[logical], Serializers.camera_images) for logical in camera_dict},
    }
    commands = {
        'robot_command': Command(proxy.commands['robot_command'], roboarm_command.Reset(), Serializers.robot_command),
        'target_grip': Command(proxy.commands['target_grip'], 0.0, None),
    }
    return Embodiment(
        descriptor=descriptor,
        observations=observations,
        commands=commands,
        static_meta={**ROBOT_STATIC_META, **(static_meta or {})},
        meta_source=proxy.robot_meta,
        control_systems=(proxy,),
        simulated=True,
    )
