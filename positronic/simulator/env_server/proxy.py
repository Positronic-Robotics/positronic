"""``RemoteEnvControlSystem``: a remote env server driven as one pimm control system.

A dumb translator: it owns the pimm ports (command receivers, observation + privileged emitters,
``robot_meta``, the Task's ``done``) and the trial lifecycle, but no command logic. Each control period
it hands the latest command messages to the ``EnvAdapter``, round-trips the raw action it returns over
the wire, and re-emits the canonical signals the adapter maps back — so only raw arrays cross the
boundary and the World's virtual clock advances by the env's ``control_dt`` per step. The adapter owns
trajectory playing, holding, and the canonical<->raw mappings; ``control_dt`` is whatever the latest
observation reports (``reset`` and every ``step``).
"""

from collections.abc import Iterator
from typing import Any

import pimm
from positronic.simulator.env_server.adapter import EnvAdapter
from positronic.simulator.env_server.client import EnvConnection

# Pacing before the first reset, when the env's ``control_dt`` is still unknown. Only sets the instant
# frame-0 lands at, then the env's reported ``control_dt`` takes over.
_IDLE_DT = 0.1


class RemoteEnvControlSystem(pimm.ControlSystem):
    def __init__(self, adapter: EnvAdapter, host: str, port: int):
        self._adapter = adapter
        self._conn = EnvConnection(host, port)

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
        # Set by ``reset``; the run loop publishes frame-0 (instead of stepping) on its next turn and clears it.
        self._reset_pending = False

    def reset(self, seed: int | None = None) -> None:
        """Re-randomize the env from ``seed`` and arm frame-0 publication for the next turn (the ``RUN`` hook).

        Resets the remote env (acquiring the fresh frame and its ``control_dt``), then flags the run loop
        to publish the scene meta, a full observation payload (frame-0) and a non-terminal ``done`` on its
        next turn — in sequence, so the recorder samples frame-0 before any step. Stale commands queued
        while inactive (e.g. the inter-episode home) are dropped so the first step doesn't apply them.
        """
        for _, receiver in self.commands.items():
            receiver.read()
        self._frame = self._conn.reset(self._adapter.reset_token(seed))
        self._reset_pending = True
        self._active = True

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
                    self.robot_meta.emit(self._frame['meta'])
                    self._emit_payload(self._frame['obs'])
                    self.done.emit({})
                elif self._active:
                    self._frame = self._step_env(clock)
                    self._emit_payload(self._frame['obs'])
        finally:
            self._conn.close()

    def _step_env(self, clock: pimm.Clock) -> dict[str, Any]:
        commands = {name: receiver.read() for name, receiver in self.commands.items()}
        result = self._conn.step(self._adapter.action(commands, clock.now_ns()))
        payload = self._adapter.terminal(result)
        if payload:  # truthy-valued done: a non-empty payload ends the trial, an empty/``None`` one continues
            self.done.emit(payload)
            self._active = False
        return result
