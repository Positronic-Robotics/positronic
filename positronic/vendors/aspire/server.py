"""Serve an ASPIRE CapApi program over the offboard protocol, one run of the program per session.

uv run --locked python -m positronic.vendors.aspire.server --program=path/to/program.py
"""

import hashlib
import logging
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import configuronic as cfn
from positronic_model_server import server_wire
from pydantic import BaseModel

from pimm.logging import init_logging
from positronic import keys
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame, ee_frames_entry
from positronic.offboard import server as offboard_server
from positronic.offboard.server import serve_model
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs, PolicyEnd
from positronic.policy.processors import ChunkedSchedule, EndOnField, TrajectoryPlayer
from positronic.policy.sequential import Sequential
from positronic.utils import flatten_dict

from .bridge import BridgeSettings, CapBridge, ProgramOutcome

logger = logging.getLogger(__name__)

POLICY_TYPE = 'aspire_program'
META_PROGRAM = 'program'
META_PROGRAM_SHA256 = 'program_sha256'
META_SETTINGS = 'settings'


class SessionRecord(BaseModel):
    """What ``<sessions_dir>/<session_id>.json`` holds: how the session's program ended, and its traceback."""

    session_id: str
    program_sha256: str
    end: PolicyEnd | None
    error: str | None


class CapProgramModel(Model):
    """One program, run once per session in a ``CapBridge`` of its own.

    A session writes its record when its program ends, and again when the session ends. A record of a session that
    ended before its program has no ``end``.
    """

    def __init__(self, program: Path, settings: BridgeSettings, sessions_dir: Path):
        source = program.read_text()
        self._code = compile(source, str(program), 'exec')
        self._program_name = program.name
        self._program_sha256 = hashlib.sha256(source.encode()).hexdigest()
        self._settings = settings
        self._sessions_dir = sessions_dir
        self._bridges: dict[str, CapBridge] = {}
        self._recorded: set[str] = set()

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        bridge = self._bridges.get(session_id)
        if bridge is None:
            bridge = self._bridges[session_id] = CapBridge(self._code, self._settings)
        chunk = bridge.answer(obs)
        if bridge.outcome is not None and session_id not in self._recorded:
            self._recorded.add(session_id)
            self._write(session_id, bridge.outcome)
        return chunk

    def end_session(self, session_id: str) -> None:
        bridge = self._bridges.pop(session_id, None)
        if bridge is None:
            return
        bridge.close()
        self._recorded.discard(session_id)
        self._write(session_id, bridge.outcome)

    def _write(self, session_id: str, outcome: ProgramOutcome | None) -> None:
        record = SessionRecord(
            session_id=session_id,
            program_sha256=self._program_sha256,
            end=None if outcome is None else outcome.end,
            error=None if outcome is None else outcome.error,
        )
        path = self._sessions_dir / f'{session_id}.json'
        pending = path.with_suffix('.json.tmp')
        pending.write_text(record.model_dump_json())
        pending.replace(path)

    def meta(self) -> dict[str, Any]:
        # The program reads the measured hand pose and moves the hand at the rig's grip point.
        signals = [f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.EE_POSE_SUFFIX}' for arm in keys.BIMANUAL_ARMS] + [
            f'{keys.arm_channel(keys.ROBOT_COMMAND, arm)}{keys.POSE_SUFFIX}' for arm in keys.BIMANUAL_ARMS
        ]
        frames = ee_frames_entry(dict.fromkeys(signals, EEFrame(site=GRIPPER_SITE)))
        return frames | flatten_dict({
            policy_keys.TYPE: POLICY_TYPE,
            META_PROGRAM: self._program_name,
            META_PROGRAM_SHA256: self._program_sha256,
            META_SETTINGS: asdict(self._settings),
        })

    def close(self) -> None:
        for session_id in list(self._bridges):
            self.end_session(session_id)


def deployment(settings: BridgeSettings) -> PolicyDeployment:
    """The client stack that plays each chunk at ``settings.fps`` and ends the run on the program's ``END``."""
    return PolicyDeployment(
        Sequential(EndOnField(), TrajectoryPlayer(replan_on_error=False), ChunkedSchedule(settings.fps))
    )


@cfn.config(
    settings=cfn.Config(BridgeSettings),
    sessions_dir='aspire_sessions',
    websocket=offboard_server.websocket,
    grpc=None,
    idle_timeout_min=None,
)
def serve(
    program: str,
    settings: BridgeSettings,
    sessions_dir: str,
    websocket: server_wire.Wire | None,
    grpc: server_wire.Wire | None,
    idle_timeout_min: float | None,
):
    """Serve the CapApi program at ``program`` to bimanual sessions, with one run of the program per session.

    ``--settings.*`` sets the pacing and how the bridge decides that a move ended. Each session's record goes to
    ``sessions_dir``, relative to the working directory unless absolute. The wire flags are those of the offboard
    ``serve``.
    """
    directory = Path(sessions_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    logger.info('Session records go to %s', directory)
    build = partial(CapProgramModel, Path(program), settings, directory)
    wires = [w for w in (websocket, grpc) if w is not None]
    serve_model(build, deployment(settings), wires, idle_timeout_min)


if __name__ == '__main__':
    init_logging()
    cfn.cli(serve)
