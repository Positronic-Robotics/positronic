"""Serve ASPIRE CapApi programs over the offboard protocol. Each session names the program it runs.

uv run --locked python -m positronic.vendors.aspire.server --programs_dir=path/to/programs
"""

import logging
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import configuronic as cfn
from positronic_model_server import server_wire
from pydantic import BaseModel, TypeAdapter

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

from .bridge import BridgeSettings, CallTrace, CameraCalibration, CapBridge, Program, check_calibration
from .contracts import SegmentationProvider

logger = logging.getLogger(__name__)

POLICY_TYPE = 'aspire_program'
# The session parameter that names the program, relative to the server's programs directory.
PARAM_PROGRAM = 'program'
META_PROGRAM = 'program'
META_PROGRAM_SHA256 = 'program_sha256'
META_SESSION_RECORD = 'session_record'
META_SETTINGS = 'settings'
META_CALIBRATION = 'calibration'
META_SEGMENTATION = 'segmentation'
EXAMPLES_DIR = Path(__file__).parent / 'examples'


class SessionRecord(BaseModel):
    """What ``<sessions_dir>/<session_id>.json`` holds: the program, how it ended, and each CapApi call it made."""

    session_id: str
    program: str
    program_sha256: str
    program_source: str
    end: PolicyEnd | None
    error: str | None
    calls: list[CallTrace]


class CapProgramModel(Model):
    """Runs the program each session names, in a ``CapBridge`` of the session's own.

    A session writes its record when its program ends, and again when the session ends. A record of a session that
    ended before its program has no ``end``. ``calibration`` is a JSON file that maps each camera name to its
    ``CameraCalibration``. All sessions share the calibration and the segmentation provider.
    """

    def __init__(
        self,
        programs_dir: Path,
        settings: BridgeSettings,
        sessions_dir: Path,
        *,
        calibration: Path | None = None,
        segmentation: SegmentationProvider | None = None,
    ):
        self._programs_dir = programs_dir.resolve()
        self._settings = settings
        self._sessions_dir = sessions_dir
        self._calibration: dict[str, CameraCalibration] = {}
        if calibration is not None:
            self._calibration = TypeAdapter(dict[str, CameraCalibration]).validate_json(calibration.read_text())
        check_calibration(self._calibration, settings)
        if segmentation is not None and (
            isinstance(segmentation, type) or not isinstance(segmentation, SegmentationProvider)
        ):
            raise TypeError(f'segmentation must be a SegmentationProvider instance, got {segmentation!r}')
        self._segmentation = segmentation
        self._bridges: dict[str, CapBridge] = {}
        self._recorded: set[str] = set()

    def parameters(self) -> dict[str, Any]:
        return {PARAM_PROGRAM: None}

    def open_session(self, session_id: str, params: dict[str, Any]) -> dict[str, Any]:
        name = params[PARAM_PROGRAM]
        if not isinstance(name, str) or not name:
            raise ValueError(f'Name the program with the session parameter {PARAM_PROGRAM!r}')
        path = (self._programs_dir / name).resolve()
        if not path.is_relative_to(self._programs_dir) or not path.is_file():
            raise ValueError(f'{name!r} names no program in the programs directory')
        program = Program(name, path.read_text())
        self._bridges[session_id] = CapBridge(
            program, self._settings, calibration=self._calibration, segmentation=self._segmentation
        )
        return {
            META_PROGRAM: program.name,
            META_PROGRAM_SHA256: program.sha256,
            META_SESSION_RECORD: self._record_path(session_id).name,
        }

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        bridge = self._bridges[session_id]
        chunk = bridge.answer(obs)
        if bridge.outcome is not None and session_id not in self._recorded:
            self._recorded.add(session_id)
            self._write(session_id, bridge)
        return chunk

    def end_session(self, session_id: str) -> None:
        bridge = self._bridges.pop(session_id, None)
        if bridge is None:
            return
        bridge.close()
        self._recorded.discard(session_id)
        self._write(session_id, bridge)

    def _record_path(self, session_id: str) -> Path:
        return self._sessions_dir / f'{session_id}.json'

    def _write(self, session_id: str, bridge: CapBridge) -> None:
        outcome = bridge.outcome
        record = SessionRecord(
            session_id=session_id,
            program=bridge.program.name,
            program_sha256=bridge.program.sha256,
            program_source=bridge.program.source,
            end=None if outcome is None else outcome.end,
            error=None if outcome is None else outcome.error,
            calls=bridge.calls,
        )
        path = self._record_path(session_id)
        pending = path.with_suffix('.json.tmp')
        pending.write_text(record.model_dump_json())
        pending.replace(path)

    def meta(self) -> dict[str, Any]:
        # The program reads the measured hand pose and moves the hand at the rig's grip point.
        arms = self._settings.arms
        signals = [f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.EE_POSE_SUFFIX}' for arm in arms] + [
            f'{keys.arm_channel(keys.ROBOT_COMMAND, arm)}{keys.POSE_SUFFIX}' for arm in arms
        ]
        frames = ee_frames_entry(dict.fromkeys(signals, EEFrame(site=GRIPPER_SITE)))
        calibration = {name: camera.model_dump() for name, camera in self._calibration.items()}
        meta = {policy_keys.TYPE: POLICY_TYPE, META_SETTINGS: asdict(self._settings), META_CALIBRATION: calibration}
        if self._segmentation is not None:
            provider = type(self._segmentation)
            meta[META_SEGMENTATION] = f'{provider.__module__}.{provider.__qualname__}'
        return frames | flatten_dict(meta)

    def close(self) -> None:
        for session_id in list(self._bridges):
            self.end_session(session_id)


def deployment(settings: BridgeSettings) -> PolicyDeployment:
    """The client stack that plays each chunk at ``settings.fps`` and ends the run on the program's ``END``."""
    return PolicyDeployment(
        Sequential(EndOnField(), TrajectoryPlayer(replan_on_error=False), ChunkedSchedule(settings.fps))
    )


@cfn.config(
    programs_dir=str(EXAMPLES_DIR),
    settings=cfn.Config(BridgeSettings),
    sessions_dir='aspire_sessions',
    calibration=None,
    segmentation=None,
    websocket=offboard_server.websocket,
    grpc=None,
    idle_timeout_min=None,
)
def serve(
    programs_dir: str,
    settings: BridgeSettings,
    sessions_dir: str,
    calibration: str | None,
    segmentation: SegmentationProvider | None,
    websocket: server_wire.Wire | None,
    grpc: server_wire.Wire | None,
    idle_timeout_min: float | None,
):
    """Serve the CapApi programs in ``programs_dir`` to the sessions of one rig.

    Each session names its program with the session parameter ``program``, a path relative to ``programs_dir``, and
    runs the file as it reads at the session's start. ``--settings.*`` sets the rig's arms, the pacing and how the
    bridge decides that a move ended. Each session's record goes to ``sessions_dir``, relative to the working
    directory unless absolute. ``calibration`` names the calibration file. ``segmentation`` is the segmentation
    provider. The wire flags are those of the offboard ``serve``.
    """
    directory = Path(sessions_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    logger.info('Session records go to %s', directory)
    build = partial(
        CapProgramModel,
        Path(programs_dir),
        settings,
        directory,
        calibration=None if calibration is None else Path(calibration),
        segmentation=segmentation,
    )
    wires = [w for w in (websocket, grpc) if w is not None]
    serve_model(build, deployment(settings), wires, idle_timeout_min)


if __name__ == '__main__':
    init_logging()
    cfn.cli(serve)
