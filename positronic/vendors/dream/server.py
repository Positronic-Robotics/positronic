"""Serve Dream-1 of Dream Labs over positronic's session protocol.

The policy runs at Dream Labs, behind DreamAPI. `dreamlabs-client` reads the API key from `DREAM_API_KEY` or from
`~/.dreamlabs/credentials`. Every face in a frame is blurred before the frame leaves this server, and each chunk keeps
`limits.ChunkLimits`. Each session first sends a canary: a fixed observation whose reply changes only when the model
changes. Each session writes `<sessions_dir>/<session_id>.json`: the canary, and the trace id, timing and limits of
each call. `<session_id>.npz` beside it holds each chunk as DreamAPI answered it and as the client plays it.

Usage
  uv run --locked --extra dream python -m positronic.vendors.dream.server yam_bimanual
  uv run --locked --extra dream positronic eval run --eval=.sim.amazon_abc.put_bottles --policy=.remote \\
      --policy.address.host=localhost --policy.address.port=8000 --charge_inference_time=False
"""

import base64
import dataclasses
import hashlib
import json
import logging
import ssl
import time
from pathlib import Path
from typing import Any

import configuronic as cfn
import cv2
import dream as dream_client
import numpy as np
from shared.faceblur import FaceBlurrer

from pimm.logging import init_logging
from positronic import keys
from positronic.offboard.server import serve
from positronic.offboard.server_utils import warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codecs import ACTION, ResizeMode, TransformImages
from positronic.policy.processors import ChunkedSchedule, TemporalStack
from positronic.vendors import dream
from positronic.vendors.dream import codecs
from positronic.vendors.dream.limits import ChunkLimits, LimitReport, YamChunkLimiter

logger = logging.getLogger(__name__)

# Dream-1 reads each frame as a JPEG of this quality.
JPEG_QUALITY = 85
# The client bounds each view to this size, and the codec on this server cuts it to `codecs.FRAME_SIZE`.
UPLOAD_SIZE = (640, 400)


def _canary_request() -> dict[str, Any]:
    """A fixed observation: a gradient with a red block that moves right over the window, and both arms raised."""
    width, height = codecs.FRAME_SIZE
    frames = []
    for index in range(codecs.WINDOW_FRAMES):
        frame = np.zeros((height, width, 3), np.uint8)
        frame[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)[None]
        frame[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        cv2.rectangle(frame, (40 + 4 * index, 60), (120 + 4 * index, 140), (200, 50, 50), -1)
        frames.append(frame)
    window = np.stack(frames)
    observation: dict[str, Any] = {dream.PROMPT: 'pick up the red block', **dream.STATION_ROBOT}
    for arm, sign in ((keys.LEFT_ARM, 1.0), (keys.RIGHT_ARM, -1.0)):
        slot = dream.ARM_SLOTS[arm]
        observation[dream.JOINT_POS.format(slot=slot)] = [0.1 * sign, 0.8, 0.9, 0.0, 0.2 * sign, 0.0]
        observation[dream.GRIPPER_POS.format(slot=slot)] = [1.0]
    return {
        dream.OBSERVATION: observation,
        dream.FRAMES: {dream.FRAME_WINDOW.format(camera=camera): window for camera in dream.CAMERAS},
    }


def _watch_traces(client: dream_client.DreamClient) -> list[str | None]:
    """The trace id of each reply that `client` gets from now on, in order, `None` for a reply with none."""
    traces: list[str | None] = []
    # HACK: `DreamClient` drops `status.trace_id` from a reply, and Dream Labs needs it to name the model that
    # answered. This reads it off the private transport of dreamlabs-client 1.0.9, which the `dream` extra pins.
    transport = client._policy._transport
    request = transport.request

    def traced(*args: Any, **kwargs: Any) -> Any:
        response = request(*args, **kwargs)
        status = response.get('status') if isinstance(response, dict) else None
        traces.append(status.get('trace_id') if isinstance(status, dict) else None)
        return response

    transport.request = traced
    return traces


class _Session:
    """One session: its DreamAPI client, the trace ids of its replies, the time of its frames, its chunks, and its
    record."""

    def __init__(self, config: dict[str, Any]):
        self.client = dream_client.DreamClient()
        self.client.policy(config)
        self.traces = _watch_traces(self.client)
        # The time of the last frame sent, on a clock of this session only: DreamAPI reads 0.2 s between frames.
        self.frame_s = 0.0
        # The session time at the first row of the next chunk.
        self.elapsed_s = 0.0
        # When the last call answered, on the monotonic clock.
        self.answered_s: float | None = None
        # Each chunk as DreamAPI answered it and as the client plays it, `(rows, 14)`, left arm first.
        self.answered: list[np.ndarray] = []
        self.played: list[np.ndarray] = []
        self.record: dict[str, Any] = {'calls': [], 'expired': 0, 'renewed': 0}

    def close(self) -> None:
        self.client.close()


def _measured(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """The state of each arm in a Dream observation, as one row: 6 joints, then the gripper opening."""
    return {
        arm: np.concatenate([
            observation[dream.JOINT_POS.format(slot=slot)],
            observation[dream.GRIPPER_POS.format(slot=slot)],
        ])
        for arm, slot in dream.ARM_SLOTS.items()
    }


class DreamModel(Model):
    """Dream-1 behind DreamAPI, one DreamAPI session per session.

    A call sends the frame windows of its request, and answers the chunk, limited, as one action per row. A call
    that times out, finds its session expired or its connection closed, or that DreamAPI refuses as retryable, is
    sent again only while the wait asked for and a whole call timeout still end inside `retry_deadline_s`; otherwise
    it raises at once. A call more than `session_idle_s` after the last one starts a new DreamAPI session, as
    DreamAPI ends an idle session.
    """

    def __init__(
        self,
        config: dict[str, Any],
        limiter: YamChunkLimiter,
        blurrer: FaceBlurrer,
        sessions_dir: Path,
        retry_deadline_s: float,
        session_idle_s: float,
        meta: dict[str, Any],
    ):
        self._config = config
        self._limiter = limiter
        self._blurrer = blurrer
        self._sessions_dir = sessions_dir
        self._retry_deadline_s = retry_deadline_s
        self._call_timeout_s = config['timeout_ms'] / 1000
        self._session_idle_s = session_idle_s
        self._meta = meta
        self._sessions: dict[str, _Session] = {}

    def _jpeg(self, frame: np.ndarray) -> str:
        bgr = self._blurrer.scrub_frame(cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2BGR))
        ok, buffer = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError('A frame does not encode as JPEG')
        return base64.b64encode(buffer.tobytes()).decode('ascii')

    def _send(self, session: _Session, request: dict[str, Any]) -> dict[str, np.ndarray]:
        """The chunk of each arm that DreamAPI answers to `request`, as `(rows, 7)`."""
        observation = request[dream.OBSERVATION]
        windows = {key: [self._jpeg(frame) for frame in stack] for key, stack in request[dream.FRAMES].items()}
        counts = {len(window) for window in windows.values()}
        if len(counts) != 1:
            raise ValueError(f'Every camera must send the same number of frames, got {sorted(counts)}')
        if session.answered_s is not None and time.monotonic() - session.answered_s > self._session_idle_s:
            session.client.reset_episode()
            session.record['renewed'] += 1
        for index in range(counts.pop()):
            session.frame_s += codecs.FRAME_PERIOD_S
            frames = {key: window[index] for key, window in windows.items()}
            session.client.set_observation(frames, observation, timestamp=session.frame_s)
        reply = self._get_action(session)
        session.answered_s = time.monotonic()
        return {arm: np.asarray(reply[dream.CHUNKS[arm]], dtype=np.float64) for arm in keys.BIMANUAL_ARMS}

    def _get_action(self, session: _Session) -> dict[str, Any]:
        """DreamAPI's reply to the observation that `session` holds."""
        deadline = time.monotonic() + self._retry_deadline_s
        stale_sent = False
        while True:
            try:
                return session.client.get_action()
            except dream_client.DreamProtocolError as error:
                # dreamlabs-client 1.0.9 sends a call again on a fresh connection when the gateway has closed its idle
                # keep-alive one, but over TLS that close raises `ssl.SSLEOFError`, which it reports as a gateway it
                # cannot reach. It has dropped the connection, so the call goes once more, on a new one.
                stale = isinstance(error.__cause__, ssl.SSLEOFError)
                if stale_sent or not stale or not _ends_by(deadline, self._call_timeout_s):
                    raise
                stale_sent = True
                logger.warning('The gateway closed an idle connection; sending the call again on a new one')
            except dream_client.DreamTimeout:
                if not _ends_by(deadline, self._call_timeout_s):
                    raise
                logger.warning('DreamAPI did not answer in time; sending the call again')
            except dream_client.DreamSessionExpired as error:
                # DreamAPI ends a session that waits too long between calls. It names a new session, which starts with
                # a reset, and the client takes it.
                if not _ends_by(deadline, self._call_timeout_s):
                    raise
                session.record['expired'] += 1
                logger.warning('%s; sending the call again in the new session', error)
            except dream_client.DreamServerError as error:
                wait = error.retry_after_s or 1.0
                if not error.retryable or not _ends_by(deadline, wait + self._call_timeout_s):
                    raise
                logger.warning('DreamAPI refused a call (%s); sending it again in %.1f s', error, wait)
                time.sleep(wait)
                # The gateway may close the new connection during the wait too.
                stale_sent = False

    def _call_record(self, session: _Session, rows: int, reports: dict[str, LimitReport]) -> dict[str, Any]:
        timing = session.client.last_timing or {}
        return {
            'trace_id': session.traces[-1] if session.traces else None,
            'rows': rows,
            **{name: timing.get(name) for name in ('rtt_ms', 'serve_infer_ms', 'gateway_ms')},
            'limits': {arm: dataclasses.asdict(report) for arm, report in reports.items()},
        }

    def _canary(self) -> dict[str, Any]:
        session = _Session(self._config)
        try:
            chunks = self._send(session, _canary_request())
            actions = np.concatenate([chunks[arm] for arm in keys.BIMANUAL_ARMS], axis=1)
            record = {
                'hash': hashlib.sha256(np.round(actions, 3).tobytes()).hexdigest()[:16],
                'mean': np.round(actions.mean(axis=0), 4).tolist(),
                'trace_id': session.traces[-1] if session.traces else None,
            }
        finally:
            session.close()
        logger.info('Dream canary %s (trace %s)', record['hash'], record['trace_id'])
        return record

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        session = self._sessions.get(session_id)
        if session is None:
            canary = self._canary()
            session = self._sessions[session_id] = _Session(self._config)
            session.record['canary'] = canary
        chunks = self._send(session, dict(obs))
        measured = _measured(obs[dream.OBSERVATION])
        limited, reports = {}, {}
        for arm in keys.BIMANUAL_ARMS:
            limited[arm], reports[arm] = self._limiter.limit(chunks[arm], measured[arm], session.elapsed_s)
        rows = len(limited[keys.LEFT_ARM])
        session.elapsed_s += rows / codecs.FPS
        call = self._call_record(session, rows, reports)
        session.record['calls'].append(call)
        logger.info('Dream call: trace %s, %s ms round trip', call['trace_id'], call['rtt_ms'])
        session.answered.append(np.concatenate([chunks[arm] for arm in keys.BIMANUAL_ARMS], axis=1))
        actions = np.concatenate([limited[arm] for arm in keys.BIMANUAL_ARMS], axis=1)
        session.played.append(actions)
        return [{ACTION: action} for action in actions]

    def end_session(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        try:
            record = {**self._meta, **session.record, policy_keys.SESSION_ID: session_id}
            path = self._sessions_dir / f'{session_id}.json'
            pending = path.with_suffix('.json.tmp')
            pending.write_text(json.dumps(record, indent=1))
            pending.replace(path)
            if session.answered:
                np.savez_compressed(
                    path.with_suffix('.npz'), answered=np.stack(session.answered), played=np.stack(session.played)
                )
        finally:
            session.close()

    def meta(self) -> dict[str, Any]:
        return self._meta

    def close(self) -> None:
        for session_id in list(self._sessions):
            self.end_session(session_id)


def _ends_by(deadline: float, duration_s: float) -> bool:
    """Whether `duration_s` from now ends by `deadline`, on the monotonic clock.

    A call that cannot end by its deadline would only delay the error, and every other session with it, as the
    server makes one call at a time.
    """
    return time.monotonic() + duration_s <= deadline


def _wait_until_warm(config: dict[str, Any], deadline_s: float) -> None:
    """Send black frames until the model answers. A cold model takes up to an hour to start."""
    width, height = codecs.FRAME_SIZE
    black = base64.b64encode(cv2.imencode('.jpg', np.zeros((height, width, 3), np.uint8))[1].tobytes()).decode()
    observation = _canary_request()[dream.OBSERVATION]
    frames = {dream.FRAME_WINDOW.format(camera=camera): black for camera in dream.CAMERAS}
    deadline = time.monotonic() + deadline_s
    with dream_client.DreamClient() as client:
        client.policy(config)
        client.set_observation(frames, observation, timestamp=0.0)
        call_timeout_s = config['timeout_ms'] / 1000
        stale_sent = False
        while True:
            try:
                client.get_action()
                return
            except dream_client.DreamProtocolError as error:
                # The gateway may close the keep-alive connection during the wait, as in `DreamModel._get_action`.
                stale = isinstance(error.__cause__, ssl.SSLEOFError)
                if stale_sent or not stale or not _ends_by(deadline, call_timeout_s):
                    raise
                stale_sent = True
                logger.info('The gateway closed an idle connection; asking again on a new one')
            except dream_client.DreamTimeout:
                if not _ends_by(deadline, call_timeout_s):
                    raise
                logger.info('Dream-1 did not answer in time while it starts; asking again')
            except dream_client.DreamSessionExpired as error:
                # A wait for a cold model outlasts DreamAPI's idle limit of a session, and the client takes the new one.
                if not _ends_by(deadline, call_timeout_s):
                    raise
                logger.info('%s; asking again in the new session', error)
            except dream_client.DreamServerError as error:
                wait = error.retry_after_s or 30.0
                if not error.retryable or not _ends_by(deadline, wait + call_timeout_s):
                    raise
                logger.info('Dream-1 is starting (%s); asking again in %.0f s', error.reason, wait)
                time.sleep(wait)
                stale_sent = False


@cfn.config(
    checkpoint='dreamlabs/dream-1-beta',
    limits=cfn.Config(ChunkLimits),
    sessions_dir='dream_sessions',
    timeout_ms=45000,
    warm_deadline_s=4500.0,
    retry_deadline_s=120.0,
    session_idle_s=20.0,
)
def dream_model(
    checkpoint: str,
    limits: ChunkLimits,
    sessions_dir: str,
    timeout_ms: int,
    warm_deadline_s: float,
    retry_deadline_s: float,
    session_idle_s: float,
) -> Model:
    """Dream-1 checkpoint `checkpoint` behind DreamAPI, warm before it serves.

    `sessions_dir` is relative to the working directory unless absolute, and a leading `~` is the home directory.
    `_record` off asks DreamAPI not to save the sessions to the account. The terms of the API still let Dream Labs
    train on every request.
    """
    config = {'checkpoint': checkpoint, 'timeout_ms': timeout_ms, 'strict': False, '_record': False}
    directory = Path(sessions_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    logger.info('Session records go to %s', directory)
    meta = {
        policy_keys.CHECKPOINT_ID: checkpoint,
        policy_keys.TYPE: 'dream',
        'dream.client_version': dream_client.__version__,
        'dream.limits': dataclasses.asdict(limits),
    }
    limiter = YamChunkLimiter(limits, codecs.FPS)
    model = DreamModel(config, limiter, FaceBlurrer(), directory, retry_deadline_s, session_idle_s, meta)
    _wait_until_warm(config, warm_deadline_s)
    warmup(model, _canary_request())
    return model


@cfn.config(crop_to_aspect=True)
def pipeline(crop_to_aspect: bool) -> PolicyDeployment:
    """The client keeps 9 frames of each camera, 0.2 s apart, and plays each chunk whole at 20 Hz.

    The ramp of the limiter counts every row of a chunk as played, so the schedule takes no horizon.
    """
    offsets = tuple(codecs.FRAME_PERIOD_S * (index + 1 - codecs.WINDOW_FRAMES) for index in range(codecs.WINDOW_FRAMES))
    stack = TemporalStack(
        keys=(keys.EXTERIOR_IMAGE, keys.WRIST_LEFT_IMAGE, keys.WRIST_RIGHT_IMAGE), offsets_sec=offsets, pad_start=False
    )
    local = Sequential(stack, ChunkedSchedule(codecs.FPS), TransformImages(*UPLOAD_SIZE, mode=ResizeMode.BOUND))
    return PolicyDeployment(local, codecs.yam_bimanual.override(crop_to_aspect=crop_to_aspect).instantiate())


COMMANDS = {'yam_bimanual': serve.override(model=dream_model, pipeline=pipeline)}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
