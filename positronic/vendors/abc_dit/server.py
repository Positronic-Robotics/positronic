"""Serve an ABC-DiT checkpoint over positronic's session protocol.

The policy runs in ABC's own interpreter, as ABC's policy server started by `backend.py`, on this machine.
`multitask_200k` serves `abc_dit_xl_200k_model.pt`, ABC's multi-task sim policy, with the prompts it trained under.
`multitask_200k_yambox_crop` serves it with the yambox views cropped to the field of view of ABC's cameras, and
`multitask_200k_yambox_crop_no_sim_prefix` does the same with the prompts of its sim tasks sent without the sim prefix.
A session may change the crop, the schedule and the prompt prefix with the query of its URL, for example
`?crop=yambox_wide_top&schedule=rtc&sim_prefix=false`. See `pipeline`.

Usage
  python -m positronic.vendors.abc_dit.server multitask_200k --model.abc_python=<ABC's python> \
      --model.checkpoint=<dir>/abc_dit_xl_200k_model.pt
"""

import socket
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import configuronic as cfn
import numpy as np
from platform_client.policy_container import WARMUP_DEADLINE_S

from pimm.logging import init_logging
from positronic.offboard.roboarena import RoboarenaClient
from positronic.offboard.server import serve
from positronic.offboard.server_utils import wait_for_subprocess_ready, warmup
from positronic.offboard.spec import Model, PolicyDeployment
from positronic.policy import Sequential
from positronic.policy import keys as policy_keys
from positronic.policy.base import Obs
from positronic.policy.codecs import ACTION, ResizeMode, TransformImages
from positronic.policy.processors import ChunkedSchedule, ConstantDuration
from positronic.vendors import abc_dit
from positronic.vendors.abc_dit import codecs

# The script that ABC's interpreter runs: ABC's own websocket policy server, on loopback.
BACKEND_SCRIPT = Path(__file__).with_name('backend.py')


class AbcDitModel(Model):
    """An ABC-DiT checkpoint behind ABC's policy server in a child process, which this model owns.

    Each request carries standard normal flow noise from one generator seeded with `noise_seed`, as ABC's sim eval
    draws it. A request with a prefix length gets ABC's real-time chunking: the prefix is the last actions of the
    session's previous reply, or the state for its first request, as ABC's deploy does. The backend warmed only
    `prefix_length`, so a request with another length is refused.
    """

    def __init__(
        self, backend: subprocess.Popen, port: int, noise_seed: int, prefix_length: int | None, meta: dict[str, Any]
    ):
        self._backend = backend
        self._client = RoboarenaClient(port=port)
        self._noise_seed = noise_seed
        self._prefix_length = prefix_length
        self.restart_noise()
        self._meta = meta
        # The actions of each session's last reply, in ABC's action space, which the client plays in full.
        self._played: dict[str, np.ndarray] = {}

    def restart_noise(self) -> None:
        self._noise = np.random.default_rng(self._noise_seed)

    def __call__(self, obs: Obs, *, session_id: str) -> list[dict[str, Any]]:
        request = dict(obs)
        executed = request.pop(abc_dit.EXECUTED_ACTIONS, None)
        length = request.get(abc_dit.PREFIX_LENGTH, 0)
        if length:
            if length != self._prefix_length:
                raise ValueError(f'The backend serves an action prefix of {self._prefix_length}, not {length}')
            played = self._played.get(session_id)
            state = np.asarray(request[abc_dit.STATE], dtype=np.float32)
            request[abc_dit.ACTION_PREFIX] = np.tile(state, (length, 1)) if played is None else played[-length:]
        request[abc_dit.NOISE] = self._noise.standard_normal(
            (codecs.CHUNK_LENGTH, codecs.BIMANUAL_SIZE), dtype=np.float32
        )
        actions = np.asarray(self._client.infer(request)[abc_dit.ACTIONS])
        if length:
            # The first actions of the chunk are the prefix, which the robot plays while the policy computes.
            actions = actions[length : length + executed]
            self._played[session_id] = actions
        return [{ACTION: action} for action in actions]

    def end_session(self, session_id: str) -> None:
        self._played.pop(session_id, None)

    def meta(self) -> dict[str, Any]:
        return self._meta

    def close(self) -> None:
        self._client.close()
        self._backend.terminate()
        try:
            self._backend.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._backend.kill()
            self._backend.wait()


def _exit_status(process: subprocess.Popen) -> tuple[bool, int | None]:
    code = process.poll()
    return code is not None, code


def _warm_observation() -> dict[str, Any]:
    """Black views and a zero state, in the fields of ABC's policy server."""
    width, height = codecs.VIEW_SIZE
    view = np.zeros((3, height, width), dtype=np.uint8)
    cameras = (abc_dit.TOP_CAMERA, abc_dit.LEFT_CAMERA, abc_dit.RIGHT_CAMERA)
    return {
        abc_dit.STATE: np.zeros(codecs.BIMANUAL_SIZE, dtype=np.float32),
        abc_dit.IMAGES: dict.fromkeys(cameras, view),
        abc_dit.PROMPT: '',
    }


@cfn.config(fast_inference=True, noise_seed=0, backend_port=9000, prefix_length=None)
def abc_dit_model(
    checkpoint: str,
    abc_python: str,
    fast_inference: bool,
    noise_seed: int,
    backend_port: int,
    prefix_length: int | None,
) -> Model:
    """The DiT checkpoint at `checkpoint`, served by ABC's policy server in ABC's interpreter `abc_python`.

    `fast_inference` runs the DiT in bfloat16 and compiles it, as ABC's sim eval does; it needs a CUDA device.
    `prefix_length` is the one action prefix length that the backend warms and serves; `None` serves none.
    """
    checkpoint_path, python = Path(checkpoint), Path(abc_python)
    command = [
        python,
        '-P',
        BACKEND_SCRIPT,
        f'--policy.checkpoint-path={checkpoint_path}',
        '--policy-type=dit',
        '--policy.fast-inference' if fast_inference else '--policy.no-fast-inference',
        f'--port={backend_port}',
    ]
    if prefix_length is not None:
        command.append(f'--policy.rtc-prefix-length={prefix_length}')
    # Refuse a port another server holds: the readiness probe would reach that server, not this checkpoint.
    with socket.socket() as port_check:
        port_check.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port_check.bind(('127.0.0.1', backend_port))
    backend = subprocess.Popen(command)
    meta = {
        policy_keys.CHECKPOINT_ID: checkpoint_path.name,
        policy_keys.TYPE: 'abc_dit',
        'fast_inference': fast_inference,
    }
    model = AbcDitModel(backend, backend_port, noise_seed, prefix_length, meta)
    try:
        probe = RoboarenaClient(port=backend_port)
        wait_for_subprocess_ready(
            lambda: probe.probe() is None,
            partial(_exit_status, backend),
            'ABC policy server',
            max_wait=WARMUP_DEADLINE_S,
        )
        warmup(model, _warm_observation())
        # ABC's sim eval warms up with a generator of its own, so its first rollout gets the seed's first draw.
        model.restart_noise()
    except Exception:
        model.close()
        raise
    return model


@dataclass(frozen=True)
class Schedule:
    """The client plays `executed` actions of each chunk, and asks for the next chunk when `lead` of them are left.
    With a `prefix`, the next chunk starts with the last `prefix` played actions, as ABC's real-time chunking does."""

    executed: int
    lead: int = 0
    prefix: int = 0

    def __post_init__(self):
        if not 0 <= self.prefix <= self.lead < self.executed:
            raise ValueError(f'A schedule needs 0 <= prefix <= lead < executed, got {self}')

    def client_schedule(self) -> ChunkedSchedule:
        return ChunkedSchedule(
            codecs.FPS, horizon_sec=self.executed / codecs.FPS, inference_lead=ConstantDuration(self.lead / codecs.FPS)
        )

    def action_prefix(self) -> codecs.ActionPrefix | None:
        return codecs.ActionPrefix(self.prefix, self.executed) if self.prefix else None


# The schedules that a session can name. `sync` is ABC's synchronous sim eval: 15 actions, then a new observation.
# `rtc` takes the flags of ABC's real-robot deploy, `--rtc-prefix-length=4 --rtc-inference-lead-steps=7
# --execute-chunk-dim=16`: the client asks 7 actions before the chunk ends, which covers the round trip.
SCHEDULES = {'sync': Schedule(executed=15), 'rtc': Schedule(executed=16, lead=7, prefix=4)}
# The one prefix length of the schedules, which the backend warms.
PREFIX_LENGTH = SCHEDULES['rtc'].prefix

# A crop box down to half of a frame's width and height still holds ABC's view size when the client sends this bound.
CROPPED_UPLOAD_SIZE = (2 * codecs.VIEW_SIZE[0], 2 * codecs.VIEW_SIZE[1])


@cfn.config(crop='none', schedule='sync', sim_prefix=True)
def pipeline(crop: str, schedule: str, sim_prefix: bool) -> PolicyDeployment:
    """The views cropped by the set `crop` of `codecs.CROPS`, played by the client as `schedule` of `SCHEDULES`, with
    the sim prompt prefix when `sim_prefix`. A session sets each one in its URL query. The client shrinks each view
    to fit ABC's view size, or twice that size for a cropped view, and the codec runs on this server."""
    if crop not in codecs.CROPS:
        raise ValueError(f'Unknown crop {crop!r}; the crops are {sorted(codecs.CROPS)}')
    if schedule not in SCHEDULES:
        raise ValueError(f'Unknown schedule {schedule!r}; the schedules are {sorted(SCHEDULES)}')
    if not isinstance(sim_prefix, bool):
        raise ValueError(f'sim_prefix must be true or false, got {sim_prefix!r}')
    plan = SCHEDULES[schedule]
    codec = codecs.multitask_200k.override(**{
        'obs.crops': codecs.CROPS[crop],
        'obs.prompt_prefix': codecs.SIM_PROMPT_PREFIX if sim_prefix else '',
        'obs.action_prefix': plan.action_prefix(),
    }).instantiate()
    upload_size = CROPPED_UPLOAD_SIZE if codecs.CROPS[crop] else codecs.VIEW_SIZE
    return PolicyDeployment(
        Sequential(plan.client_schedule(), TransformImages(*upload_size, mode=ResizeMode.BOUND)), codec
    )


yambox_crop_pipeline = pipeline.override(crop='yambox')
model_with_prefix = abc_dit_model.override(prefix_length=PREFIX_LENGTH)

COMMANDS = {
    'multitask_200k': serve.override(model=model_with_prefix, pipeline=pipeline),
    'multitask_200k_yambox_crop': serve.override(model=model_with_prefix, pipeline=yambox_crop_pipeline),
    'multitask_200k_yambox_crop_no_sim_prefix': serve.override(
        model=model_with_prefix, pipeline=yambox_crop_pipeline.override(sim_prefix=False)
    ),
}


if __name__ == '__main__':
    init_logging()
    cfn.cli(COMMANDS)
