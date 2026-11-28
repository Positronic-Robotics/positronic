#!/usr/bin/env python3
"""Time the image transform of a declared client stack, per control step and per policy call.

The probe builds each stack twice, with the transform at the call and moved in front of the history, and feeds it
random camera frames at the control rate. The policy answers at once, so a call's time is the client's own work.

Usage
    uv run python utilities/probe_frame_transform.py
    uv run python utilities/probe_frame_transform.py --width 1280 --height 720 --steps 300
"""

import argparse
import logging
import statistics
import time
from collections.abc import Callable

import numpy as np
from positronic_model_server.spec import component, parallel, sequence

from positronic import keys, telemetry, telemetry_keys
from positronic.cfg.hardware.roboarm import DROID_IMPEDANCE
from positronic.dataset.transforms.image import Resample, ResizeMode
from positronic.policy.arrival import transform_frames_on_arrival
from positronic.policy.base import Processor
from positronic.policy.codecs import TASK_FIELD, SetControlMode, TransformImages
from positronic.policy.codecs.action import AbsolutePositionAction
from positronic.policy.codecs.observation import SelectObservationFields
from positronic.policy.compatibility import CHUNKED_SCHEDULE, PauseOnUnavailable, RTCScheduleV1, restrict_image_size_v1
from positronic.policy.executor import Executor, WaitStatus
from positronic.policy.keys import RTC
from positronic.policy.processors import TemporalStack
from positronic.policy.spec import from_spec

logger = logging.getLogger(__name__)

FPS = 15.0
DROID_CAMERAS = (keys.EXTERIOR_IMAGE, keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE_2)
BIMANUAL_CAMERAS = (keys.EXTERIOR_IMAGE, keys.WRIST_LEFT_IMAGE, keys.WRIST_RIGHT_IMAGE)


def _offsets(count: int) -> tuple[float, ...]:
    return tuple(-i / FPS for i in reversed(range(count)))


def three_camera_history() -> dict:
    """Four history frames of three cameras, a real-time-chunking schedule, and a stretch to 640x360 at each call."""
    return sequence(
        TemporalStack(DROID_CAMERAS, _offsets(4)).to_spec(),
        RTCScheduleV1(FPS, call_after_sec=0.53, latency_estimate=0.3).to_spec(),
        parallel(
            SelectObservationFields({**{name: name for name in DROID_CAMERAS}, TASK_FIELD: TASK_FIELD}).to_spec(),
            sequence(
                SetControlMode(DROID_IMPEDANCE).to_spec(),
                AbsolutePositionAction(
                    keys.TARGET_EE_POSE, keys.TARGET_GRIP, commands_path=(RTC.ACTION_PREFIX, RTC.PREFIX_ACTIONS)
                ).to_spec(),
            ),
        ),
        TransformImages(
            640, 360, mode=ResizeMode.STRETCH, resample=Resample.AREA, paths=[[name] for name in DROID_CAMERAS]
        ).to_spec(),
    )


def bimanual_bounded_history() -> dict:
    """25 history frames of three cameras, a chunked schedule, and a 416x416 bound at each call."""
    return sequence(
        PauseOnUnavailable().to_spec(),
        TemporalStack(BIMANUAL_CAMERAS, _offsets(25), pad_start=False).to_spec(),
        component(CHUNKED_SCHEDULE, fps=FPS),
        restrict_image_size_v1(width=416, height=416),
    )


SHAPES: dict[str, tuple[Callable[[], dict], tuple[str, ...]]] = {
    'three_camera_history': (three_camera_history, DROID_CAMERAS),
    'bimanual_bounded_history': (bimanual_bounded_history, BIMANUAL_CAMERAS),
}


class _CallTimes:
    """Collect the wall time of each policy call, from the ``policy.submit`` span that the runtime opens for it."""

    def __init__(self):
        self.ms: list[float] = []

    def __call__(self, name: str, start_ns: int, end_ns: int) -> None:
        if name == telemetry_keys.SPAN_POLICY_SUBMIT:
            self.ms.append((end_ns - start_ns) / 1e6)


def _declared(spec: dict) -> Processor:
    stack = from_spec(spec)
    if not isinstance(stack, Processor):
        raise TypeError('A client stack must be a processor')
    return stack


def _drive(stack: Processor, cameras: tuple[str, ...], width: int, height: int, steps: int) -> list[float]:
    """Step ``stack`` at ``FPS`` with new frames each step, and return the wall time of each step in ms."""
    rng = np.random.default_rng(0)
    now = [0]
    runtime = Executor(lambda: now[0], simulated=True, charge_inference_time=False)
    run = runtime.start(stack, lambda obs: None)
    step_ms = []
    try:
        for step in range(steps):
            now[0] = round(step * 1e9 / FPS)
            runtime.start_tick(now[0])
            obs = {
                **{name: rng.integers(0, 256, (height, width, 3), dtype=np.uint8) for name in cameras},
                keys.JOINTS: np.zeros(7),
                keys.EE_POSE: np.zeros(7),
                keys.GRIP: np.zeros(1),
                keys.ROBOT_STATUS: 0,
                TASK_FIELD: 'pick',
            }
            start = time.perf_counter()
            run.send(obs)
            step_ms.append((time.perf_counter() - start) * 1e3)
            while runtime.wait(timeout_sec=30).status is WaitStatus.ANSWERS_READY:
                pass
    finally:
        run.close()
        runtime.close()
    return step_ms


def _summary(values: list[float]) -> str:
    if not values:
        return 'none'
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return f'n={len(values)} p50={statistics.median(values):.1f} ms p95={p95:.1f} ms max={ordered[-1]:.1f} ms'


def main() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--width', type=int, default=1280, help='camera frame width')
    parser.add_argument('--height', type=int, default=720, help='camera frame height')
    parser.add_argument('--steps', type=int, default=150, help='control steps to drive')
    parser.add_argument('--shape', choices=sorted(SHAPES), action='append', help='declared stack; default: all')
    args = parser.parse_args()
    for name in args.shape or sorted(SHAPES):
        declare, cameras = SHAPES[name]
        for form, stack in (
            ('at each call', _declared(declare())),
            ('on arrival', transform_frames_on_arrival(_declared(declare()))),
        ):
            calls = _CallTimes()
            with telemetry.timings_to(calls):
                step_ms = _drive(stack, cameras, args.width, args.height, args.steps)
            logger.info('%s, transform %s', name, form)
            logger.info('  policy call, before the request goes out: %s', _summary(calls.ms))
            logger.info('  control step:                             %s', _summary(step_ms))


if __name__ == '__main__':
    main()
