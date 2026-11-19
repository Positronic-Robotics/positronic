"""The Luxonis driver with `depthai` faked: the stamps it puts on each frame."""

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from pimm.tests.testing import MockClock
from positronic.drivers.camera import CAPTURE_TIME, HANDOFF_TIME
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.tests.testing_coutils import RecordingEmitter

LUXONIS_PATH = Path(__file__).parents[1] / 'luxonis.py'


class FakeFrame:
    def __init__(self, captured: timedelta):
        self._captured = captured

    def getCvFrame(self) -> np.ndarray:  # noqa: N802 - the depthai name
        return np.zeros((4, 6, 3), np.uint8)

    def getTimestamp(self) -> timedelta:  # noqa: N802 - the depthai name
        return self._captured


@pytest.fixture
def luxonis_module(monkeypatch):
    """Load `luxonis.py` under a private name against a fake `depthai`, so no other test sees the fake."""
    dai = MagicMock(name='depthai')
    monkeypatch.setitem(sys.modules, 'depthai', dai)
    spec = importlib.util.spec_from_file_location('luxonis_with_fake_sdk', LUXONIS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, dai


def test_each_frame_carries_its_capture_and_hand_off_times_on_the_host_steady_clock(luxonis_module):
    module, dai = luxonis_module
    frames = [FakeFrame(timedelta(seconds=10)), FakeFrame(timedelta(seconds=10, milliseconds=50))]
    queue = dai.Device.return_value.__enter__.return_value.getOutputQueue.return_value
    queue.tryGet.side_effect = lambda: frames.pop(0) if frames else None
    dai.Clock.now.side_effect = [timedelta(seconds=10, milliseconds=12), timedelta(seconds=10, milliseconds=65)]

    camera, emitted, stop = module.LuxonisCamera(), RecordingEmitter(), StopFlag()
    camera.frame._bind(emitted, clock=MockClock())
    loop = camera.run(stop, MockClock())
    for _ in range(3):
        next(loop)

    stamps = [(time[CAPTURE_TIME], time[HANDOFF_TIME]) for time, _ in emitted.emitted]
    assert stamps == [(10_000_000_000, 10_012_000_000), (10_050_000_000, 10_065_000_000)]
