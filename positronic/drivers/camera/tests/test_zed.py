"""The ZED driver with the SDK faked: its open path, and its recovery of a camera that drops off the bus."""

import enum
import importlib.util
import inspect
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

import pimm
from pimm.tests.testing import MockClock
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.tests.testing_coutils import RecordingEmitter

SERIAL = 39567055
ZED_PATH = Path(__file__).parents[1] / 'zed.py'


class ErrorCode(enum.Enum):
    SUCCESS = 'SUCCESS'
    CAMERA_NOT_DETECTED = 'CAMERA NOT DETECTED'
    CAMERA_REBOOTING = 'CAMERA REBOOTING'
    INVALID_FUNCTION_CALL = 'INVALID FUNCTION CALL'
    FAILURE = 'FAILURE'


class FakeSdk:
    """Stands in for `pyzed.sl`: scripted open results, a device list, and a reboot that records its calls."""

    def __init__(self, opens: list[ErrorCode], listed: list[bool], reboot_result: ErrorCode = ErrorCode.SUCCESS):
        self.opens = list(opens)
        self.listed = list(listed)
        self.reboot_result = reboot_result
        self.open_calls = 0
        self.list_calls = 0
        self.reboots: list[int] = []

    def open(self, _init_params) -> ErrorCode:
        self.open_calls += 1
        return self.opens.pop(0)

    def get_device_list(self) -> list[types.SimpleNamespace]:
        self.list_calls += 1
        present = self.listed.pop(0) if len(self.listed) > 1 else self.listed[0]
        return [types.SimpleNamespace(serial_number=SERIAL)] if present else []

    def reboot(self, serial: int) -> ErrorCode:
        self.reboots.append(serial)
        return self.reboot_result


@pytest.fixture
def zed_module(monkeypatch):
    """Load `zed.py` under a private name against a fake `pyzed`, so no other test sees the fake."""
    sl = types.ModuleType('pyzed.sl')
    pyzed = types.ModuleType('pyzed')
    monkeypatch.setitem(sys.modules, 'pyzed', pyzed)
    monkeypatch.setitem(sys.modules, 'pyzed.sl', sl)
    spec = importlib.util.spec_from_file_location('zed_with_fake_sdk', ZED_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'device_open_lock', nullcontext)
    return module


def _bind(module, sdk: FakeSdk) -> None:
    module.sl.ERROR_CODE = ErrorCode
    module.sl.Camera = types.SimpleNamespace(get_device_list=sdk.get_device_list, reboot=sdk.reboot)


def _open(module, sdk: FakeSdk, reboot_serial: int | None = SERIAL) -> list[pimm.Sleep]:
    _bind(module, sdk)
    zed = types.SimpleNamespace(open=sdk.open)
    return list(module.SLCamera._open_under_device_lock(zed, None, reboot_serial))


NOT_DETECTED = ErrorCode.CAMERA_NOT_DETECTED


def test_a_camera_the_sdk_cannot_find_is_rebooted_once_and_opened(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 3 + [ErrorCode.SUCCESS], listed=[False, True])
    _open(zed_module, sdk)
    assert sdk.reboots == [SERIAL]
    assert sdk.open_calls == 4


def test_the_open_waits_until_the_rebooted_camera_is_listed(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 3 + [ErrorCode.SUCCESS], listed=[False, False, True])
    sleeps = _open(zed_module, sdk)
    retry_sleeps, unlisted_polls = 2, 2
    assert len(sleeps) == retry_sleeps + unlisted_polls
    assert sdk.open_calls == 4


def test_a_camera_missing_from_the_device_list_is_rebooted_whatever_the_open_error(zed_module):
    sdk = FakeSdk(opens=[ErrorCode.FAILURE] * 3 + [ErrorCode.SUCCESS], listed=[False, True])
    _open(zed_module, sdk)
    assert sdk.reboots == [SERIAL]


def test_a_listed_camera_that_fails_to_open_is_not_rebooted(zed_module):
    sdk = FakeSdk(opens=[ErrorCode.FAILURE] * 3, listed=[True])
    with pytest.raises(RuntimeError, match='after 3 attempts: ErrorCode.FAILURE'):
        _open(zed_module, sdk)
    assert sdk.reboots == []


def test_a_failed_open_after_the_reboot_raises_without_a_second_reboot(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 4, listed=[True])
    with pytest.raises(RuntimeError, match='after its reboot'):
        _open(zed_module, sdk)
    assert sdk.reboots == [SERIAL]
    assert sdk.open_calls == 4


def test_a_reboot_the_sdk_refuses_raises_the_open_error(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 3, listed=[False], reboot_result=ErrorCode.FAILURE)
    with pytest.raises(RuntimeError, match='CAMERA_NOT_DETECTED; the SDK did not reboot camera'):
        _open(zed_module, sdk)
    assert sdk.open_calls == 3


def test_a_camera_that_is_not_listed_after_the_reboot_raises(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 3, listed=[False])
    with pytest.raises(RuntimeError, match='still not listed after its reboot'):
        _open(zed_module, sdk)
    assert sdk.list_calls == zed_module.SLCamera.REBOOTED_CAMERA_MAX_POLLS
    assert sdk.reboots == [SERIAL]
    assert sdk.open_calls == 3


def test_an_open_without_a_serial_is_not_rebooted(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 3, listed=[False])
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        _open(zed_module, sdk, reboot_serial=None)
    assert sdk.reboots == []


class DroppingSdk:
    """Stands in for `sl.Camera` over one ZED that leaves the device list at ``lost_at`` and returns at ``listed_at``.

    A camera opened before the drop grabs again only from ``sdk_recovers_at``. ``None`` is a model whose loss the SDK
    never recovers, such as the ZED Mini. The first ``failed_reopens`` opens after the drop fail.
    """

    def __init__(
        self,
        clock: MockClock,
        lost_at: float,
        listed_at: float,
        sdk_recovers_at: float | None = None,
        failed_reopens: int = 0,
    ):
        self.clock = clock
        self.lost_at = lost_at
        self.listed_at = listed_at
        self.sdk_recovers_at = sdk_recovers_at
        self.failed_reopens = failed_reopens
        self.cameras: list[FakeCamera] = []
        self.reboots: list[int] = []

    def __call__(self) -> 'FakeCamera':
        camera = FakeCamera(self)
        self.cameras.append(camera)
        return camera

    def is_listed(self) -> bool:
        return not self.lost_at <= self.clock.now() < self.listed_at

    def get_device_list(self) -> list[types.SimpleNamespace]:
        return [types.SimpleNamespace(serial_number=SERIAL)] if self.is_listed() else []

    def reboot(self, serial: int) -> ErrorCode:
        self.reboots.append(serial)
        return ErrorCode.INVALID_FUNCTION_CALL


class FakeCamera:
    def __init__(self, sdk: DroppingSdk):
        self._sdk = sdk
        self.opened_at: float | None = None
        self.closed_at: float | None = None

    def open(self, _init_params) -> ErrorCode:
        if not self._sdk.is_listed():
            return ErrorCode.CAMERA_NOT_DETECTED
        if self._sdk.clock.now() >= self._sdk.lost_at and self._sdk.failed_reopens > 0:
            self._sdk.failed_reopens -= 1
            return ErrorCode.FAILURE
        self.opened_at = self._sdk.clock.now()
        return ErrorCode.SUCCESS

    def grab(self) -> ErrorCode:
        assert self.opened_at is not None and self.closed_at is None, 'grab on a camera that is not open'
        now, sdk = self._sdk.clock.now(), self._sdk
        if now < sdk.lost_at or self.opened_at >= sdk.listed_at:
            return ErrorCode.SUCCESS
        if sdk.sdk_recovers_at is not None and now >= sdk.sdk_recovers_at:
            return ErrorCode.SUCCESS
        return ErrorCode.CAMERA_REBOOTING

    def close(self) -> None:
        self.closed_at = self._sdk.clock.now()

    def get_timestamp(self, _reference) -> types.SimpleNamespace:
        return types.SimpleNamespace(get_nanoseconds=self._sdk.clock.now_ns)

    def retrieve_image(self, _image, _view) -> ErrorCode:
        return ErrorCode.SUCCESS


class FakeInitParameters:
    def set_from_serial_number(self, serial: int) -> None:
        self.serial = serial


class FakeMat:
    def get_data(self) -> np.ndarray:
        return np.zeros((2, 2, 4), dtype=np.uint8)


def _camera_loop(module, sdk: DroppingSdk, stop: StopFlag, frames: RecordingEmitter) -> pimm.Run[None]:
    """The driver's `run` loop for the camera `SERIAL`, bound to ``sdk``."""
    module.sl.__dict__.update(
        ERROR_CODE=ErrorCode,
        Camera=sdk,
        InitParameters=FakeInitParameters,
        Mat=FakeMat,
        RESOLUTION=types.SimpleNamespace(AUTO='auto'),
        VIEW=types.SimpleNamespace(LEFT='left'),
        DEPTH_MODE=types.SimpleNamespace(NONE='none'),
        UNIT=types.SimpleNamespace(METER='meter'),
        TIME_REFERENCE=types.SimpleNamespace(IMAGE='image'),
    )
    camera = module.SLCamera(serial_number=SERIAL)
    camera.frame._bind(frames)
    return camera.run(stop, sdk.clock)


def _drive(loop: pimm.Run[None], clock: MockClock, until: float) -> None:
    """Drive ``loop`` on ``clock`` until the clock reaches ``until`` or the loop returns."""
    for command in loop:
        assert isinstance(command, pimm.Sleep)
        clock.advance(command.seconds)
        if clock.now() >= until:
            return


def _returned(loop: pimm.Run[None]) -> bool:
    return inspect.getgeneratorstate(loop) == inspect.GEN_CLOSED


def _frame_times(frames: RecordingEmitter) -> list[float]:
    return [ts for ts, _ in frames.emitted]


LOST_AT = 1.0
RECOVERY_TIME = 10.0  # the driver's default `max_recovery_time_sec`
LISTED_AGAIN_AT = 18.0
NEVER = float('inf')


def test_a_camera_lost_past_the_recovery_time_is_reopened_and_its_frames_resume(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=LOST_AT, listed_at=LISTED_AGAIN_AT)
    loop = _camera_loop(zed_module, sdk, stop, frames)
    _drive(loop, clock, until=30.0)
    assert not _returned(loop)
    lost, reopened = sdk.cameras
    assert lost.closed_at is not None and lost.closed_at >= LOST_AT + RECOVERY_TIME
    assert reopened.opened_at is not None and reopened.opened_at >= LISTED_AGAIN_AT
    assert not [t for t in _frame_times(frames) if LOST_AT < t < LISTED_AGAIN_AT]
    assert max(_frame_times(frames)) > reopened.opened_at
    assert sdk.reboots == []


def test_the_loop_does_not_return_while_the_camera_stays_away(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=LOST_AT, listed_at=NEVER)
    loop = _camera_loop(zed_module, sdk, stop, frames)
    _drive(loop, clock, until=600.0)
    assert not _returned(loop)
    assert max(_frame_times(frames)) < LOST_AT + 0.02
    assert len(sdk.cameras) == 1


def test_a_camera_the_sdk_recovers_inside_the_recovery_time_is_not_reopened(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=LOST_AT, listed_at=3.0, sdk_recovers_at=LOST_AT + RECOVERY_TIME - 0.5)
    loop = _camera_loop(zed_module, sdk, stop, frames)
    _drive(loop, clock, until=30.0)
    assert not _returned(loop)
    (camera,) = sdk.cameras
    assert camera.closed_at is None
    assert max(_frame_times(frames)) > LOST_AT + RECOVERY_TIME


def test_a_reopen_that_fails_is_tried_again_until_the_camera_opens(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=LOST_AT, listed_at=LISTED_AGAIN_AT, failed_reopens=3)
    loop = _camera_loop(zed_module, sdk, stop, frames)
    _drive(loop, clock, until=40.0)
    assert not _returned(loop)
    lost, failed, reopened = sdk.cameras
    assert failed.opened_at is None and failed.closed_at is not None
    assert reopened.opened_at is not None and max(_frame_times(frames)) > reopened.opened_at


def test_the_world_stopping_while_the_camera_is_away_ends_the_loop(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=LOST_AT, listed_at=NEVER)
    loop = _camera_loop(zed_module, sdk, stop, frames)
    _drive(loop, clock, until=20.0)
    assert not _returned(loop)
    stop.stopped = True
    _drive(loop, clock, until=clock.now() + 1.0)
    assert _returned(loop)
