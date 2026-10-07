"""The ZED driver with the SDK faked: its open path, the error it keeps for a lost camera, its ready call, and the
settings it reads back."""

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
from pimm.tests.testing import MockClock, wire_call
from positronic.drivers.camera import CAPTURE_TIME
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


class VideoSettings(enum.Enum):
    EXPOSURE = enum.auto()
    GAIN = enum.auto()
    WHITEBALANCE_TEMPERATURE = enum.auto()
    AEC_AGC = enum.auto()
    WHITEBALANCE_AUTO = enum.auto()


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
    sl.__dict__.update(VIDEO_SETTINGS=VideoSettings)
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


def _opening(module, sdk: FakeSdk, reboot_serial: int | None = SERIAL, stop: StopFlag | None = None):
    _bind(module, sdk)
    zed = types.SimpleNamespace(open=sdk.open)
    return module.SLCamera._open_under_device_lock(zed, None, reboot_serial, stop or StopFlag())


def _open(module, sdk: FakeSdk, reboot_serial: int | None = SERIAL) -> list[pimm.Sleep]:
    return list(_opening(module, sdk, reboot_serial))


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


def test_the_world_stopping_while_a_rebooted_camera_is_away_ends_the_open(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 3, listed=[False])
    stop = StopFlag()
    opening = _opening(zed_module, sdk, stop=stop)
    retry_sleeps = 2
    for _ in range(retry_sleeps + 3):
        next(opening)
    stop.stopped = True
    with pytest.raises(RuntimeError, match='the world stopped'):
        next(opening)
    assert sdk.list_calls < zed_module.SLCamera.REBOOTED_CAMERA_MAX_POLLS


class DroppingSdk:
    """Stands in for `sl.Camera` over one ZED that leaves the device list at ``lost_at`` and returns at ``listed_at``.

    A camera opened before the drop grabs again only from ``sdk_recovers_at``. ``None`` is a model whose loss the SDK
    never recovers, such as the ZED Mini. The first ``failed_reopens`` opens after the drop fail. A camera opened
    before ``images_stop_at`` grabs with no error from then on, and gives no image.
    """

    def __init__(
        self,
        clock: MockClock,
        lost_at: float,
        listed_at: float,
        sdk_recovers_at: float | None = None,
        failed_reopens: int = 0,
        images_stop_at: float = float('inf'),
    ):
        self.clock = clock
        self.lost_at = lost_at
        self.listed_at = listed_at
        self.sdk_recovers_at = sdk_recovers_at
        self.failed_reopens = failed_reopens
        self.images_stop_at = images_stop_at
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
        self.settings_reads = 0

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
        assert self.opened_at is not None
        if self.opened_at < self._sdk.images_stop_at <= self._sdk.clock.now():
            return ErrorCode.FAILURE
        return ErrorCode.SUCCESS

    def get_camera_settings(self, setting: VideoSettings) -> tuple[ErrorCode, int]:
        self.settings_reads += 1
        return ErrorCode.SUCCESS, SETTINGS[setting]


SETTINGS = {
    VideoSettings.EXPOSURE: 45,
    VideoSettings.GAIN: 12,
    VideoSettings.WHITEBALANCE_TEMPERATURE: 4700,
    VideoSettings.AEC_AGC: 1,
    VideoSettings.WHITEBALANCE_AUTO: 1,
}
STATE = {
    'exposure': 45,
    'gain': 12,
    'white_balance_temperature': 4700,
    'auto_exposure_gain': 1,
    'auto_white_balance': 1,
}


class FakeInitParameters:
    def set_from_serial_number(self, serial: int) -> None:
        self.serial = serial


class FakeMat:
    def get_data(self) -> np.ndarray:
        return np.zeros((2, 2, 4), dtype=np.uint8)


LOST_AT = 1.0
LISTED_AGAIN_AT = 3.0
NEVER = float('inf')


@pytest.fixture
def world():
    with pimm.World() as w:
        yield w


def _camera(module, sdk: DroppingSdk, frames: RecordingEmitter):
    """The driver for the camera `SERIAL`, bound to ``sdk``, sending its frames to ``frames``."""
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
    camera.frame._bind(frames, clock=sdk.clock)
    return camera


def _readier(world: pimm.World, camera) -> pimm.calls.Caller[None, None]:
    caller = pimm.calls.ControlSystemCaller[None, None](camera)
    wire_call(world, caller, camera.ready)
    return caller


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
    return [time[CAPTURE_TIME] / 1e9 for time, data in frames.emitted if not isinstance(data, pimm.SignalError)]


def _errors(frames: RecordingEmitter) -> list[pimm.SignalError]:
    return [data for _, data in frames.emitted if isinstance(data, pimm.SignalError)]


def test_a_lost_camera_sends_an_error_and_keeps_it_until_it_is_asked_to_be_ready(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=LOST_AT, listed_at=LISTED_AGAIN_AT, sdk_recovers_at=LISTED_AGAIN_AT)
    loop = _camera(zed_module, sdk, frames).run(stop, clock)
    _drive(loop, clock, until=30.0)
    assert not _returned(loop)
    (camera,) = sdk.cameras
    assert camera.closed_at is None
    (error,) = _errors(frames)
    assert 'CAMERA_REBOOTING' in str(error)
    assert isinstance(frames.emitted[-1][1], pimm.SignalError)
    assert max(_frame_times(frames)) < LOST_AT


def test_a_ready_call_on_a_camera_with_a_recent_frame_is_answered_at_once(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER)
    camera = _camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=2.0)
    answer = _readier(world, camera)(None)
    _drive(loop, clock, until=clock.now() + 0.01)
    assert answer.result() is None
    assert len(sdk.cameras) == 1


def test_a_ready_call_asked_while_the_camera_opens_is_answered_by_that_open(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER)
    camera = _camera(zed_module, sdk, frames)
    answer = _readier(world, camera)(None)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=0.05)
    assert answer.result() is None
    assert len(sdk.cameras) == 1


def test_a_ready_call_reopens_a_lost_camera_and_answers_once_a_frame_arrives(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=LOST_AT, listed_at=LISTED_AGAIN_AT)
    camera = _camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=10.0)
    sent = len(_frame_times(frames))
    answer = _readier(world, camera)(None)
    _drive(loop, clock, until=clock.now() + 0.01)
    assert answer.result() is None
    lost, reopened = sdk.cameras
    assert lost.closed_at is not None and reopened.opened_at is not None and reopened.opened_at >= 10.0
    assert len(_frame_times(frames)) > sent
    assert not isinstance(frames.emitted[-1][1], pimm.SignalError)


def test_a_ready_call_reopens_a_camera_whose_images_stopped_with_no_error(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER, images_stop_at=LOST_AT)
    camera = _camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=5.0)
    assert not _errors(frames) and max(_frame_times(frames)) < LOST_AT
    sent = len(_frame_times(frames))
    answer = _readier(world, camera)(None)
    _drive(loop, clock, until=clock.now() + 0.01)
    assert answer.result() is None
    stale, _ = sdk.cameras
    assert stale.closed_at is not None and len(_frame_times(frames)) > sent


def test_a_camera_that_does_not_open_answers_ready_with_its_error_and_the_next_call_opens_it(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=LOST_AT, listed_at=NEVER)
    camera = _camera(zed_module, sdk, frames)
    ready = _readier(world, camera)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=10.0)
    refused = ready(None)
    _drive(loop, clock, until=clock.now() + 20.0)
    with pytest.raises(pimm.SignalError, match='did not open'):
        refused.result()
    assert not _returned(loop)
    assert max(_frame_times(frames)) < LOST_AT

    sdk.listed_at = clock.now()
    sent = len(_frame_times(frames))
    answer = ready(None)
    _drive(loop, clock, until=clock.now() + 0.01)
    assert answer.result() is None
    assert len(_frame_times(frames)) > sent


def test_a_camera_absent_at_start_up_sends_an_error_until_a_ready_call_opens_it(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=0.0, listed_at=5.0)
    camera = _camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=20.0)
    assert not _returned(loop)
    assert not _frame_times(frames) and len(_errors(frames)) == 1
    answer = _readier(world, camera)(None)
    _drive(loop, clock, until=clock.now() + 0.01)
    assert answer.result() is None
    assert _frame_times(frames)


def test_the_world_stopping_while_the_camera_is_away_ends_the_loop(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=LOST_AT, listed_at=NEVER)
    loop = _camera(zed_module, sdk, frames).run(stop, clock)
    _drive(loop, clock, until=20.0)
    assert not _returned(loop)
    stop.stopped = True
    _drive(loop, clock, until=clock.now() + 1.0)
    assert _returned(loop)


class SettingsCamera:
    """Answers each setting with the value it holds, and refuses the ones it does not."""

    def __init__(self, values: dict[VideoSettings, int]):
        self._values = values

    def get_camera_settings(self, setting: VideoSettings) -> tuple[ErrorCode, int]:
        if setting in self._values:
            return ErrorCode.SUCCESS, self._values[setting]
        return ErrorCode.FAILURE, -1


def test_the_state_read_reports_every_setting_the_camera_answers(zed_module):
    zed_module.sl.ERROR_CODE = ErrorCode
    assert zed_module.SLCamera._read_state(SettingsCamera(SETTINGS)) == STATE


def test_the_state_read_leaves_out_a_setting_the_camera_refuses(zed_module):
    zed_module.sl.ERROR_CODE = ErrorCode
    camera = SettingsCamera({VideoSettings.EXPOSURE: 45})
    assert zed_module.SLCamera._read_state(camera) == {'exposure': 45}


def test_the_camera_sends_its_settings_once_a_period_while_something_receives_them(zed_module):
    clock, stop, frames, states = MockClock(), StopFlag(), RecordingEmitter(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER)
    camera = _camera(zed_module, sdk, frames)
    camera.state._bind(states, clock=clock)
    _drive(camera.run(stop, clock), clock, until=2.5)
    assert [data for _, data in states.emitted] == [STATE] * 3


def test_a_camera_whose_settings_nothing_receives_does_not_read_them(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER)
    _drive(_camera(zed_module, sdk, frames).run(stop, clock), clock, until=2.5)
    (camera,) = sdk.cameras
    assert camera.settings_reads == 0


def test_a_ready_call_that_reopens_the_camera_sends_its_settings_before_it_answers(zed_module, world):
    clock, stop, frames, states = MockClock(), StopFlag(), RecordingEmitter(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=LOST_AT, listed_at=LISTED_AGAIN_AT)
    camera = _camera(zed_module, sdk, frames)
    camera.state._bind(states, clock=clock)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=10.0)
    answer = _readier(world, camera)(None)
    _drive(loop, clock, until=clock.now() + 0.001)
    assert answer.result() is None
    _, reopened = sdk.cameras
    assert reopened.settings_reads == len(STATE)
