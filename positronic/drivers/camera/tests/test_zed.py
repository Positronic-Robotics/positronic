"""The ZED driver with the SDK faked: its open path, the error it keeps for a lost camera, and its ready call."""

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
from positronic.drivers.faults import DeviceFault, FaultKind
from positronic.drivers.roboarm.tests.fakes import StopFlag
from positronic.drivers.utils import RecoveryOutcome
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


def _noisy_scene(low: int, high: int, shape: tuple[int, int] = (72, 128)) -> np.ndarray:
    """A BGRA image whose colour values spread at random from ``low`` up to ``high``, with opaque alpha."""
    rng = np.random.default_rng(0)
    scene = np.full((*shape, 4), 255, dtype=np.uint8)
    scene[..., :3] = rng.integers(low, high, size=(*shape, 3), dtype=np.uint8)
    return scene


class FakeModel(enum.Enum):
    ZED_M = 'ZED_M'
    ZED2 = 'ZED2'
    ZED2i = 'ZED2i'


class DroppingSdk:
    """Stands in for `sl.Camera` over one ZED of ``model`` that leaves the device list at ``lost_at`` and returns at
    ``listed_at``.

    A camera opened before the drop grabs again only from ``sdk_recovers_at``. ``None`` is a model whose loss the SDK
    never recovers, such as the ZED Mini. The first ``failed_reopens`` opens after the drop fail. A camera opened
    before ``images_stop_at`` grabs with no error from then on, and gives no image. Every image shows ``scene``, a lit
    room by default, and the grabs that ``torn_grabs`` counts from 0 give a torn image.
    """

    def __init__(
        self,
        clock: MockClock,
        lost_at: float,
        listed_at: float,
        sdk_recovers_at: float | None = None,
        failed_reopens: int = 0,
        images_stop_at: float = float('inf'),
        scene: np.ndarray | None = None,
        torn_grabs: frozenset[int] = frozenset(),
        model: FakeModel = FakeModel.ZED2i,
    ):
        self.model = model
        self.clock = clock
        self.lost_at = lost_at
        self.listed_at = listed_at
        self.sdk_recovers_at = sdk_recovers_at
        self.failed_reopens = failed_reopens
        self.images_stop_at = images_stop_at
        self.scene = _noisy_scene(60, 160) if scene is None else scene
        self.torn_grabs = torn_grabs
        self.grab_times: list[float] = []
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
        sdk.grab_times.append(now)
        if now < sdk.lost_at or self.opened_at >= sdk.listed_at:
            return ErrorCode.SUCCESS
        if sdk.sdk_recovers_at is not None and now >= sdk.sdk_recovers_at:
            return ErrorCode.SUCCESS
        return ErrorCode.CAMERA_REBOOTING

    def close(self) -> None:
        self.closed_at = self._sdk.clock.now()

    def get_camera_information(self) -> types.SimpleNamespace:
        assert self.opened_at is not None and self.closed_at is None, 'camera information of a camera that is not open'
        return types.SimpleNamespace(camera_model=self._sdk.model)

    def get_timestamp(self, _reference) -> types.SimpleNamespace:
        return types.SimpleNamespace(get_nanoseconds=self._sdk.clock.now_ns)

    def retrieve_image(self, image: 'FakeMat', _view) -> ErrorCode:
        assert self.opened_at is not None
        if self.opened_at < self._sdk.images_stop_at <= self._sdk.clock.now():
            return ErrorCode.FAILURE
        torn = len(self._sdk.grab_times) - 1 in self._sdk.torn_grabs
        image.data = _torn(self._sdk.scene) if torn else self._sdk.scene
        return ErrorCode.SUCCESS


class FakeInitParameters:
    def set_from_serial_number(self, serial: int) -> None:
        self.serial = serial


class FakeMat:
    data: np.ndarray

    def get_data(self) -> np.ndarray:
        return self.data


# The BGRA pixel that the SDK gives a raw pixel of zeros, as a ZED Mini sent it at the start of a torn frame
ZERO_FILL_BGRA = (0, 135, 0, 255)


def _torn(scene: np.ndarray) -> np.ndarray:
    """``scene`` as a torn frame: two rows of zero fill, and the rest shifted down."""
    torn = np.roll(scene, 2, axis=0)
    torn[:2] = ZERO_FILL_BGRA
    return torn


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
        VIEW=types.SimpleNamespace(LEFT='left', LEFT_UNRECTIFIED='left_unrectified'),
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
    assert isinstance(error, DeviceFault) and error.kind is FaultKind.CAMERA
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


def test_a_ready_call_on_a_reopened_camera_that_gives_no_image_answers_a_camera_fault(zed_module, world, monkeypatch):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    camera = _camera(zed_module, DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER), frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=1.0)
    monkeypatch.setattr(FakeCamera, 'retrieve_image', lambda _camera, _image, _view: ErrorCode.FAILURE)
    _drive(loop, clock, until=clock.now() + 2 * zed_module.SLCamera.STALE_FRAME_SEC)

    answer = _readier(world, camera)(None)
    _drive(loop, clock, until=clock.now() + 0.01)

    with pytest.raises(DeviceFault, match='opened and sent no frame') as raised:
        answer.result()
    assert raised.value.kind is FaultKind.CAMERA
    assert _errors(frames) == [raised.value]


def test_a_camera_that_does_not_open_answers_ready_with_its_error_and_the_next_call_opens_it(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=LOST_AT, listed_at=NEVER)
    camera = _camera(zed_module, sdk, frames)
    ready = _readier(world, camera)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=10.0)
    refused = ready(None)
    _drive(loop, clock, until=clock.now() + 20.0)
    with pytest.raises(DeviceFault, match='did not open') as raised:
        refused.result()
    assert raised.value.kind is FaultKind.CAMERA
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


# A rebooted camera stays on the device list for this long, then leaves it for `REBOOT_AWAY_S`.
REBOOT_LEAVES_AFTER_S = 1.0
REBOOT_AWAY_S = 4.0


class RebootingSdk(DroppingSdk):
    """A `DroppingSdk` over one camera of ``model`` that the SDK reboots.

    The SDK refuses the reboot while a camera object holds the device open. The device list leaves out a camera that a
    camera object holds open, as the SDK before 5.4 does, unless ``lists_open_camera``. A reboot that runs leaves a
    listed camera on the device list for ``leaves_after_s``, then off it for ``away_s``.
    """

    def __init__(
        self,
        clock: MockClock,
        model: FakeModel = FakeModel.ZED2i,
        reboot_result: ErrorCode = ErrorCode.SUCCESS,
        leaves_after_s: float = REBOOT_LEAVES_AFTER_S,
        away_s: float = REBOOT_AWAY_S,
        images_stop_at: float = NEVER,
        lists_open_camera: bool = False,
    ):
        super().__init__(clock, lost_at=NEVER, listed_at=NEVER, images_stop_at=images_stop_at, model=model)
        self.reboot_result = reboot_result
        self.leaves_after_s = leaves_after_s
        self.away_s = away_s
        self.lists_open_camera = lists_open_camera
        self.reboot_times: list[float] = []

    def _held_open(self) -> bool:
        return any(camera.opened_at is not None and camera.closed_at is None for camera in self.cameras)

    def get_device_list(self) -> list[types.SimpleNamespace]:
        listed = types.SimpleNamespace(serial_number=SERIAL, camera_model=self.model)
        shown = self.is_listed() and (self.lists_open_camera or not self._held_open())
        return [listed] if shown else []

    def reboot(self, serial: int) -> ErrorCode:
        self.reboots.append(serial)
        if self._held_open():
            return ErrorCode.FAILURE
        if self.reboot_result is not ErrorCode.SUCCESS:
            return self.reboot_result
        now = self.clock.now()
        self.reboot_times.append(now)
        if self.is_listed():
            self.lost_at = now + self.leaves_after_s
        self.listed_at = max(self.lost_at, now) + self.away_s
        return ErrorCode.SUCCESS


def _rebooting_camera(module, sdk: RebootingSdk, frames: RecordingEmitter, mono: bool = False):
    """The driver for the camera `SERIAL` over ``sdk``, a mono one where ``mono`` says so."""
    stereo = _camera(module, sdk, frames)
    module.sl.__dict__.update(MODEL=FakeModel, CameraOne=sdk, InitParametersOne=FakeInitParameters)
    if not mono:
        return stereo
    camera = module.SLCamera(serial_number=SERIAL, mono=True)
    camera.frame._bind(frames, clock=sdk.clock)
    return camera


def _recoverer(world: pimm.World, camera) -> pimm.calls.Caller[None, RecoveryOutcome]:
    caller = pimm.calls.ControlSystemCaller[None, RecoveryOutcome](camera)
    wire_call(world, caller, camera.recover)
    return caller


def _drive_until_answered(loop: pimm.Run[None], clock: MockClock, answer, limit_s: float = 300.0) -> None:
    """Drive ``loop`` until ``answer`` is done, for at most ``limit_s`` seconds of the clock."""
    until = clock.now() + limit_s
    while not answer.done() and clock.now() < until:
        _drive(loop, clock, until=clock.now() + 0.01)


def test_a_recover_call_releases_the_camera_then_reboots_it_then_opens_it_again(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = RebootingSdk(clock)
    camera = _rebooting_camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=2.0)
    sent = len(_frame_times(frames))

    answer = _recoverer(world, camera)(None)
    _drive_until_answered(loop, clock, answer)

    assert answer.result() is RecoveryOutcome.CLEARED
    (rebooted_at,) = sdk.reboot_times
    released, reopened = sdk.cameras
    assert released.closed_at is not None and released.closed_at <= rebooted_at
    assert reopened.opened_at is not None and reopened.opened_at >= sdk.listed_at
    assert len(_frame_times(frames)) > sent
    assert not _errors(frames)


def test_a_recover_call_whose_reboot_the_sdk_refuses_opens_the_camera_again_and_answers_not_cleared(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = RebootingSdk(clock, reboot_result=ErrorCode.FAILURE)
    camera = _rebooting_camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=2.0)
    sent = len(_frame_times(frames))

    answer = _recoverer(world, camera)(None)
    _drive_until_answered(loop, clock, answer)

    assert answer.result() is RecoveryOutcome.NOT_CLEARED
    assert sdk.reboots == [SERIAL]
    _, reopened = sdk.cameras
    assert reopened.opened_at is not None and reopened.closed_at is None
    assert len(_frame_times(frames)) > sent


def test_a_recover_call_whose_camera_never_leaves_the_device_list_answers_not_cleared(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = RebootingSdk(clock, leaves_after_s=NEVER)
    camera = _rebooting_camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=2.0)
    sent = len(_frame_times(frames))

    answer = _recoverer(world, camera)(None)
    _drive_until_answered(loop, clock, answer)

    assert answer.result() is RecoveryOutcome.NOT_CLEARED
    assert len(sdk.reboot_times) == 1
    _, reopened = sdk.cameras
    assert reopened.opened_at is not None and reopened.closed_at is None
    assert len(_frame_times(frames)) > sent


def test_a_recover_call_on_a_camera_that_does_not_come_back_answers_a_camera_fault_and_holds_it(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = RebootingSdk(clock, away_s=NEVER)
    camera = _rebooting_camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=2.0)

    answer = _recoverer(world, camera)(None)
    _drive_until_answered(loop, clock, answer)

    with pytest.raises(DeviceFault, match='did not open') as raised:
        answer.result()
    assert raised.value.kind is FaultKind.CAMERA
    assert _errors(frames) == [raised.value]
    assert not _returned(loop)


@pytest.mark.parametrize(
    ('images_stop_at', 'outcome'),
    [(NEVER, RecoveryOutcome.CLEARED), (LOST_AT, RecoveryOutcome.NOT_CLEARED)],
    ids=['sending', 'stale'],
)
@pytest.mark.parametrize('lists_open_camera', [True, False], ids=['listed_while_open', 'unlisted_while_open'])
def test_a_recover_call_does_not_touch_a_model_the_sdk_cannot_reboot(
    zed_module, world, images_stop_at, outcome, lists_open_camera
):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = RebootingSdk(clock, model=FakeModel.ZED_M, images_stop_at=images_stop_at, lists_open_camera=lists_open_camera)
    camera = _rebooting_camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=5.0)

    answer = _recoverer(world, camera)(None)
    _drive_until_answered(loop, clock, answer)

    assert answer.result() is outcome
    assert sdk.reboots == []
    (held,) = sdk.cameras
    assert held.closed_at is None


def test_a_recover_call_reboots_a_camera_that_never_opened_and_that_the_sdk_does_not_list(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = RebootingSdk(clock, model=FakeModel.ZED_M, reboot_result=ErrorCode.INVALID_FUNCTION_CALL)
    sdk.lost_at = 0.0
    camera = _rebooting_camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=5.0)
    reboots_of_the_open = len(sdk.reboots)

    answer = _recoverer(world, camera)(None)
    _drive_until_answered(loop, clock, answer)

    assert len(sdk.reboots) > reboots_of_the_open
    with pytest.raises(DeviceFault, match='did not open'):
        answer.result()


def test_a_recover_call_does_not_reboot_a_mono_camera(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = RebootingSdk(clock)
    camera = _rebooting_camera(zed_module, sdk, frames, mono=True)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=2.0)

    answer = _recoverer(world, camera)(None)
    _drive_until_answered(loop, clock, answer)

    assert answer.result() is RecoveryOutcome.CLEARED
    assert sdk.reboots == []
    (held,) = sdk.cameras
    assert held.closed_at is None


def test_a_torn_frame_is_dropped_with_no_error_and_the_next_frame_is_sent(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER, torn_grabs=frozenset({3, 7, 8}))
    _drive(_camera(zed_module, sdk, frames).run(stop, clock), clock, until=0.2)
    assert not _errors(frames)
    sent = [t for i, t in enumerate(sdk.grab_times) if i not in sdk.torn_grabs]
    assert _frame_times(frames) == pytest.approx(sent)


def test_a_ready_call_that_reopens_the_camera_grabs_again_after_a_torn_frame(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER)
    camera = _camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=1.0)
    sdk.images_stop_at = clock.now()
    _drive(loop, clock, until=clock.now() + 2 * zed_module.SLCamera.STALE_FRAME_SEC)

    answer = _readier(world, camera)(None)
    stale_grab = len(sdk.grab_times)
    sdk.torn_grabs = frozenset({stale_grab + 1, stale_grab + 2})
    _drive(loop, clock, until=clock.now() + 0.01)

    assert answer.result() is None
    assert not _errors(frames)
    assert len(sdk.cameras) == 2


@pytest.mark.parametrize(
    'scene',
    [np.full((720, 1280, 4), (0, 0, 0, 255), np.uint8), _noisy_scene(0, 12, (720, 1280))],
    ids=['black', 'dark'],
)
def test_a_black_or_dark_frame_is_sent(zed_module, scene):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER, scene=scene)
    _drive(_camera(zed_module, sdk, frames).run(stop, clock), clock, until=0.2)
    assert _frame_times(frames) == pytest.approx(sdk.grab_times)


# The mean levels of these images are near 6 of 255, as a failing ZED sends, near 32 for a dim room, and near 110.
NEAR_BLACK = _noisy_scene(0, 12)
DIM = _noisy_scene(16, 48)
LIT = _noisy_scene(60, 160)
NEAR_BLACK_FOR_LONG = 30.0


def test_images_that_stay_near_black_send_a_camera_fault_that_the_camera_keeps(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER, scene=LIT)
    loop = _camera(zed_module, sdk, frames).run(stop, clock)
    _drive(loop, clock, until=1.0)
    sdk.scene, black_from = NEAR_BLACK, clock.now()
    _drive(loop, clock, until=black_from + NEAR_BLACK_FOR_LONG)

    (error,) = _errors(frames)
    assert isinstance(error, DeviceFault) and error.kind is FaultKind.CAMERA
    assert 'near-black' in str(error)
    assert isinstance(frames.emitted[-1][1], pimm.SignalError)
    assert max(_frame_times(frames)) == pytest.approx(black_from + zed_module.SLCamera.NEAR_BLACK_FAULT_SEC, abs=0.05)


def test_near_black_stretches_shorter_than_the_limit_send_no_fault(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER, scene=LIT)
    loop = _camera(zed_module, sdk, frames).run(stop, clock)
    short = zed_module.SLCamera.NEAR_BLACK_FAULT_SEC - 1.0
    for scene, seconds in [(NEAR_BLACK, short), (LIT, 0.1), (NEAR_BLACK, short), (LIT, 1.0)]:
        sdk.scene = scene
        _drive(loop, clock, until=clock.now() + seconds)
    assert not _errors(frames)
    assert _frame_times(frames) == pytest.approx(sdk.grab_times)


def test_dim_images_send_no_fault(zed_module):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER, scene=DIM)
    _drive(_camera(zed_module, sdk, frames).run(stop, clock), clock, until=NEAR_BLACK_FOR_LONG)
    assert not _errors(frames)
    assert _frame_times(frames) == pytest.approx(sdk.grab_times)


def test_a_camera_reopened_after_a_near_black_fault_gets_the_full_limit_again(zed_module, world):
    clock, stop, frames = MockClock(), StopFlag(), RecordingEmitter()
    sdk = DroppingSdk(clock, lost_at=NEVER, listed_at=NEVER, scene=NEAR_BLACK)
    camera = _camera(zed_module, sdk, frames)
    loop = camera.run(stop, clock)
    _drive(loop, clock, until=NEAR_BLACK_FOR_LONG)
    assert len(_errors(frames)) == 1

    answer = _readier(world, camera)(None)
    _drive(loop, clock, until=clock.now() + 0.01)
    assert answer.result() is None
    reopened_at = sdk.cameras[-1].opened_at
    _drive(loop, clock, until=clock.now() + NEAR_BLACK_FOR_LONG)

    assert len(_errors(frames)) == 2
    assert max(_frame_times(frames)) == pytest.approx(reopened_at + zed_module.SLCamera.NEAR_BLACK_FAULT_SEC, abs=0.05)
