import logging
from collections.abc import Generator, Iterator
from enum import Enum, auto
from typing import Literal

import numpy as np

import pimm
from pimm.shared_memory import NumpySMAdapter
from positronic.drivers import vendor_import
from positronic.drivers.camera import CAPTURE_TIME
from positronic.drivers.camera.device_open_lock import device_open_lock

with vendor_import('pyzed', 'ZED camera support', platforms=('linux',)):
    import pyzed.sl as sl

logger = logging.getLogger(__name__)


class CameraOpenError(RuntimeError):
    """The SDK did not open the camera."""


class GrabOutcome(Enum):
    """What one grab of the camera did."""

    SENT = auto()
    NO_IMAGE = auto()  # the grab succeeded, and the SDK gave no image
    LOST = auto()  # the grab failed, and the camera holds the error


class SLCamera(pimm.ControlSystem):
    def __init__(
        self,
        serial_number: int | None = None,
        fps: int | None = None,
        view: Literal['left', 'right', 'side_by_side'] = 'left',
        resolution: Literal[
            'hd4k', 'qhdplus', 'hd2k', 'hd1080', 'hd1200', 'hd1536', 'hd720', 'svga', 'vga', 'auto'
        ] = 'auto',
        depth_mode: Literal['none', 'near', 'far', 'high', 'ultra'] = 'none',
        max_depth: float = 10,
        depth_mask: bool = False,
        image_enhancement: bool = False,
        mono: bool = False,
        state_period_sec: float = 1.0,
    ):
        """
        StereoLabs camera driver.

        Args:
            fps: (int) Frames per second
            view: (sl.VIEW) View to use
            resolution: (sl.RESOLUTION) Resolution to use
            depth_mode: (sl.DEPTH_MODE) Depth mode to use
            coordinate_units: (sl.UNIT) Coordinate units to use
            max_depth: (float) Maximum depth to use. Depth NaNs and +Inf will be set to this distance.
                        -Inf will be set to 0. All values above this will be set to max_depth.
            depth_mask: (bool) If True, will also generate image with 0 set to NaNs pixels, and 1 set to valid pixels
            mono: (bool) Open a single-sensor camera (e.g. ZED X One) via ``sl.CameraOne``. Mono cameras
                  support only ``view='left'``, ``depth_mode='none'`` and no image enhancement.
            state_period_sec: (float) How often ``state`` reports the exposure, gain and white balance the
                  camera runs at. Automatic control moves them as the scene changes, so one reading per
                  episode is not enough; each reading is a control request to the camera, so once a frame
                  is too many.
        """
        super().__init__()
        # IMPORTANT: This control system may be spawned under multiprocessing "spawn".
        # Keep only plain-Python config on self so the instance is picklable; construct
        # pyzed objects inside `run()`.
        self._serial_number = serial_number
        self._fps = fps
        self._view_name = view
        self._resolution_name = resolution
        self._depth_mode_name = depth_mode
        self._image_enhancement = image_enhancement
        self._depth_mask_requested = depth_mask
        self._mono = mono
        self._state_period_sec = state_period_sec

        self.max_depth = max_depth

        # Main frame channel (always present)
        self.frame: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)
        self._frame_adapter = None  # Lazy init

        # Depth channels (always available for connection, but checked at runtime)
        self.depth: pimm.ControlSystemEmitter = pimm.ControlSystemEmitter(self)
        self._depth_adapter = None  # Lazy init

        self.depth_mask: pimm.ControlSystemEmitter = pimm.ControlSystemEmitter(self)
        self._depth_mask_adapter = None  # Lazy init

        # The exposure, gain and white balance the camera runs at, read back from it every ``state_period_sec``
        # while something receives them.
        self.state: pimm.ControlSystemEmitter[dict[str, int]] = pimm.ControlSystemEmitter(self)
        self._state_due_at = float('-inf')

        self.ready = pimm.calls.ControlSystemHandler[None, None](self)
        # The camera a run holds open, the error it keeps until a ready call repairs the camera, and the time of
        # its last frame
        self._camera: sl.Camera | sl.CameraOne | None = None
        self._error: pimm.SignalError | None = None
        self._frame_at: float | None = None

    DEVICE_LIST_POLL_SEC = 0.5
    REBOOTED_CAMERA_MAX_POLLS = 60
    # A ready call reopens a camera whose last frame is older than this.
    STALE_FRAME_SEC = 1.0

    @staticmethod
    def _is_listed(serial: int) -> bool:
        with device_open_lock():
            devices = sl.Camera.get_device_list()
        return any(device.serial_number == serial for device in devices)

    @staticmethod
    def _lost_by_the_sdk(error_code, serial: int) -> bool:
        return error_code == sl.ERROR_CODE.CAMERA_NOT_DETECTED or not SLCamera._is_listed(serial)

    @staticmethod
    def _reboot_and_reopen(
        zed, init_params, serial: int, failure: str, should_stop: pimm.SignalReceiver
    ) -> Iterator[pimm.Sleep]:
        """Reboot the camera over its HID half, with no replug, wait until the SDK lists it again, and open it once.

        The SDK refuses the reboot for a model that does not support it, such as the ZED Mini.
        """
        logger.warning(f'The SDK cannot find camera {serial}; rebooting it')
        with device_open_lock():
            result = sl.Camera.reboot(serial)
        if result != sl.ERROR_CODE.SUCCESS:
            raise CameraOpenError(f'{failure}; the SDK did not reboot camera {serial}: {result}')
        for _ in range(SLCamera.REBOOTED_CAMERA_MAX_POLLS):
            if SLCamera._is_listed(serial):
                break
            if should_stop.value:
                raise CameraOpenError(f'{failure}; the world stopped before camera {serial} was listed again')
            yield pimm.Sleep(SLCamera.DEVICE_LIST_POLL_SEC)
        else:
            raise CameraOpenError(f'{failure}; camera {serial} is still not listed after its reboot')
        logger.info(f'Camera {serial} is listed again after its reboot')
        with device_open_lock():
            error_code = zed.open(init_params)
        if error_code != sl.ERROR_CODE.SUCCESS:
            raise CameraOpenError(f'Failed to open camera {serial} after its reboot: {error_code}')
        logger.info(f'Opened camera {serial} after its reboot')

    @staticmethod
    def _open_under_device_lock(
        zed, init_params, reboot_serial: int | None, should_stop: pimm.SignalReceiver
    ) -> Iterator[pimm.Sleep]:
        """Open the camera, retrying: the lock binds only openers that take it, so an open can still lose the bus.

        When the retries fail and the SDK cannot find the camera ``reboot_serial`` names, the camera is rebooted once
        and opened once more.
        """
        OPEN_ATTEMPTS = 3
        OPEN_RETRY_SEC = 1.0
        for attempt in range(1, OPEN_ATTEMPTS + 1):
            with device_open_lock():
                error_code = zed.open(init_params)
            if error_code == sl.ERROR_CODE.SUCCESS:
                return
            logger.error(f'Failed to open camera (attempt {attempt} of {OPEN_ATTEMPTS}): {error_code}')
            if attempt < OPEN_ATTEMPTS and not should_stop.value:
                yield pimm.Sleep(OPEN_RETRY_SEC)
                continue
            failure = f'Failed to open camera after {attempt} attempts: {error_code}'
            if should_stop.value or reboot_serial is None or not SLCamera._lost_by_the_sdk(error_code, reboot_serial):
                raise CameraOpenError(failure)
            yield from SLCamera._reboot_and_reopen(zed, init_params, reboot_serial, failure, should_stop)

    def _open_camera(self, should_stop: pimm.SignalReceiver) -> Generator[pimm.Sleep, None, 'sl.Camera | sl.CameraOne']:
        """Open a new camera object through the open path. Raise ``CameraOpenError`` when it does not open."""
        zed = sl.CameraOne() if self._mono else sl.Camera()
        # The serial `sl.Camera` lists and reboots. A mono camera is an `sl.CameraOne`, so it has none here.
        reboot_serial = None if self._mono else self._serial_number
        try:
            yield from self._open_under_device_lock(zed, self._init_params(), reboot_serial, should_stop)
        except CameraOpenError:
            zed.close()
            raise
        return zed

    def _hold(self, error: pimm.SignalError) -> pimm.SignalError:
        """Keep ``error`` until a ready call repairs the camera, and send it on every channel in place of data."""
        logger.error('Camera %s holds an error until it is asked to be ready: %s', self._serial_number, error)
        self._error = error
        for emitter in (self.frame, self.depth, self.depth_mask):
            emitter.emit(error)
        return error

    def _open_or_hold(self, should_stop: pimm.SignalReceiver) -> Generator[pimm.Sleep, None, None]:
        """Open a new camera and clear the error, or hold the error that the open path raises."""
        try:
            self._camera = yield from self._open_camera(should_stop)
        except CameraOpenError as e:
            self._hold(pimm.SignalError(f'Camera {self._serial_number} did not open: {e}'))
            return
        self._error = None

    def _make_ready(
        self, call: pimm.calls.Call[None, None], clock: pimm.Clock, should_stop: pimm.SignalReceiver
    ) -> Generator[pimm.Sleep, None, None]:
        """Answer ``call`` at once while the last frame is recent.

        Otherwise reopen the camera, and answer once it sends a frame and its settings, or with the error that it
        holds.
        """
        if self._error is None and self._frame_at is not None and clock.now() - self._frame_at <= self.STALE_FRAME_SEC:
            call.set_result(None)
            return
        logger.info('Reopening camera %s to make it ready', self._serial_number)
        if self._camera is not None:
            self._camera.close()
            self._camera = None
        yield from self._open_or_hold(should_stop)
        if self._error is None and self._grab_frame(clock) is GrabOutcome.SENT:
            self._emit_state(clock)
            call.set_result(None)
            return
        error = self._error
        if error is None:
            error = self._hold(pimm.SignalError(f'Camera {self._serial_number} opened and sent no frame'))
        call.set_exception(error)

    def _init_params(self):
        init_params = sl.InitParametersOne() if self._mono else sl.InitParameters()
        init_params.camera_resolution = getattr(sl.RESOLUTION, self._resolution_name.upper())
        if self._fps is not None:
            init_params.camera_fps = self._fps
        if self._serial_number is not None:
            init_params.set_from_serial_number(self._serial_number)
        init_params.coordinate_units = sl.UNIT.METER
        init_params.sdk_verbose = 1
        init_params.async_grab_camera_recovery = True
        if not self._mono:
            init_params.depth_mode = self._depth_mode
            init_params.enable_image_enhancement = self._image_enhancement
        return init_params

    def _emit_depth(self, zed, capture_time: pimm.Time) -> None:
        depth = sl.Mat()
        if zed.retrieve_measure(depth, sl.MEASURE.DEPTH) != sl.ERROR_CODE.SUCCESS:
            return
        depth_data = depth.get_data()

        # Process and emit depth mask if connected
        if self.depth_mask.num_bound > 0:
            depth_mask = np.nan_to_num(depth_data, nan=0, posinf=0, neginf=0)
            depth_mask[depth_mask != 0] = 255

            self._depth_mask_adapter = NumpySMAdapter.lazy_init(
                depth_mask.astype(np.uint8)[..., np.newaxis], self._depth_mask_adapter
            )
            self.depth_mask.emit(self._depth_mask_adapter, time=capture_time)

        # Process and emit depth if connected
        if self.depth.num_bound > 0:
            depth_data = np.nan_to_num(depth_data, copy=False, nan=self.max_depth, posinf=self.max_depth, neginf=0)
            depth_data = depth_data.clip(max=self.max_depth) / self.max_depth * 255
            depth_uint8 = depth_data.astype(np.uint8)[..., np.newaxis]

            self._depth_adapter = NumpySMAdapter.lazy_init(depth_uint8, self._depth_adapter)
            self.depth.emit(self._depth_adapter, time=capture_time)

    @property
    def _view(self):
        return getattr(sl.VIEW, self._view_name.upper())

    @property
    def _depth_mode(self):
        return getattr(sl.DEPTH_MODE, self._depth_mode_name.upper())

    def _grab_frame(self, clock: pimm.Clock) -> GrabOutcome:
        """Grab and send one frame. Hold the error when the grab fails."""
        camera = self._camera
        assert camera is not None, 'a camera that holds no error is open'
        result = camera.grab()
        if result != sl.ERROR_CODE.SUCCESS:
            self._hold(pimm.SignalError(f'Camera {self._serial_number} is lost: {result}'))
            return GrabOutcome.LOST
        image = sl.Mat()
        capture_time = pimm.Time(**{CAPTURE_TIME: camera.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()})
        if camera.retrieve_image(image, self._view) != sl.ERROR_CODE.SUCCESS:
            return GrabOutcome.NO_IMAGE
        # The images are in BGRA format, convert to RGB
        np_image = image.get_data()[:, :, [2, 1, 0]]

        # Emit main frame (either single view or side-by-side)
        # Note: For side-by-side, we emit the full (H, W*2, 3) image
        # Consumer is responsible for splitting if needed
        self._frame_adapter = NumpySMAdapter.lazy_init(np_image, self._frame_adapter)
        self.frame.emit(self._frame_adapter, time=capture_time)
        self._frame_at = clock.now()

        # Only retrieve depth data if depth is enabled and at least one depth channel is connected
        if self._depth_mode != sl.DEPTH_MODE.NONE and (self.depth.num_bound > 0 or self.depth_mask.num_bound > 0):
            self._emit_depth(camera, capture_time)
        return GrabOutcome.SENT

    # The settings a camera's automatic control moves, keyed by the name each records under. ``auto_exposure_gain``
    # is the SDK's one switch for automatic exposure and gain together, and ``auto_white_balance`` is its switch for
    # the white balance; each says whether the values it governs are the sensor's own choice or a set point.
    STATE_SETTINGS = {
        'exposure': sl.VIDEO_SETTINGS.EXPOSURE,
        'gain': sl.VIDEO_SETTINGS.GAIN,
        'white_balance_temperature': sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE,
        'auto_exposure_gain': sl.VIDEO_SETTINGS.AEC_AGC,
        'auto_white_balance': sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO,
    }

    @staticmethod
    def _read_state(camera) -> dict[str, int]:
        """What ``camera`` reports for each of ``STATE_SETTINGS`` now. A setting the SDK refuses is left out."""
        state = {}
        for name, setting in SLCamera.STATE_SETTINGS.items():
            error_code, value = camera.get_camera_settings(setting)
            if error_code == sl.ERROR_CODE.SUCCESS:
                state[name] = int(value)
        return state

    def _emit_state(self, clock: pimm.Clock) -> None:
        """Send the settings the open camera runs at, and read them next ``state_period_sec`` later."""
        if self.state.num_bound == 0:
            return
        assert self._camera is not None, 'a camera that holds no error is open'
        self.state.emit(self._read_state(self._camera))
        self._state_due_at = clock.now() + self._state_period_sec

    def _emit_state_when_due(self, clock: pimm.Clock) -> None:
        if self._error is None and clock.now() >= self._state_due_at:
            self._emit_state(clock)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        fps_counter = pimm.utils.RateCounter('Camera')

        if self._mono and (self._view_name != 'left' or self._depth_mode_name != 'none' or self._image_enhancement):
            raise RuntimeError('mono cameras support only view="left", depth_mode="none" and no image enhancement')

        depth_mask_enabled = self._depth_mode != sl.DEPTH_MODE.NONE and self._depth_mask_requested

        # Runtime validation: check if depth channels are connected but not enabled
        if self.depth.num_bound > 0 and self._depth_mode == sl.DEPTH_MODE.NONE:
            raise RuntimeError(
                'depth channel is connected but depth_mode is "none". '
                'Set depth_mode to "near", "far", "high", or "ultra" to enable depth.'
            )

        if self.depth_mask.num_bound > 0 and not depth_mask_enabled:
            raise RuntimeError(
                'depth_mask channel is connected but depth_mask parameter is False. '
                'Set depth_mask=True to enable depth mask output.'
            )

        yield from self._open_or_hold(should_stop)
        while not should_stop.value:
            # Grab first, so a ready call asked while the camera opened finds the frame that open gave.
            if self._error is None:
                self._grab_frame(clock)
                fps_counter.tick()
                self._emit_state_when_due(clock)
            for call in self.ready.incoming():
                yield from self._make_ready(call, clock, should_stop)
            yield pimm.Sleep(0.01)
        if self._camera is not None:
            self._camera.close()
