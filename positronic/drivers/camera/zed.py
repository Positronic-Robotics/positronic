import logging
from collections.abc import Iterator
from typing import Literal

import numpy as np
import pyzed.sl as sl

import pimm
from pimm.shared_memory import NumpySMAdapter


# TODO: Currently in order to have pyzed available, one need to install Stereolabs SDK, and then generate
# wheel file from it. We need to find a better solution how to install it via uv or at least Docker.
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
        max_recovery_time_sec: float = 10,
        image_enhancement: bool = False,
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
            max_recovery_time_sec: (float) Maximum time to wait for camera recovery. If exceeded, will stop the camera.
        """
        super().__init__()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = getattr(sl.RESOLUTION, resolution.upper())
        if fps is not None:
            self.init_params.camera_fps = fps
        if serial_number is not None:
            inpt = sl.InputType()
            inpt.set_from_serial_number(serial_number)
            self.init_params.input = inpt

        self.view = getattr(sl.VIEW, view.upper())
        self.init_params.depth_mode = getattr(sl.DEPTH_MODE, depth_mode.upper())
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.sdk_verbose = 1
        self.init_params.enable_image_enhancement = image_enhancement
        self.init_params.async_grab_camera_recovery = True

        self.max_depth = max_depth
        self.depth_mask_enabled = self.init_params.depth_mode != sl.DEPTH_MODE.NONE and depth_mask
        self.max_recovery_time_sec = max_recovery_time_sec

        # Main frame channel (always present)
        self.frame: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)
        self._frame_adapter = None  # Lazy init

        # Depth channels (always available for connection, but checked at runtime)
        self.depth: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)
        self._depth_adapter = None  # Lazy init

        self.depth_mask: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)
        self._depth_mask_adapter = None  # Lazy init

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        SUCCESS = sl.ERROR_CODE.SUCCESS
        TIME_REF_IMAGE = sl.TIME_REFERENCE.IMAGE
        fps_counter = pimm.utils.RateCounter('Camera')

        # Runtime validation: check if depth channels are connected but not enabled
        if self.depth.num_bound > 0 and self.init_params.depth_mode == sl.DEPTH_MODE.NONE:
            raise RuntimeError(
                'depth channel is connected but depth_mode is "none". '
                'Set depth_mode to "near", "far", "high", or "ultra" to enable depth.'
            )

        if self.depth_mask.num_bound > 0 and not self.depth_mask_enabled:
            raise RuntimeError(
                'depth_mask channel is connected but depth_mask parameter is False. '
                'Set depth_mask=True to enable depth mask output.'
            )

        zed = sl.Camera()
        error_code = zed.open(self.init_params)
        if error_code != SUCCESS:
            print(f'Failed to open camera: {error_code}')
            return

        self.recovery_start_time = None

        while not should_stop.value:
            result = zed.grab()
            if result != SUCCESS:
                if self.recovery_start_time is None:
                    logging.warning('Camera lost with error code %s, starting recovery', result)
                    self.recovery_start_time = clock.now()
                if clock.now() - self.recovery_start_time > self.max_recovery_time_sec:
                    logging.error(f'Recovery time exceeded {self.max_recovery_time_sec} seconds, stopping')
                    return
                yield pimm.Sleep(0.01)
                continue

            if self.recovery_start_time is not None:
                logging.info(f'Camera recovered after {clock.now() - self.recovery_start_time:.2f} seconds')
                self.recovery_start_time = None

            image = sl.Mat()
            ts_s = zed.get_timestamp(TIME_REF_IMAGE).get_nanoseconds() / 1e9
            if zed.retrieve_image(image, self.view) == SUCCESS:
                # The images are in BGRA format, convert to RGB
                np_image = image.get_data()[:, :, [2, 1, 0]]

                # Emit main frame (either single view or side-by-side)
                # Note: For side-by-side, we emit the full (H, W*2, 3) image
                # Consumer is responsible for splitting if needed
                self._frame_adapter = NumpySMAdapter.lazy_init(np_image, self._frame_adapter)
                self.frame.emit(self._frame_adapter, ts=ts_s)

                # Handle depth if enabled and connected
                if self.init_params.depth_mode != sl.DEPTH_MODE.NONE:
                    # Only retrieve depth data if at least one depth channel is connected
                    if self.depth.num_bound > 0 or self.depth_mask.num_bound > 0:
                        depth = sl.Mat()
                        if zed.retrieve_measure(depth, sl.MEASURE.DEPTH) == SUCCESS:
                            depth_data = depth.get_data()

                            # Process and emit depth mask if connected
                            if self.depth_mask.num_bound > 0:
                                depth_mask = np.nan_to_num(depth_data, nan=0, posinf=0, neginf=0)
                                depth_mask[depth_mask != 0] = 255

                                self._depth_mask_adapter = NumpySMAdapter.lazy_init(
                                    depth_mask.astype(np.uint8)[..., np.newaxis], self._depth_mask_adapter
                                )
                                self.depth_mask.emit(self._depth_mask_adapter, ts=ts_s)

                            # Process and emit depth if connected
                            if self.depth.num_bound > 0:
                                depth_data = np.nan_to_num(
                                    depth_data, copy=False, nan=self.max_depth, posinf=self.max_depth, neginf=0
                                )
                                depth_data = depth_data.clip(max=self.max_depth) / self.max_depth * 255
                                depth_uint8 = depth_data.astype(np.uint8)[..., np.newaxis]

                                self._depth_adapter = NumpySMAdapter.lazy_init(depth_uint8, self._depth_adapter)
                                self.depth.emit(self._depth_adapter, ts=ts_s)

            fps_counter.tick()
            yield pimm.Sleep(0.01)
        zed.close()
