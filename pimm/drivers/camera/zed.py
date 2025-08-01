from typing import Iterator

import numpy as np
import pyzed.sl as sl

import ironic2 as ir
from ironic.utils import FPSCounter


class SLCamera:
    frame: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(
            self,
            fps: int = 30,
            view: sl.VIEW = sl.VIEW.LEFT,
            resolution: sl.RESOLUTION = sl.RESOLUTION.AUTO,
            depth_mode: sl.DEPTH_MODE = sl.DEPTH_MODE.NONE,
            coordinate_units: sl.UNIT = sl.UNIT.METER,
            max_depth: float = 10,
            depth_mask: bool = False
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
        """
        super().__init__()
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = resolution
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = depth_mode
        self.init_params.coordinate_units = coordinate_units
        self.init_params.sdk_verbose = 1
        self.init_params.enable_image_enhancement = False
        self.init_params.async_grab_camera_recovery = False
        self.view = view

        self.max_depth = max_depth
        self.depth_mask = depth_mask

    def run(self, should_stop: ir.SignalReader, clock: ir.Clock) -> Iterator[ir.Sleep]:
        fps_counter = FPSCounter("Camera")
        zed = sl.Camera()
        zed.open(self.init_params)

        SUCCESS = sl.ERROR_CODE.SUCCESS
        TIME_REF_IMAGE = sl.TIME_REFERENCE.IMAGE

        while not should_stop.value:
            result = zed.grab()
            frame = {}
            if result != SUCCESS:
                yield ir.Sleep(0.001)
                continue

            image = sl.Mat()
            ts_s = zed.get_timestamp(TIME_REF_IMAGE).get_nanoseconds() / 1e9
            if zed.retrieve_image(image, self.view) == SUCCESS:
                # The images are in BGRA format
                np_image = image.get_data()[:, :, [2, 1, 0]]

                if self.view == sl.VIEW.SIDE_BY_SIDE:
                    w = np_image.shape[1] // 2
                    frame['left'] = np_image[:, :w, :]
                    frame['right'] = np_image[:, w:, :]
                else:
                    frame['image'] = np_image

                if self.init_params.depth_mode != sl.DEPTH_MODE.NONE:
                    depth = sl.Mat()
                    if zed.retrieve_measure(depth, sl.MEASURE.DEPTH) == SUCCESS:
                        data = depth.get_data()
                        if self.depth_mask:
                            depth_mask = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
                            depth_mask[depth_mask != 0] = 255
                            frame['depth_mask'] = depth_mask.astype(np.uint8)[..., np.newaxis]

                        data = np.nan_to_num(data, copy=False, nan=self.max_depth, posinf=self.max_depth, neginf=0)
                        data = data.clip(max=self.max_depth) / self.max_depth * 255
                        # Adding last axis so that it has same number of dimensions as normal image
                        frame['depth'] = data.astype(np.uint8)[..., np.newaxis]
            self.frame.emit(frame, ts=ts_s)
            fps_counter.tick()
            yield ir.Sleep(0.001)
        zed.close()
