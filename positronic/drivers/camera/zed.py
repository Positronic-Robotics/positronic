from typing import Iterator, Literal

import numpy as np
import pyzed.sl as sl

import pimm


# TODO: Currently in order to have pyzed available, one need to install Stereolabs SDK, and then generate
# wheel file from it. We need to find a better solution how to install it via uv or at least Docker.
class SLCamera(pimm.ControlSystem):
    def __init__(self,
                 serial_number: int | None = None,
                 fps: int | None = None,
                 view: Literal["left", "right", "side_by_side"] = "left",
                 resolution: Literal["hd4k", "qhdplus", "hd2k", "hd1080", "hd1200", "hd1536", "hd720", "svga", "vga",
                                     "auto"] = "auto",
                 depth_mode: Literal["none", "near", "far", "high", "ultra"] = "none",
                 max_depth: float = 10,
                 depth_mask: bool = False):
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
        self.init_params.enable_image_enhancement = False
        self.init_params.async_grab_camera_recovery = False

        self.max_depth = max_depth
        self.depth_mask = depth_mask
        self.frame: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        SUCCESS = sl.ERROR_CODE.SUCCESS
        TIME_REF_IMAGE = sl.TIME_REFERENCE.IMAGE
        fps_counter = pimm.utils.RateCounter("Camera")

        zed = sl.Camera()
        error_code = zed.open(self.init_params)
        if error_code != SUCCESS:
            print(f"Failed to open camera: {error_code}")
            return

        while not should_stop.value:
            result = zed.grab()
            frame = {}
            if result != SUCCESS:
                yield pimm.Sleep(0.01)
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
            yield pimm.Sleep(0.01)
        zed.close()
