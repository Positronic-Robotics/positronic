from typing import Dict, Optional, Tuple
from drivers.camera.merge import merge_on_pulse

import ironic as ir


def _add_image_mapping(mapping: Optional[Dict[str, str]], camera: ir.ControlSystem):
    if mapping is None:
        return camera

    def map_images(frame):
        return {new_k: frame[k] for k, new_k in mapping.items()}

    map_system = ir.utils.MapPortCS(map_images)
    map_system.bind(input=camera.outs.frame)
    return ir.compose(camera, map_system, outputs={'frame': map_system.outs.output})


@ir.config(width=640, height=480, fps=30, pixel_format="MJPG", image_mapping=None)
def linux(device_path: str, width: int, height: int, fps: int, pixel_format: str, image_mapping: Optional[Dict[str,
                                                                                                               str]]):
    from drivers.camera.linuxpy_video import LinuxPyCamera
    camera = LinuxPyCamera(device_path, width, height, fps, pixel_format)
    return _add_image_mapping(image_mapping, camera)


@ir.config(fps=60)
def luxonis(fps: int, image_mapping: Optional[Dict[str, str]] = None):
    from drivers.camera.luxonis import LuxonisCamera
    return _add_image_mapping(image_mapping, LuxonisCamera(fps))


@ir.config(resolution=(640, 480),
           fps=30,
           enable_color=True,
           enable_depth=False,
           enable_infrared=False,
           image_mapping=None)
def realsense(resolution: Tuple[int, int], fps: int, enable_color: bool, enable_depth: bool, enable_infrared: bool,
              image_mapping: Optional[Dict[str, str]]):
    from drivers.camera.realsense import RealsenseCamera
    return _add_image_mapping(image_mapping,
                              RealsenseCamera(resolution, fps, enable_color, enable_depth, enable_infrared))


@ir.config(fps=30,
           view="LEFT",
           resolution="AUTO",
           depth_mode="NONE",
           depth_mask=False,
           max_depth=10,
           coordinate_units="METER",
           image_mapping=None)
def stereolabs(fps: int, view: str, resolution: str, depth_mode: str, max_depth: float, coordinate_units: str,
               depth_mask: bool, image_mapping: Optional[Dict[str, str]]):
    from drivers.camera.sl import SLCamera
    import pyzed.sl as sl

    view = getattr(sl.VIEW, view)
    resolution = getattr(sl.RESOLUTION, resolution)
    depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
    coordinate_units = getattr(sl.UNIT, coordinate_units)
    camera = SLCamera(fps, view, resolution, depth_mode, coordinate_units, max_depth, depth_mask)
    return _add_image_mapping(image_mapping, camera)


@ir.config(camera_id=0, resolution=(640, 480), fps=30, image_mapping=None)
def opencv(camera_id: int, resolution: Tuple[int, int], fps: int, image_mapping: Optional[Dict[str, str]]):
    from drivers.camera.opencv import OpenCVCameraCS, OpenCVCamera
    return _add_image_mapping(image_mapping, OpenCVCameraCS(OpenCVCamera(camera_id, resolution, fps)))


sl_vga = stereolabs.override(fps=30, view='SIDE_BY_SIDE', resolution='VGA', depth_mode='NONE', depth_mask=False)
arducam_video0 = linux.override(device_path="/dev/video0")
arducam_video2 = linux.override(device_path="/dev/video2")  # Yes, it is on video2
merged = ir.Config(merge_on_pulse, cameras={'first': arducam_video0, 'second': arducam_video2}, fps=30)
