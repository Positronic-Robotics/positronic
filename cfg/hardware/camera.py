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


@ir.config
def linux(device_path: str,
          width: int = 640,
          height: int = 480,
          fps: int = 30,
          pixel_format: str = "MJPG",
          image_mapping: Optional[Dict[str, str]] = None):
    from drivers.camera.linuxpy_video import LinuxPyCamera
    camera = LinuxPyCamera(device_path, width, height, fps, pixel_format)
    return _add_image_mapping(image_mapping, camera)


@ir.config(fps=60)
def luxonis(fps: int = 60, image_mapping: Optional[Dict[str, str]] = None):
    from drivers.camera.luxonis import LuxonisCamera
    return _add_image_mapping(image_mapping, LuxonisCamera(fps))


@ir.config(resolution=(640, 480), fps=30, enable_color=True, enable_depth=False, enable_infrared=False)
def realsense(resolution: Tuple[int, int],
              fps: int,
              enable_color: bool,
              enable_depth: bool,
              enable_infrared: bool,
              image_mapping: Optional[Dict[str, str]] = None):
    from drivers.camera.realsense import RealsenseCamera
    return _add_image_mapping(image_mapping,
                              RealsenseCamera(resolution, fps, enable_color, enable_depth, enable_infrared))


@ir.config
def stereolabs(fps: int,
               view: str = "LEFT",
               resolution: str = "AUTO",
               depth_mode: str = "NONE",
               max_depth: float = 10,
               coordinate_units: str = "METER",
               depth_mask: bool = False,
               image_mapping: Optional[Dict[str, str]] = None):
    from drivers.camera.sl import SLCamera
    import pyzed.sl as sl

    view = getattr(sl.VIEW, view)
    resolution = getattr(sl.RESOLUTION, resolution)
    depth_mode = getattr(sl.DEPTH_MODE, depth_mode)
    coordinate_units = getattr(sl.UNIT, coordinate_units)
    camera = SLCamera(fps, view, resolution, depth_mode, coordinate_units, max_depth, depth_mask)
    return _add_image_mapping(image_mapping, camera)


@ir.config
def opencv(camera_id: int = 0, resolution: Tuple[int, int] = (640, 480), fps: int = 30):
    from drivers.camera.opencv import OpenCVCameraCS, OpenCVCamera
    return OpenCVCameraCS(OpenCVCamera(camera_id, resolution, fps))


sl_vga = stereolabs.override(fps=30, view='SIDE_BY_SIDE', resolution='VGA', depth_mode='NONE', depth_mask=False)
arducam_video0 = linux.override(device_path="/dev/video0")
arducam_video2 = linux.override(device_path="/dev/video2")  # Yes, it is on video2
merged = ir.Config(merge_on_pulse, cameras={'first': arducam_video0, 'second': arducam_video2}, fps=30)
