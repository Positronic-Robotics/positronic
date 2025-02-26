from typing import Dict, Optional, Tuple
from drivers.camera.merge import merge_on_pulse

from hydra_zen import builds, store

import ironic as ir


def _add_image_mapping(mapping: Optional[Dict[str, str]], camera: ir.ControlSystem):
    if mapping is None:
        return camera

    def map_images(frame):
        return {new_k: frame[k] for k, new_k in mapping.items()}

    map_system = ir.utils.MapPortCS(map_images)
    map_system.bind(input=camera.outs.frame)
    return ir.compose(camera, map_system, outputs={'frame': map_system.outs.output})


def _linux_camera(device_path: str,
                  width: int = 640,
                  height: int = 480,
                  fps: int = 30,
                  pixel_format: str = "MJPG",
                  image_mapping: Optional[Dict[str, str]] = None):
    from drivers.camera.linuxpy_video import LinuxPyCamera
    camera = LinuxPyCamera(device_path, width, height, fps, pixel_format)
    return _add_image_mapping(image_mapping, camera)


def _luxonis_camera(fps: int = 60, image_mapping: Optional[Dict[str, str]] = None):
    from drivers.camera.luxonis import LuxonisCamera
    return _add_image_mapping(image_mapping, LuxonisCamera(fps))


def _realsense_camera(resolution: Tuple[int, int] = (640, 480),
                      fps: int = 30,
                      enable_color: bool = True,
                      enable_depth: bool = True,
                      enable_infrared: bool = True,
                      image_mapping: Optional[Dict[str, str]] = None):
    from drivers.camera.realsense import RealsenseCamera
    return _add_image_mapping(image_mapping,
                              RealsenseCamera(resolution, fps, enable_color, enable_depth, enable_infrared))


def _stereolabs_camera(fps: int,
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


def _opencv_camera(camera_id: int = 0, resolution: Tuple[int, int] = (640, 480), fps: int = 30):
    from drivers.camera.opencv import OpenCVCameraCS, OpenCVCamera
    return OpenCVCameraCS(OpenCVCamera(camera_id, resolution, fps))


linux_camera = builds(_linux_camera, populate_full_signature=True)
luxonis_camera = builds(_luxonis_camera, populate_full_signature=True)
realsense_camera = builds(_realsense_camera, populate_full_signature=True)
stereolabs_camera = builds(_stereolabs_camera, populate_full_signature=True)
merge_camera = builds(merge_on_pulse, populate_full_signature=True)
opencv_camera = builds(_opencv_camera, populate_full_signature=True)

arducam_video0 = linux_camera(device_path="/dev/video0")

cam_store = store(group="hardware/cameras")
cam_store(arducam_video0, name='arducam')
cam_store(luxonis_camera(fps=60), name='luxonis')
cam_store(realsense_camera(resolution=(640, 480), fps=30, enable_color=True, enable_depth=False, enable_infrared=False),
          name='realsense')
cam_store(stereolabs_camera(fps=30, view='SIDE_BY_SIDE', resolution='VGA', depth_mode='NONE', depth_mask=False),
          name='sl_vga')
cam_store(opencv_camera(camera_id=0, resolution=(640, 480), fps=30), name='opencv')

arducam_video2 = linux_camera(device_path="/dev/video2")  # Yes, it is on video2
cam_store(merge_camera(cameras={'first': arducam_video0, 'second': arducam_video2}, fps=30), name='merged')

cam_store.add_to_hydra_store()
