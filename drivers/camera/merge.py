import asyncio
from typing import Any, Dict, List, Tuple, Union
import ironic as ir

def _merge_frames(cam2frame_dicts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple camera frames into a single dictionary.

    Args:
        cam2frame_dicts: Dictionary mapping camera names to their frame dictionaries.
                        Each frame dictionary maps image names to numpy arrays.

    Returns:
        A merged dictionary with keys of format "{camera_name}.{image_name}".
        Returns ir.NoValue if any frame is ir.NoValue.
    """
    result = {}
    for cam_key, frame_dict in cam2frame_dicts.items():
        if frame_dict is ir.NoValue:
            return ir.NoValue
        result.update({f'{cam_key}.{im_key}': image for im_key, image in frame_dict.items()})
    return result


def merge_on_pulse(cameras: Dict[str, ir.ControlSystem], fps: float) -> ir.ControlSystem:
    """Merge multiple cameras into a single virtual camera system with pulse-based triggering.

    Combines frames from multiple cameras into a single output frame, triggered by a pulse signal.
    Output frame keys are prefixed with camera names (e.g. "cam1.image", "cam2.depth").
    Outputs ir.NoValue if any camera hasn't produced a frame yet.

    Args:
        cameras: Dict mapping camera names to control systems with 'frame' output ports
        fps: Frames per second to emit

    Returns:
        Control system with 'frame' output port emitting merged frames

    Example:
        ```python
        cam1 = SLCamera()  # {"left": img1, "right": img2}
        cam2 = RealsenseCamera()  # {"image": img3, "depth": img4}
        merged = merge_on_pulse({"cam1": cam1, "cam2": cam2}, fps=10)
        # Output: {"cam1.left": img1, "cam1.right": img2,
        #          "cam2.image": img3, "cam2.depth": img4}
        ```
    """
    pulse = ir.utils.Pulse(fps)

    last_values = {name: ir.utils.last_value(camera.outs.frame) for name, camera in cameras.items()}
    merge_dict = ir.utils.properties_dict(**last_values)

    async def merge_frames(_ignored_pulse_data: Any) -> Dict[str, Any]:
        cam2frame_dicts = await merge_dict()
        return _merge_frames(cam2frame_dicts.data)

    return ir.compose(*cameras.values(), pulse, outputs={'frame': ir.utils.map_port(merge_frames, pulse.outs.pulse)})


def merge_on_camera(main_camera: Tuple[str, ir.ControlSystem], extension_cameras: Dict[str, ir.ControlSystem]) -> ir.ControlSystem:
    """Merge multiple cameras into a single virtual camera system, triggered by a main camera.

    Combines frames from multiple cameras into a single output frame, triggered by the main camera.
    Output frame keys are prefixed with camera names (e.g. "cam1.image", "cam2.depth").
    Outputs ir.NoValue if any extension camera hasn't produced a frame yet.

    Args:
        main_camera: (name, camera) tuple where camera triggers frame merging. Camera must have
                    'frame' output port emitting image dict.
        extension_cameras: Dict mapping names to camera systems with 'frame' output ports.

    Returns:
        Control system with 'frame' output port emitting merged frames.

    Example:
        ```python
        main = SLCamera()  # {"left": img1, "right": img2}
        ext = RealsenseCamera()  # {"image": img3, "depth": img4}
        merged = merge_on_camera(("main", main), {"ext": ext})
        # Output: {"main.left": img1, "main.right": img2,
        #          "ext.image": img3, "ext.depth": img4}
        ```

    Raises:
        AssertionError: If main camera name conflicts with extension cameras.
    """
    last_values = {name: ir.utils.last_value(camera.outs.frame) for name, camera in extension_cameras.items()}
    merge_dict = ir.utils.properties_dict(**last_values)

    name, camera = main_camera
    assert name not in extension_cameras, f"Main camera {name} is also an extension camera"

    async def merge_frames(main_camera_frame: Any) -> Dict[str, Any]:
        cam2frame_dicts = (await merge_dict()).data
        cam2frame_dicts[name] = main_camera_frame
        return _merge_frames(cam2frame_dicts)

    return ir.compose(*extension_cameras.values(), camera, outputs={'frame': ir.utils.map_port(merge_frames, camera.outs.frame)})
