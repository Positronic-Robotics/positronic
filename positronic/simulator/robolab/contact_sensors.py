"""Contact sensors that update when a reader asks for their data, not at every physics substep."""

from typing import Any


def make_lazy(scene_cfg: Any, sensor_cfg_type: type) -> int:
    """Set ``history_length`` to 0 on each sensor of ``sensor_cfg_type`` in ``scene_cfg``, and return how many.

    Isaac Lab updates a sensor with a history at every physics substep, and each contact sensor update waits for
    the GPU. A sensor without a history updates when its ``data`` is read. RoboLab reads only the current contact
    forces, once per control step, so it reads the same values with one update in place of ``decimation``.
    """
    sensors = [each for each in vars(scene_cfg).values() if isinstance(each, sensor_cfg_type)]
    for sensor in sensors:
        sensor.history_length = 0
    return len(sensors)
