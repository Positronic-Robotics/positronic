"""The i2rt YAM arm: the driver in `driver`, its FK/IK in `kinematics`, its settle tuning in `settle`, how its gripper
closes on an object in `grasp`, the torque it asks i2rt to add in `compensation`."""

import importlib

_SUBMODULES = ('driver', 'kinematics', 'settle', 'grasp', 'compensation')


def __getattr__(name: str):
    # The driver loads on first use: it needs the yam extra, and the configs import `settle`, `grasp` and
    # `compensation` without it.
    if name in _SUBMODULES or name.startswith('__'):
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    return getattr(importlib.import_module(f'{__name__}.driver'), name)
