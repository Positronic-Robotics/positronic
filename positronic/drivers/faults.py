"""The error a device driver puts on its signal, typed by the kind of device that failed."""

from enum import Enum

import pimm


class FaultKind(Enum):
    E_STOP = 'e_stop'
    CAMERA = 'camera'
    ARM = 'arm'
    GRIPPER = 'gripper'


class DeviceFault(pimm.SignalError):
    """A ``pimm.SignalError`` whose ``kind`` names the kind of device that failed.

    The arguments are ``(message, kind)``: the error crosses a process boundary, and pickle builds it again from them.
    """

    def __init__(self, message: str, kind: FaultKind):
        super().__init__(message, kind)
        self.kind = kind

    def __str__(self) -> str:
        return self.args[0]
