"""The error a device driver puts on its signal or answers a call with, typed by the kind of device that failed."""

from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum

import pimm


class FaultKind(Enum):
    E_STOP = 'e_stop'
    CAMERA = 'camera'
    ARM = 'arm'
    GRIPPER = 'gripper'


class DeviceFault(pimm.SignalError):
    """A ``pimm.SignalError`` whose ``kind`` names the kind of device that failed.

    The arguments are ``(message, kind)``: the error crosses a process boundary, and pickle builds it again from them
    and its ``__cause__``.
    """

    def __init__(self, message: str, kind: FaultKind):
        super().__init__(message, kind)
        self.kind = kind

    def __str__(self) -> str:
        return self.args[0]

    def __reduce__(self):
        return (type(self), self.args, {'__cause__': self.__cause__})


def device_fault(cause: Exception, kind: FaultKind, message: str = '') -> DeviceFault:
    """``cause`` when it is a ``DeviceFault``, else a ``DeviceFault`` of ``kind`` with ``cause`` as its ``__cause__``.

    ``message`` defaults to the text of ``cause``.
    """
    if isinstance(cause, DeviceFault):
        return cause
    fault = DeviceFault(message or str(cause) or repr(cause), kind)
    fault.__cause__ = cause
    return fault


@contextmanager
def raise_as_device_fault(kind: FaultKind) -> Iterator[None]:
    """Raise each exception of the block as ``device_fault(exc, kind)``."""
    try:
        yield
    except DeviceFault:
        raise
    except Exception as exc:
        raise device_fault(exc, kind) from exc
