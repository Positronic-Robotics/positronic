import pickle

import pytest

import pimm
from positronic.drivers.faults import DeviceFault, FaultKind


@pytest.mark.parametrize('kind', list(FaultKind))
def test_a_device_fault_keeps_its_kind_and_message_through_pickle(kind):
    fault = pickle.loads(pickle.dumps(DeviceFault('the device is lost', kind)))

    assert type(fault) is DeviceFault
    assert fault.kind is kind
    assert str(fault) == 'the device is lost'


def test_a_device_fault_on_a_process_pipe_is_caught_as_a_signal_error_and_keeps_its_kind():
    with pimm.World() as world:
        emitter, reader = world.mp_pipes()
        assert isinstance(reader, pimm.SignalReceiver)
        emitter.emit(DeviceFault('the arm is lost', FaultKind.ARM))

        with pytest.raises(pimm.SignalError, match='the arm is lost') as raised:
            _ = reader.value
    assert isinstance(raised.value, DeviceFault) and raised.value.kind is FaultKind.ARM
