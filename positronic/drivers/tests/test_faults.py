import pickle

import pytest

import pimm
from positronic.drivers.faults import DeviceFault, FaultKind, device_fault, raise_as_device_fault


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


def test_a_device_fault_keeps_the_raw_error_as_its_cause_through_pickle():
    fault = pickle.loads(pickle.dumps(device_fault(RuntimeError('libfranka: UDP receive: Timeout'), FaultKind.ARM)))

    assert fault.kind is FaultKind.ARM
    assert str(fault) == 'libfranka: UDP receive: Timeout'
    assert type(fault.__cause__) is RuntimeError and str(fault.__cause__) == 'libfranka: UDP receive: Timeout'


def test_device_fault_names_its_own_message_and_keeps_the_text_of_the_cause_by_default():
    cause = TimeoutError('the arm stopped short')

    assert str(device_fault(cause, FaultKind.ARM, 'the arm is lost')) == 'the arm is lost'
    assert device_fault(cause, FaultKind.ARM, 'the arm is lost').__cause__ is cause
    assert str(device_fault(TimeoutError(), FaultKind.ARM)) == 'TimeoutError()'


def test_device_fault_returns_a_device_fault_unchanged():
    fault = DeviceFault('the emergency stop is pressed', FaultKind.E_STOP)

    assert device_fault(fault, FaultKind.ARM) is fault


def test_raise_as_device_fault_raises_an_error_of_the_block_as_a_device_fault_of_the_kind():
    with pytest.raises(DeviceFault, match='bus closed') as raised, raise_as_device_fault(FaultKind.GRIPPER):
        raise ConnectionError('bus closed')

    assert raised.value.kind is FaultKind.GRIPPER
    assert type(raised.value.__cause__) is ConnectionError


def test_raise_as_device_fault_keeps_the_kind_of_a_device_fault_from_the_block():
    with pytest.raises(DeviceFault) as raised, raise_as_device_fault(FaultKind.ARM):
        raise DeviceFault('the emergency stop is pressed', FaultKind.E_STOP)

    assert raised.value.kind is FaultKind.E_STOP
