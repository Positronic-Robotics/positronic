"""The ZED open path, with the SDK faked: a camera the SDK loses is rebooted once and opened once more."""

import enum
import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest

import pimm

SERIAL = 39567055
ZED_PATH = Path(__file__).parents[1] / 'zed.py'


class ErrorCode(enum.Enum):
    SUCCESS = 'SUCCESS'
    CAMERA_NOT_DETECTED = 'CAMERA NOT DETECTED'
    FAILURE = 'FAILURE'


class FakeSdk:
    """Stands in for `pyzed.sl`: scripted open results, a device list, and a reboot that records its calls."""

    def __init__(self, opens: list[ErrorCode], listed: list[bool], reboot_result: ErrorCode = ErrorCode.SUCCESS):
        self.opens = list(opens)
        self.listed = list(listed)
        self.reboot_result = reboot_result
        self.open_calls = 0
        self.list_calls = 0
        self.reboots: list[int] = []

    def open(self, _init_params) -> ErrorCode:
        self.open_calls += 1
        return self.opens.pop(0)

    def get_device_list(self) -> list[types.SimpleNamespace]:
        self.list_calls += 1
        present = self.listed.pop(0) if len(self.listed) > 1 else self.listed[0]
        return [types.SimpleNamespace(serial_number=SERIAL)] if present else []

    def reboot(self, serial: int) -> ErrorCode:
        self.reboots.append(serial)
        return self.reboot_result


@pytest.fixture
def zed_module(monkeypatch):
    """Load `zed.py` under a private name against a fake `pyzed`, so no other test sees the fake."""
    sl = types.ModuleType('pyzed.sl')
    pyzed = types.ModuleType('pyzed')
    monkeypatch.setitem(sys.modules, 'pyzed', pyzed)
    monkeypatch.setitem(sys.modules, 'pyzed.sl', sl)
    spec = importlib.util.spec_from_file_location('zed_with_fake_sdk', ZED_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'device_open_lock', nullcontext)
    return module


def _bind(module, sdk: FakeSdk) -> None:
    module.sl.ERROR_CODE = ErrorCode
    module.sl.Camera = types.SimpleNamespace(get_device_list=sdk.get_device_list, reboot=sdk.reboot)


def _open(module, sdk: FakeSdk, reboot_serial: int | None = SERIAL) -> list[pimm.Sleep]:
    _bind(module, sdk)
    zed = types.SimpleNamespace(open=sdk.open)
    return list(module.SLCamera._open_under_device_lock(zed, None, reboot_serial))


NOT_DETECTED = ErrorCode.CAMERA_NOT_DETECTED


def test_a_camera_the_sdk_cannot_find_is_rebooted_once_and_opened(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 3 + [ErrorCode.SUCCESS], listed=[False, True])
    _open(zed_module, sdk)
    assert sdk.reboots == [SERIAL]
    assert sdk.open_calls == 4


def test_the_open_waits_until_the_rebooted_camera_is_listed(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 3 + [ErrorCode.SUCCESS], listed=[False, False, True])
    sleeps = _open(zed_module, sdk)
    retry_sleeps, unlisted_polls = 2, 2
    assert len(sleeps) == retry_sleeps + unlisted_polls
    assert sdk.open_calls == 4


def test_a_camera_missing_from_the_device_list_is_rebooted_whatever_the_open_error(zed_module):
    sdk = FakeSdk(opens=[ErrorCode.FAILURE] * 3 + [ErrorCode.SUCCESS], listed=[False, True])
    _open(zed_module, sdk)
    assert sdk.reboots == [SERIAL]


def test_a_listed_camera_that_fails_to_open_is_not_rebooted(zed_module):
    sdk = FakeSdk(opens=[ErrorCode.FAILURE] * 3, listed=[True])
    with pytest.raises(RuntimeError, match='after 3 attempts: ErrorCode.FAILURE'):
        _open(zed_module, sdk)
    assert sdk.reboots == []


def test_a_failed_open_after_the_reboot_raises_without_a_second_reboot(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 4, listed=[True])
    with pytest.raises(RuntimeError, match='after its reboot'):
        _open(zed_module, sdk)
    assert sdk.reboots == [SERIAL]
    assert sdk.open_calls == 4


def test_a_reboot_the_sdk_refuses_raises_the_open_error(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 3, listed=[False], reboot_result=ErrorCode.FAILURE)
    with pytest.raises(RuntimeError, match='CAMERA_NOT_DETECTED; the SDK did not reboot camera'):
        _open(zed_module, sdk)
    assert sdk.open_calls == 3


def test_a_camera_that_is_not_listed_after_the_reboot_raises(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 3, listed=[False])
    with pytest.raises(RuntimeError, match='still not listed after its reboot'):
        _open(zed_module, sdk)
    assert sdk.list_calls == zed_module.REBOOTED_CAMERA_MAX_POLLS
    assert sdk.reboots == [SERIAL]
    assert sdk.open_calls == 3


def test_an_open_without_a_serial_is_not_rebooted(zed_module):
    sdk = FakeSdk(opens=[NOT_DETECTED] * 3, listed=[False])
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        _open(zed_module, sdk, reboot_serial=None)
    assert sdk.reboots == []
