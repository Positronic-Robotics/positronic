"""The backend that `server.py` runs in ABC's interpreter, against stand-ins for the ABC modules it imports."""

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from positronic.vendors.abc_dit import server

_POLICY = object()


class _PolicyServer:
    """Records where ABC's websocket policy server is asked to listen."""

    started: list[dict[str, Any]] = []

    def __init__(self, policy: Any, host: str = '0.0.0.0', port: int | None = None):
        self.address = {'policy': policy, 'host': host, 'port': port}

    def serve_forever(self) -> None:
        _PolicyServer.started.append(self.address)


@pytest.fixture
def backend(monkeypatch) -> types.ModuleType:
    """The backend script, loaded as ABC's interpreter loads it, over stand-in ABC modules."""
    modules = {
        'tyro': types.SimpleNamespace(cli=lambda cls: cls()),
        'deploy': types.ModuleType('deploy'),
        'deploy.serve_policy': types.SimpleNamespace(create_policy=lambda args: _POLICY),
        'deploy.serve_policy_config': types.SimpleNamespace(Args=lambda: types.SimpleNamespace(port=9123)),
        'deploy.websocket_server': types.SimpleNamespace(WebsocketPolicyServer=_PolicyServer),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    _PolicyServer.started.clear()
    spec = importlib.util.spec_from_file_location('abc_dit_backend', Path(server.BACKEND_SCRIPT))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_backend_serves_abcs_policy_on_loopback_only(backend):
    backend.main(types.SimpleNamespace(port=9123))

    # The backend takes no token, so only the server on this machine may reach it.
    assert _PolicyServer.started == [{'policy': _POLICY, 'host': '127.0.0.1', 'port': 9123}]
