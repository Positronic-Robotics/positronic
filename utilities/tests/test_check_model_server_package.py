"""Exercise the runtime dependency gate using installed distribution metadata."""

from functools import partial
from importlib.metadata import distributions

import pytest

from utilities import check_model_server_package as gate


@pytest.fixture
def install_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'distributions', partial(distributions, path=[str(tmp_path)]))

    def install(name, *requires):
        directory = tmp_path / f'{name}-1.0.dist-info'
        directory.mkdir(exist_ok=True)
        metadata = f'Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n'
        metadata += ''.join(f'Requires-Dist: {requirement}\n' for requirement in requires)
        (directory / 'METADATA').write_text(metadata)

    for name in ('positronic-model-server', 'positronic-wire', 'msgpack', 'numpy', 'pillow'):
        install(name)
    return install


@pytest.mark.parametrize(
    ('wires', 'packages'),
    [
        ([], []),
        (['websocket', 'websocket_tls', 'websocket_unix', 'roboarena'], ['websockets', 'uvicorn', 'click', 'h11']),
        (['grpc', 'grpc_tls'], ['grpcio', 'typing_extensions']),
    ],
)
def test_approved_runtime_installations_pass(install_metadata, wires, packages):
    for package in packages:
        install_metadata(package)
    gate.check_runtime_dependencies(wires)


@pytest.mark.parametrize('parent', ['positronic-model-server', 'msgpack'], ids=['direct', 'indirect'])
def test_dependency_additions_fail(install_metadata, parent):
    install_metadata(parent, 'unexpected-runtime>=1')
    install_metadata('unexpected-runtime')
    with pytest.raises(SystemExit, match='unexpected-runtime.*explicit design review'):
        gate.check_runtime_dependencies([])


@pytest.mark.parametrize(('wires', 'package'), [([], 'websockets'), (['websocket'], 'grpcio'), (['grpc'], 'uvicorn')])
def test_transport_dependencies_are_only_allowed_with_their_transport(install_metadata, wires, package):
    install_metadata(package)
    with pytest.raises(SystemExit, match=package):
        gate.check_runtime_dependencies(wires)


@pytest.mark.parametrize('package', ['Positronic_Model_Server', 'POSITRONIC...WIRE', 'Pillow'])
def test_distribution_names_are_normalized(install_metadata, package):
    install_metadata(package)
    gate.check_runtime_dependencies([])


def test_test_tools_are_not_allowed_in_the_runtime_check(install_metadata):
    install_metadata('pytest')
    with pytest.raises(SystemExit, match='pytest.*before installing test tools'):
        gate.check_runtime_dependencies([])
