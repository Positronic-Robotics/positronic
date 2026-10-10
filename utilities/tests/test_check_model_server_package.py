"""Exercise the runtime dependency gate using installed distribution metadata."""

from functools import partial
from importlib.metadata import distributions
from pathlib import Path

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


@pytest.fixture
def wrapper(tmp_path):
    package = tmp_path / 'wrapper'
    package.mkdir()
    (package / 'keys.py').write_text("FIELD = 'field'\n")
    return package


@pytest.mark.parametrize(
    'source',
    [
        'from . import keys\ndef response(): return {keys.FIELD: 1}\n',
        'from .keys import FIELD\ndef response(): return {FIELD: 1}\n',
        'from .keys import FIELD as field\ndef response(): return {field: 1}\n',
        'import wrapper.keys as fields\ndef response(): return {fields.FIELD: 1}\n',
        'import wrapper.keys\ndef response(): return {wrapper.keys.FIELD: 1}\n',
    ],
)
def test_constant_used_by_wrapper_code_passes(wrapper, source):
    (wrapper / 'server.py').write_text(source)
    gate.check_internal_constants(wrapper)


@pytest.mark.parametrize('path', ['keys.py', 'spec.py', 'vendor_keys.py', 'nested/settings.py'])
@pytest.mark.parametrize('definition', ["MODEL_SETTING = 'setting'", "MODEL_SETTING: str = 'setting'"])
def test_constants_in_any_wrapper_module_need_internal_use(wrapper, path, definition):
    (wrapper / 'keys.py').write_text("FIELD = 'field'\ndef response(): return {FIELD: 1}\n")
    target = wrapper / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('a') as output:
        output.write(definition + '\n')
    with pytest.raises(SystemExit, match='MODEL_SETTING.*owning modules'):
        gate.check_internal_constants(wrapper)


@pytest.mark.parametrize('path', ['tests/test_keys.py', 'examples/serve.py', 'test_keys.py', 'keys_test.py'])
def test_tests_and_examples_do_not_count_as_internal_use(wrapper, path):
    target = wrapper / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('from wrapper import keys\nassert keys.FIELD\n')
    with pytest.raises(SystemExit, match='wrapper.keys.FIELD'):
        gate.check_internal_constants(wrapper)


@pytest.mark.parametrize(
    'source',
    [
        'from .keys import FIELD\n',
        'from . import keys\nFIELD = keys.FIELD\n',
        "from .keys import FIELD\n__all__ = ['FIELD']\n",
        'from .keys import FIELD\n__all__ = [FIELD]\n',
        'import unrelated.keys\ndef response(): return {unrelated.keys.FIELD: 1}\n',
        'FIELD = "another field"\ndef response(): return {FIELD: 1}\n',
    ],
)
def test_reexports_and_unrelated_names_do_not_count_as_internal_use(wrapper, source):
    (wrapper / '__init__.py').write_text(source)
    with pytest.raises(SystemExit, match='wrapper.keys.FIELD'):
        gate.check_internal_constants(wrapper)


def test_model_server_constants_have_internal_production_uses():
    gate.check_internal_constants(Path(gate.spec.__file__).parent)
