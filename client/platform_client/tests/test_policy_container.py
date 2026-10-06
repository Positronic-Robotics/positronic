"""The published container contract: its values, the flags that set them, and the compressed-size rule."""

import json
import re
from pathlib import Path

import pytest
from platform_client import policy_container
from platform_client.policy_container import ImageManifest, docker_limit_flags
from pydantic import ValidationError

_DOC = Path(__file__).resolve().parents[3] / 'docs' / 'submit-a-policy-image.md'
_CONSTANT = r'[A-Z][A-Z0-9_]*'
_ROW = re.compile(rf'^\| (.+?) \| (.+?) \| `({_CONSTANT})` \|$', re.MULTILINE)
_SIZE_UNITS = (('GiB', 2**30), ('GB', 10**9), ('MiB', 2**20), ('MB', 10**6))


def _size(value: int) -> str:
    """`value` bytes in the largest unit that divides it."""
    for unit, scale in _SIZE_UNITS:
        if value % scale == 0:
            return f'{value // scale} {unit}'
    return f'{value} bytes'


def _shown(name: str, value: object) -> str:
    """How the doc's table writes the value of the constant `name`."""
    if isinstance(value, str):
        return f'`{value}`'
    assert isinstance(value, int | float)
    if name.endswith('_BYTES_PER_S'):
        return f'{_size(int(value))}/s'
    if name.endswith('_BYTES'):
        return _size(int(value))
    if name.endswith('_BITS_PER_S'):
        return f'{int(value) // 10**6} Mbit/s'
    if name.endswith('_IOPS'):
        return f'{int(value):,}/s'
    if name.endswith('_S'):
        return f'{value:g} s'
    if name.endswith('_MIB'):
        return f'{value} MiB'
    if name.endswith('_PORT'):
        return str(value)
    return f'{value:,}'


def test_the_doc_states_every_published_value_as_the_module_holds_it():
    rows = {name: value for _limit, value, name in _ROW.findall(_DOC.read_text())}
    published = {name: value for name, value in vars(policy_container).items() if re.fullmatch(_CONSTANT, name)}
    assert rows.keys() == published.keys()
    for name, value in published.items():
        assert rows[name] == _shown(name, value), name


def test_both_image_budgets_and_the_log_fit_in_the_image_store():
    """A pull holds the compressed layers beside the unpacked ones, and the log lands in the same store."""
    held = policy_container.COMPRESSED_IMAGE_BYTES + policy_container.UNPACKED_IMAGE_BYTES
    assert held + policy_container.LOG_BYTES < policy_container.IMAGE_STORE_BYTES


def _value_of(flags: list[str], flag: str) -> str:
    return flags[flags.index(flag) + 1]


def test_the_flags_carry_the_published_limits():
    flags = docker_limit_flags(Path('/dev/vda'))
    assert _value_of(flags, '--cap-drop') == 'ALL'
    assert _value_of(flags, '--security-opt') == 'no-new-privileges'
    assert _value_of(flags, '--memory') == str(policy_container.MEMORY_BYTES)
    assert _value_of(flags, '--cpus') == str(policy_container.CPUS)
    assert _value_of(flags, '--pids-limit') == str(policy_container.PIDS)
    assert _value_of(flags, '--log-opt') == f'max-size={policy_container.LOG_BYTES}'


def test_the_container_gets_no_swap():
    """Docker allows swap as large as the memory limit again when the swap limit is unset."""
    flags = docker_limit_flags(Path('/dev/vda'))
    assert _value_of(flags, '--memory-swap') == _value_of(flags, '--memory')


def test_the_disk_rates_bound_the_disk_they_name():
    flags = docker_limit_flags(Path('/dev/nvme1n1'))
    assert _value_of(flags, '--device-read-bps') == f'/dev/nvme1n1:{policy_container.DISK_READ_BYTES_PER_S}'
    assert _value_of(flags, '--device-write-bps') == f'/dev/nvme1n1:{policy_container.DISK_WRITE_BYTES_PER_S}'
    assert _value_of(flags, '--device-read-iops') == f'/dev/nvme1n1:{policy_container.DISK_READ_IOPS}'
    assert _value_of(flags, '--device-write-iops') == f'/dev/nvme1n1:{policy_container.DISK_WRITE_IOPS}'


# An OCI image manifest as a registry serves it, the fields this module does not read included.
_MANIFEST = {
    'schemaVersion': 2,
    'mediaType': 'application/vnd.oci.image.manifest.v1+json',
    'config': {'mediaType': 'application/vnd.oci.image.config.v1+json', 'digest': 'sha256:' + 'a' * 64, 'size': 7023},
    'layers': [
        {'mediaType': 'application/vnd.oci.image.layer.v1.tar+gzip', 'digest': 'sha256:' + 'b' * 64, 'size': 32654},
        {'mediaType': 'application/vnd.oci.image.layer.v1.tar+gzip', 'digest': 'sha256:' + 'c' * 64, 'size': 16724},
    ],
    'annotations': {'org.opencontainers.image.created': '2026-01-01T00:00:00Z'},
}


def test_the_compressed_size_is_the_config_and_every_layer():
    assert ImageManifest.model_validate_json(json.dumps(_MANIFEST)).compressed_size == 7023 + 32654 + 16724


def test_a_manifest_index_is_no_image_manifest():
    """An index lists a manifest per platform and no layers; the size is read off the platform's manifest."""
    index = {'schemaVersion': 2, 'manifests': [{'digest': 'sha256:' + 'd' * 64, 'size': 1, 'platform': {}}]}
    with pytest.raises(ValidationError):
        ImageManifest.model_validate_json(json.dumps(index))


def test_a_negative_size_is_refused():
    manifest = {**_MANIFEST, 'config': {**_MANIFEST['config'], 'size': -1}}
    with pytest.raises(ValidationError):
        ImageManifest.model_validate_json(json.dumps(manifest))
