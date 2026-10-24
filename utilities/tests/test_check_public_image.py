import io
import tarfile

from utilities.check_public_image import findings

TRACKED = {'positronic/README.md', 'positronic/docker/run_molmospaces_eval.sh'}


def _image(files: dict[str, str]) -> tarfile.TarFile:
    """A tar of `files`, path to text, read in order as the check reads an exported image."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as tar:
        for path, text in files.items():
            data = text.encode()
            member = tarfile.TarInfo(path)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    buffer.seek(0)
    return tarfile.open(fileobj=buffer, mode='r|')


def _findings(files: dict[str, str]) -> list[str]:
    return list(findings(_image(files), TRACKED))


def test_tracked_files_of_the_repository_pass():
    assert _findings({'positronic/README.md': 'the platform client', 'etc/hostname': 'box'}) == []


def test_a_file_under_the_repository_that_git_does_not_track_is_a_finding():
    assert _findings({'positronic/episodes/0.parquet': ''}) == [
        '/positronic/episodes/0.parquet: git does not track this file'
    ]


def test_build_output_under_the_repository_passes():
    files = {
        'positronic/positronic.egg-info/PKG-INFO': 'Name: positronic',
        'positronic/positronic/__pycache__/keys.cpython-313.pyc': '',
    }
    assert _findings(files) == []


def test_a_public_source_may_name_an_internal_word():
    files = {
        'usr/lib/python3/platform.py': 'import platform',
        'positronic/.venv/lib/python3.13/site-packages/platform_client/x.py': 'gateway',
        'root/.cache/positronic/molmospaces/src/.venv/lib/python3.11/site-packages/a/platform.py': '',
        'root/.cache/positronic/molmospaces/src/molmo_spaces/env.py': 'sys.platform',
    }
    assert _findings(files) == []


def test_an_internal_word_in_a_path_outside_the_public_sources_is_a_finding():
    assert _findings({'opt/gateway/config.yaml': ''}) == ['/opt/gateway/config.yaml: an internal name']


def test_the_distro_config_may_name_an_internal_word_and_hold_ca_certificates():
    files = {
        'etc/sysctl.conf': '# Do not act as a router or gateway',
        'etc/ssl/certs/Amazon_Root_CA_1.pem': '-----BEGIN CERTIFICATE-----',
    }
    assert _findings(files) == []


def test_a_private_key_or_host_name_in_the_distro_config_is_a_finding():
    found = _findings({'etc/ssh/key': '-----BEGIN OPENSSH PRIVATE KEY-----', 'etc/hosts': '10.0.0.1 box.tail1.ts.net'})
    assert found == ["/etc/ssh/key: holds '-----BEGIN OPENSSH PRIVATE KEY-----'", "/etc/hosts: holds 'tail1.ts.net'"]


def test_a_key_file_outside_the_public_sources_is_a_finding():
    assert _findings({'root/.ssh/id_ed25519': ''}) == ['/root/.ssh/id_ed25519: an internal name']


def test_a_private_key_in_a_file_outside_the_public_sources_is_a_finding():
    found = _findings({'root/notes.txt': '-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n'})
    assert found == ["/root/notes.txt: holds '-----BEGIN OPENSSH PRIVATE KEY-----'"]


def test_a_private_host_name_in_a_file_outside_the_public_sources_is_a_finding():
    found = _findings({'molmospaces/cache/hosts': 'connect to box-1.tail1234.ts.net'})
    assert found == ["/molmospaces/cache/hosts: holds 'tail1234.ts.net'"]


def test_an_internal_word_in_the_text_of_a_file_outside_the_public_sources_is_a_finding():
    found = _findings({'root/run.yaml': 'url: http://orchestrator:8080'})
    assert found == ["/root/run.yaml: holds 'orchestrator'"]
