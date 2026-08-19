"""The gate that keeps the client's code, its version, and the root's pin on it in step."""

import pytest

from utilities import check_client_version_bump as gate


def test_a_later_version_is_later():
    assert gate.is_later_version('0.1.0', '0.2.0')
    assert gate.is_later_version('0.1.0', '0.1.1')
    assert gate.is_later_version('0.9.0', '0.10.0')  # numeric, so 10 is not read as older than 9


def test_an_unchanged_or_backwards_version_is_not_later():
    assert not gate.is_later_version('0.2.0', '0.2.0')
    assert not gate.is_later_version('0.2.0', '0.1.0')


def test_a_padded_zero_is_the_same_version():
    # packaging reads 1.0 and 1.0.0 as one version, so a release "bumped" that way ships as its
    # predecessor and `skip-existing` skips it.
    assert not gate.is_later_version('1.0', '1.0.0')
    assert not gate.is_later_version('1.0.0', '1.0')


def test_pre_releases_order_the_way_the_index_orders_them():
    # PEP 440, which a hand-rolled tuple gets backwards in both directions: a release candidate
    # PRECEDES its release, and rc10 FOLLOWS rc2.
    assert gate.is_later_version('1.0rc1', '1.0')
    assert not gate.is_later_version('1.0', '1.0rc1')
    assert gate.is_later_version('1.0rc2', '1.0rc10')
    assert not gate.is_later_version('1.0rc10', '1.0rc2')


def test_a_version_the_index_could_not_read_fails_closed():
    with pytest.raises(SystemExit):
        gate.is_later_version('0.1.0', 'not-a-version')


def test_the_declared_version_is_read_from_a_manifest():
    assert gate.declared_version('[project]\nname = "x"\nversion = "0.3.1"\n') == '0.3.1'
    assert gate.declared_version('[project]\nname = "x"\n') is None
    # Parsed, so only `[project].version` is the project's own: a version under a tool's section is
    # that tool's, and to a text scan both read the same.
    assert gate.declared_version('[project]\nname = "x"\n\n[tool.bumper]\nversion = "9.9.9"\n') is None


def test_a_base_side_manifest_that_does_not_parse_abstains():
    # The base's copy is one this gate cannot judge, and an unjudgeable base has never blocked a
    # commit: `check` prints a NOTE and skips, so an offline or shallow checkout stays workable.
    assert gate.declared_version('[project\nname = "x"\n') is None


def test_this_repositorys_own_manifest_that_does_not_parse_fails_closed():
    # Named as guarded, it is the file the gate protects: present and unreadable is corrupt, not
    # absent, and reading it as "no version declared" would skip the bump check on a broken client.
    with pytest.raises(SystemExit):
        gate.declared_version('[project\nname = "x"\n', guarded=gate.CLIENT_MANIFEST)


def manifest(*dependencies: str, trailing: str = '') -> str:
    """A root manifest declaring these dependencies, and whatever else the case needs after them."""
    listed = ''.join(f'    "{entry}",\n' for entry in dependencies)
    return f'[project]\nname = "positronic"\nversion = "0.2.1"\ndependencies = [\n{listed}{trailing}]\n'


def test_the_root_pin_is_read_from_a_dependency_list():
    assert gate.pinned_version(manifest('positronic-platform-client==0.1.0', 'httpx')) == '0.1.0'
    assert gate.pinned_version(manifest('httpx', 'positronic-platform-client == 2.10.3')) == '2.10.3'
    # The name is matched as a distribution, so the spelling variants an index treats as one match.
    assert gate.pinned_version(manifest('Positronic_Platform_Client==0.1.0')) == '0.1.0'


def test_a_relaxed_or_absent_pin_reads_as_no_pin():
    # Read as absent, and `check` treats that as a FAILURE rather than a reason to skip: deleting
    # the pin reaches the same stale-or-incompatible install as letting it lag.
    assert gate.pinned_version(manifest('httpx', 'pydantic>=2')) is None
    assert gate.pinned_version(manifest('positronic-platform-client')) is None
    assert gate.pinned_version(manifest('positronic-platform-client>=0.1.0')) is None
    assert gate.pinned_version(manifest('positronic-platform-client>=0.1.0,==0.1.0')) is None


def test_a_conditional_pin_is_no_pin():
    # A marker that is false on every supported interpreter installs the client nowhere, while the
    # CLI imports `platform_client` unconditionally — so a fresh install fails at startup.
    assert gate.pinned_version(manifest("positronic-platform-client==0.2.0; python_version < '3'")) is None
    assert gate.pinned_version(manifest("positronic-platform-client==0.2.0; python_version >= '3'")) is None
    assert gate.pinned_version(manifest('positronic-platform-client==0.2.0')) == '0.2.0'


def test_a_deleted_dependency_left_behind_as_a_comment_is_no_pin():
    # The shape a deletion actually leaves: the line commented out rather than removed. Scanned as
    # text it reads as a live pin, which passes the missing-dependency case this gate exists to
    # refuse — so the dependency list is parsed, where a comment does not exist at all.
    left_behind = manifest('httpx', trailing='    # "positronic-platform-client==0.1.0",\n')
    assert gate.pinned_version(left_behind) is None


def test_a_manifest_that_does_not_parse_fails_closed():
    # Present and unreadable is a corrupt guarded file, not an absence to skip past.
    with pytest.raises(SystemExit):
        gate.pinned_version('[project\nname = "positronic"\n')


def test_only_shipped_paths_under_the_client_demand_a_bump():
    paths = [
        'client/platform_client/responses.py',
        'client/README.md',
        'positronic/cli/eval/submit.py',
        'pyproject.toml',
    ]
    # The README ships in the wheel but cannot change what an install runs; the two paths outside
    # `client/` belong to the root distribution, which carries its own version.
    assert gate.shipped_changes(paths) == ['client/platform_client/responses.py']


def test_a_client_test_counts_as_shipped():
    # It sits inside the package directory, so two revisions behind one version would differ.
    assert gate.shipped_changes(['client/platform_client/tests/test_models.py']) == [
        'client/platform_client/tests/test_models.py'
    ]
