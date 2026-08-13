"""The gate that keeps the client's code, its version, and the root's pin on it in step."""

from utilities import check_client_version_bump as gate


def test_a_later_version_increases():
    assert gate.increases('0.1.0', '0.2.0')
    assert gate.increases('0.1.0', '0.1.1')
    assert gate.increases('0.9.0', '0.10.0')  # numeric, so 10 is not read as older than 9


def test_an_unchanged_or_backwards_version_does_not_increase():
    assert not gate.increases('0.2.0', '0.2.0')
    assert not gate.increases('0.2.0', '0.1.0')


def test_a_padded_zero_is_the_same_version():
    # packaging reads 1.0 and 1.0.0 as one version, so a release "bumped" that way ships as its
    # predecessor and `skip-existing` skips it.
    assert not gate.increases('1.0', '1.0.0')
    assert not gate.increases('1.0.0', '1.0')


def test_the_declared_version_is_read_from_a_manifest():
    assert gate.declared_version('[project]\nname = "x"\nversion = "0.3.1"\n') == '0.3.1'
    assert gate.declared_version("version = '0.3.1'") == '0.3.1'
    assert gate.declared_version('[project]\nname = "x"\n') is None


def test_the_root_pin_is_read_from_a_dependency_list():
    assert gate.pinned_version('dependencies = ["positronic-platform-client==0.1.0", "httpx"]') == '0.1.0'
    assert gate.pinned_version('    "positronic-platform-client==2.10.3",\n') == '2.10.3'


def test_a_root_that_pins_no_client_reads_as_no_pin():
    # Not a failure: the gate says so on stderr and leaves the pin check alone.
    assert gate.pinned_version('dependencies = ["httpx", "pydantic>=2"]') is None
    assert gate.pinned_version('dependencies = ["positronic-platform-client"]') is None


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
