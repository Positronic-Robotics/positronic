"""The gate that keeps every workspace member's code, its version, the root's pin on it, and
the release that publishes it in step."""

import pytest
from packaging.utils import canonicalize_name

from utilities import check_member_version_bumps as gate

CLIENT = 'positronic-platform-client'


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
    # Named as guarded, it is a file the gate protects: present and unreadable is corrupt, not
    # absent, and reading it as "no version declared" would skip the bump check on a broken member.
    with pytest.raises(SystemExit):
        gate.declared_version('[project\nname = "x"\n', guarded='client/pyproject.toml')


def manifest(*dependencies: str, trailing: str = '') -> str:
    """A root manifest declaring these dependencies, and whatever else the case needs after them."""
    listed = ''.join(f'    "{entry}",\n' for entry in dependencies)
    return f'[project]\nname = "positronic"\nversion = "0.2.1"\ndependencies = [\n{listed}{trailing}]\n'


def test_the_root_pin_is_read_from_a_dependency_list():
    assert gate.pinned_version(manifest('positronic-platform-client==0.1.0', 'httpx'), CLIENT) == '0.1.0'
    assert gate.pinned_version(manifest('httpx', 'positronic-platform-client == 2.10.3'), CLIENT) == '2.10.3'
    # The name is matched as a distribution, so the spelling variants an index treats as one match.
    assert gate.pinned_version(manifest('Positronic_Platform_Client==0.1.0'), CLIENT) == '0.1.0'


def test_a_relaxed_or_absent_pin_reads_as_no_pin():
    # Read as absent, and `check` treats that as a FAILURE rather than a reason to skip: deleting
    # the pin reaches the same stale-or-incompatible install as letting it lag.
    assert gate.pinned_version(manifest('httpx', 'pydantic>=2'), CLIENT) is None
    assert gate.pinned_version(manifest('positronic-platform-client'), CLIENT) is None
    assert gate.pinned_version(manifest('positronic-platform-client>=0.1.0'), CLIENT) is None
    assert gate.pinned_version(manifest('positronic-platform-client>=0.1.0,==0.1.0'), CLIENT) is None


def test_a_conditional_pin_is_no_pin():
    # A marker that is false on every supported interpreter installs the client nowhere, while the
    # CLI imports `platform_client` unconditionally — so a fresh install fails at startup.
    assert gate.pinned_version(manifest("positronic-platform-client==0.2.0; python_version < '3'"), CLIENT) is None
    assert gate.pinned_version(manifest("positronic-platform-client==0.2.0; python_version >= '3'"), CLIENT) is None
    assert gate.pinned_version(manifest('positronic-platform-client==0.2.0'), CLIENT) == '0.2.0'


def test_a_deleted_dependency_left_behind_as_a_comment_is_no_pin():
    # The shape a deletion actually leaves: the line commented out rather than removed. Scanned as
    # text it reads as a live pin, which passes the missing-dependency case this gate exists to
    # refuse — so the dependency list is parsed, where a comment does not exist at all.
    left_behind = manifest('httpx', trailing='    # "positronic-platform-client==0.1.0",\n')
    assert gate.pinned_version(left_behind, CLIENT) is None


def test_a_manifest_that_does_not_parse_fails_closed():
    # Present and unreadable is a corrupt guarded file, not an absence to skip past.
    with pytest.raises(SystemExit):
        gate.pinned_version('[project\nname = "positronic"\n', CLIENT)


def test_only_shipped_paths_under_the_member_demand_a_bump():
    paths = [
        'client/platform_client/responses.py',
        'client/README.md',
        'positronic/cli/eval/submit.py',
        'pyproject.toml',
    ]
    # The README ships in the wheel but cannot change what an install runs; the two paths outside
    # `client/` belong to the root distribution, which carries its own version.
    assert gate.shipped_changes(paths, 'client') == ['client/platform_client/responses.py']


def test_a_member_test_counts_as_shipped():
    # It sits inside the package directory, so two revisions behind one version would differ.
    assert gate.shipped_changes(['client/platform_client/tests/test_models.py'], 'client') == [
        'client/platform_client/tests/test_models.py'
    ]


def test_one_members_paths_are_not_anothers():
    """The gate judges each member against its own manifest, so a change under one must not demand
    a bump of the other — which would make every member's version move together."""
    paths = ['vocabulary/eval_vocabulary/progress.py', 'client/platform_client/responses.py']

    assert gate.shipped_changes(paths, 'vocabulary') == ['vocabulary/eval_vocabulary/progress.py']


def test_the_members_come_from_the_root_workspace():
    """Listed here, a member added later would publish ungated — which is the failure this gate
    exists to catch, one level up."""
    members = gate.workspace_members('[tool.uv.workspace]\nmembers = ["client", "vocabulary"]\n')

    assert members == ['client', 'vocabulary']


def test_this_repository_declares_every_member_this_gate_then_judges():
    """The one that binds: a member added to the workspace without a manifest, or without a name,
    raises rather than being skipped."""
    root = (gate.REPO_ROOT / gate.ROOT_MANIFEST).read_text()

    for member in gate.workspace_members(root):
        manifest = (gate.REPO_ROOT / member / 'pyproject.toml').read_text()
        assert gate.distribution_name(manifest, guarded=member)
        assert gate.declared_version(manifest, guarded=member)


def test_a_member_spelled_another_way_is_still_held_to_its_pin():
    """`EXACTLY_PINNED` is matched the way an index matches a name, so a manifest declaring
    `Positronic_Platform_Client` does not slip the pin requirement on a spelling."""
    assert canonicalize_name('Positronic_Platform_Client') in gate.EXACTLY_PINNED


def test_a_workspace_declaring_no_members_reads_as_none():
    assert gate.workspace_members('[project]\nname = "positronic"\n') == []


RELEASE = """
jobs:
  publish-client-pypi:
    steps:
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: client/dist
  publish-vocabulary-pypi:
    steps:
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: vocabulary/dist
  publish-pypi:
    needs: [publish-client-pypi, publish-vocabulary-pypi]
    steps:
      - uses: pypa/gh-action-pypi-publish@release/v1
"""


def test_a_member_nothing_uploads_is_a_failure():
    """The gate demands a bump for every member the root declares, so a member with no publish job
    is bumped forever and published never."""
    failures = gate.release_failures(['client', 'vocabulary', 'widgets'], RELEASE)

    assert len(failures) == 1
    assert 'uploads no widgets/dist' in failures[0]


def test_a_member_the_root_does_not_wait_for_is_a_failure():
    """An install of the root resolves the members it requires, so the root's upload must follow
    theirs."""
    release = RELEASE.replace('needs: [publish-client-pypi, publish-vocabulary-pypi]', 'needs: [publish-client-pypi]')

    failures = gate.release_failures(['client', 'vocabulary'], release)

    assert len(failures) == 1
    assert 'does not need `publish-vocabulary-pypi`' in failures[0]


def test_a_member_published_and_waited_for_passes():
    """The boundary the two failures above are measured from: neither fires on the shape the
    workflow already has, and the root's own upload names no member, so it is not read as one."""
    assert gate.release_failures(['client', 'vocabulary'], RELEASE) == []
    assert gate.publishing_jobs(gate.release_jobs(RELEASE)) == {
        'client': 'publish-client-pypi',
        'vocabulary': 'publish-vocabulary-pypi',
    }


def test_this_repository_publishes_every_member_it_gates():
    """The one that binds: the workflow this repository ships covers the members it declares."""
    root = (gate.REPO_ROOT / gate.ROOT_MANIFEST).read_text()
    workflow = (gate.REPO_ROOT / gate.RELEASE_WORKFLOW).read_text()

    assert gate.release_failures(gate.workspace_members(root), workflow) == []
