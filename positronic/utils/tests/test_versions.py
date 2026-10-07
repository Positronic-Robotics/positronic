"""Published versions remain usable until a release explicitly removes their implementation."""

from datetime import date

import pytest

from positronic.utils.versions import Deprecation, Version, resolve_version


def test_passing_the_announced_date_does_not_remove_support():
    notice = Deprecation(date(2000, 1, 1), date(2000, 7, 1), 'Upgrade the server to v2.')
    implementations = {1: Version('old', notice), 2: Version('new')}
    with pytest.warns(FutureWarning, match=r'v1.*2000-01-01.*2000-07-01.*Upgrade the server'):
        assert resolve_version(implementations, 1, 'test protocol') == 'old'
    assert resolve_version(implementations, 2, 'test protocol') == 'new'


def test_removed_versions_keep_migration_instructions():
    notice = Deprecation(date(2020, 1, 1), date(2020, 7, 1), 'Upgrade the server to v2.')
    versions = {1: Version(None, notice, removed_on=date(2020, 7, 2)), 2: Version('new')}
    with pytest.raises(ValueError, match=r'test component v1 has been removed.*Upgrade the server'):
        resolve_version(versions, 1, 'test component')
    with pytest.raises(ValueError, match='before its announced deadline'):
        Version(None, notice, removed_on=date(2020, 6, 30))


@pytest.mark.parametrize('version', [0, -1, True, 1.0, '1', None, [], {}])
def test_malformed_versions_are_rejected(version):
    with pytest.raises(ValueError, match='positive integer'):
        resolve_version({1: Version('first')}, version, 'test protocol')


def test_an_unknown_version_is_not_substituted():
    with pytest.raises(ValueError, match=r'Unsupported test protocol version 3.*\[1, 2\].*Upgrade'):
        resolve_version({1: Version('first'), 2: Version('second')}, 3, 'test protocol')


def test_deprecation_requires_notice_and_a_migration_path():
    with pytest.raises(ValueError, match='follow'):
        Deprecation(date(2020, 7, 1), date(2020, 7, 1), 'Upgrade the server.')
    with pytest.raises(ValueError, match='migrate'):
        Deprecation(date(2020, 1, 1), date(2020, 7, 1), '')
    with pytest.raises(ValueError, match='retain'):
        Version(None)
