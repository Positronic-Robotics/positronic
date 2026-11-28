import pytest

from positronic.policy.keys import LICENSE, NO_LICENSE, SERVER, SOURCES, TELEOPERATOR, TYPE
from positronic.policy.license import LicenseRule
from positronic.policy.tests.test_dagger import ALLOWED, LICENSED, POLICY, RULE, _policy, _record, _teleoperator

FLUX = 'flux-non-commercial'
POLICY_LICENSE = f'{SOURCES}.{POLICY}.{LICENSE}'
SERVER_LICENSE = f'{SOURCES}.{POLICY}.{SERVER}.{LICENSE}'


def test_a_record_where_each_source_has_an_allowed_license_is_kept(tmp_path):
    record = _record(tmp_path, [_policy(), _teleoperator()])

    assert RULE.refused(record) == {}


def test_a_driver_with_a_license_that_is_not_allowed_is_refused(tmp_path):
    record = _record(tmp_path, [_policy(), _teleoperator()], static={**LICENSED, POLICY_LICENSE: 'gpl-3.0'})

    assert RULE.refused(record) == {POLICY: 'gpl-3.0'}


@pytest.mark.parametrize('static', [{}, {POLICY_LICENSE: NO_LICENSE}])
def test_a_source_with_a_missing_license_is_refused(tmp_path, static):
    static = {f'{SOURCES}.{POLICY}.{TYPE}': 'act', f'{SOURCES}.{TELEOPERATOR}.{LICENSE}': ALLOWED, **static}
    record = _record(tmp_path, [_policy(), _teleoperator()], static=static)

    assert RULE.refused(record) == {POLICY: NO_LICENSE}


def test_a_source_with_no_entry_in_the_record_is_refused(tmp_path):
    record = _record(tmp_path, [_policy(), _teleoperator()], static={POLICY_LICENSE: ALLOWED})

    assert RULE.refused(record) == {TELEOPERATOR: NO_LICENSE}


@pytest.mark.parametrize(
    ('licenses', 'refused'),
    [
        ({SERVER_LICENSE: ALLOWED}, {}),
        ({SERVER_LICENSE: FLUX}, {POLICY: FLUX}),
        ({POLICY_LICENSE: ALLOWED, SERVER_LICENSE: FLUX}, {POLICY: FLUX}),
        ({POLICY_LICENSE: FLUX, SERVER_LICENSE: ALLOWED}, {POLICY: FLUX}),
    ],
)
def test_the_rule_reads_the_license_of_a_remote_source_from_its_server_metadata_and_allows_each_license(
    tmp_path, licenses, refused
):
    static = {f'{SOURCES}.{POLICY}.{TYPE}': 'remote', f'{SOURCES}.{TELEOPERATOR}.{LICENSE}': ALLOWED, **licenses}
    record = _record(tmp_path, [_policy(), _teleoperator()], static=static)

    assert RULE.refused(record) == refused


def test_a_source_that_only_sends_shadow_actions_is_not_checked(tmp_path):
    record = _record(tmp_path, [_teleoperator()] * 2, static={f'{SOURCES}.{TELEOPERATOR}.{LICENSE}': ALLOWED})

    assert RULE.refused(record) == {}


def test_the_licenses_that_allow_training_are_not_one_string():
    with pytest.raises(TypeError, match=r"such as \['internal'\], not the string 'internal'"):
        LicenseRule('internal')


def test_some_license_allows_training():
    with pytest.raises(ValueError, match='No license allows training'):
        LicenseRule([])


def test_a_missing_license_cannot_allow_training():
    with pytest.raises(ValueError, match="'none' is a missing license"):
        LicenseRule({ALLOWED, NO_LICENSE})
