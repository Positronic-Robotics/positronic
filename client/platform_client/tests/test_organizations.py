import pytest
from platform_client.ids import OrgSlug, UserId
from platform_client.responses import MeResponse
from pydantic import ValidationError

IDENTITY = {'user_id': UserId(10).to_str(), 'tenant': 't', 'plan': 'p', 'quota': []}


def test_identity_lists_shared_and_personal_organizations_without_changing_the_default():
    answer = {**IDENTITY, 'organizations': ['acme', 'user-a'], 'personal_org': 'user-a'}
    me = MeResponse.model_validate(answer)
    assert me.organizations == [OrgSlug('acme'), OrgSlug('user-a')]
    assert me.personal_org == 'user-a'
    assert MeResponse.model_validate_json(me.model_dump_json()) == me


def test_identity_from_an_older_platform_has_an_independent_empty_list():
    first = MeResponse.model_validate(IDENTITY)
    first.organizations.append(OrgSlug('acme'))
    assert MeResponse.model_validate(IDENTITY).organizations == []


@pytest.mark.parametrize('organizations', [None, 'acme', [1], ['']])
def test_identity_refuses_malformed_organization_memberships(organizations):
    with pytest.raises(ValidationError):
        MeResponse.model_validate({**IDENTITY, 'organizations': organizations})
