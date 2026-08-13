"""The image reference a submission carries."""

from __future__ import annotations

import pytest
from platform_client.policy_images import PolicyImage
from pydantic import BaseModel, ValidationError


def test_a_reference_splits_into_its_name_and_its_digest():
    unpinned = PolicyImage('org/policy:v1')
    assert (unpinned.name, unpinned.digest) == ('org/policy:v1', None)
    pinned = PolicyImage('org/policy:v1@sha256:abc123')
    assert (pinned.name, pinned.digest) == ('org/policy:v1', 'sha256:abc123')


def test_pinning_replaces_a_digest_rather_than_appending_one():
    assert PolicyImage('org/policy:v1').pinned('sha256:abc') == 'org/policy:v1@sha256:abc'
    # An already-pinned reference re-pins to the resolved digest — never `repo@sha256:…@sha256:…`.
    repinned = PolicyImage('org/policy@sha256:aaa').pinned('sha256:bbb')
    assert repinned == 'org/policy@sha256:bbb' and repinned.count('@') == 1
    assert isinstance(repinned, PolicyImage)


@pytest.mark.parametrize(
    'value',
    [
        '',
        '@sha256:abc',
        'org/ policy',
        'org/policy@',
        'org/policy@nope',
        'org/policy@sha256:zz',
        'org/policy@sha256:a@sha256:b',
        'org/policy:',
        'org/policy:bad:tag',
        'org/policy:v1 ',
        'org//policy',
        'org/policy:-v1',
    ],
)
def test_a_reference_no_registry_could_resolve_is_refused_here(value: str):
    # The gateway resolves this reference against a registry, and an unpullable image is a CHARGED
    # terminal submission — so the shapes a typo makes are refused here rather than billed later.
    with pytest.raises(ValueError):
        PolicyImage(value)


@pytest.mark.parametrize(
    'value',
    [
        'policy',
        'org/policy',
        'org/policy:v1',
        'org/policy@sha256:abc',
        'org/team/policy:v1',
        'org/policy:v1.2_beta-3',
        'reg.io/org/p:v1',
        'reg.io:5000/org/p:v1',
        'reg.io:5000/org/p:v1@sha256:abc',
    ],
)
def test_a_reference_a_registry_can_resolve_is_kept(value: str):
    # `reg.io:5000/...` is a registry PORT, and a tag carries dots, underscores and hyphens of its
    # own — the grammar that refuses `org/policy:bad:tag` must admit all of these unchanged.
    assert PolicyImage(value) == value


def test_references_are_plain_strings_on_the_wire():
    class M(BaseModel):
        image: PolicyImage

    parsed = M.model_validate({'image': 'org/policy:v1@sha256:abc'})
    assert isinstance(parsed.image, PolicyImage) and parsed.image.digest == 'sha256:abc'
    assert parsed.model_dump() == {'image': 'org/policy:v1@sha256:abc'}
    with pytest.raises(ValidationError):
        M.model_validate({'image': '@sha256:abc'})
