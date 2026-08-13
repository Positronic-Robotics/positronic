import os

import configuronic as cfn

from positronic import keys
from positronic.offboard.server import AUTH_HEADER, AUTH_TOKEN_ENV, bearer
from positronic.policy import RemotePolicy, SampledPolicy
from positronic.policy.sampler import BalancedSampler, Sampler
from positronic.utils import nebius


@cfn.config()
def placeholder():
    raise RuntimeError(
        'This config is not supposed to be instantiated, '
        'and is used only to simplify relative imports of other policy configs.'
    )


@cfn.config(weights=None)
def sample(origins: list[cfn.Config], weights: list[float] | None):
    """One could use the following CLI:
    --policy=.sample --policy.origins='[".remote"]' --policy.origins.0.url=<yada-yada>
    """
    return SampledPolicy(*origins, weights=weights)


remote = cfn.Config(RemotePolicy, url='ws://localhost:8000')


@cfn.config()
def bearer_headers():
    token = os.environ.get(AUTH_TOKEN_ENV)
    if not token:
        raise ValueError(f'{AUTH_TOKEN_ENV} is not set; export the endpoint token before running inference')
    return {AUTH_HEADER: bearer(token)}


@cfn.config()
def nebius_bearer_headers():
    return {AUTH_HEADER: bearer(nebius.auth_token())}


# Neither carries a default URL: it names one specific endpoint, and a token must not be handed to
# whichever host a stale default points at.
authed_remote = cfn.Config(RemotePolicy, headers=bearer_headers)
nebius_remote = cfn.Config(RemotePolicy, headers=nebius_bearer_headers)


@cfn.config(balance=2)
def balanced(balance: int):
    return BalancedSampler(balance=balance)


@cfn.config(endpoints={}, weights={}, headers=None, recording_dir=None, sampler=None, group_fields=None)
def production(
    endpoints: dict[str, str],
    weights: dict[str, float],
    headers: dict[str, str] | None,
    recording_dir: str | None,
    sampler: Sampler | None,
    group_fields: list[str] | None,
):
    """Routes each episode to one of several remote endpoints, each named for CLI overrides.

    An endpoint is one URL, so `--policy.endpoints.groot=ws://desktop:8000` adds or repoints one without
    restating the others. `weights` name the same endpoints and set their sampling odds; endpoints left
    out of it weigh 1.0.

    `headers` go to every endpoint, which is what a set of endpoints behind one authenticating proxy
    needs: a sampled run reaches each of them with the same credential. Endpoints wanting different
    credentials are separate policies, sampled through `sample`.
    """
    if not endpoints:
        raise ValueError('At least one endpoint must be given, e.g. --policy.endpoints.groot=ws://desktop:8000')
    if unknown := weights.keys() - endpoints.keys():
        raise ValueError(f'weights name unknown endpoints: {sorted(unknown)}; known are {sorted(endpoints)}')
    # Every Sampler but the default uniform one picks by episode counts alone, so weights would be dropped.
    if weights and sampler is not None:
        raise ValueError(f'weights cannot be combined with {type(sampler).__name__}, which samples by count')
    policies = [RemotePolicy(url, headers=headers, recording_dir=recording_dir) for url in endpoints.values()]
    w = [weights.get(name, 1.0) for name in endpoints] if weights else None
    return SampledPolicy(*policies, weights=w, sampler=sampler, group_fields=group_fields)


@cfn.config()
def phail_single(hostname, w_openpi=1.0, w_groot=1.0, w_act=1.0):
    openpi = RemotePolicy(f'ws://{hostname}:8000')
    groot = RemotePolicy(f'ws://{hostname}:8001')
    act = RemotePolicy(f'ws://{hostname}:8002')

    return SampledPolicy(openpi, groot, act, weights=[w_openpi, w_groot, w_act])


phail_multiple = production.override(
    endpoints={
        'smolvla': 'ws://notebook:8000',
        'act': 'ws://notebook:8001',
        'groot': 'ws://desktop:8000',
        'openpi': 'ws://vm-openpi:8000',
        # DreamZero's 5B wan2.2 needs an H100, so it cannot share the consumer boxes above.
        'dreamzero': 'ws://vm-train2:8000',
    },
    sampler=balanced,
    group_fields=[keys.TASK, 'eval.object', 'eval.tote_placement', 'eval.external_camera'],
)
