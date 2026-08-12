import os
from enum import StrEnum

import configuronic as cfn

from positronic import keys
from positronic.offboard.server import AUTH_HEADER, AUTH_TOKEN_ENV, bearer
from positronic.policy import Policy, RemotePolicy, ReplayPolicy, SampledPolicy
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

# A recorded episode in place of a served model: `--policy=.replay --policy.dataset_path=<dataset>`.
replay = cfn.Config(ReplayPolicy)


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


class EndpointKind(StrEnum):
    """How one named endpoint of a multi-policy run is reached."""

    REMOTE = 'remote'  # a checkpoint served over the wire, dialled at its URL
    REPLAY = 'replay'  # a recorded dataset played back, with nothing served and no network


# The wire contract with whatever writes `--policy.endpoints` as JSON: these names stay stable.
ENDPOINT_KIND = 'kind'
ENDPOINT_URL = 'url'
ENDPOINT_DATASET = 'dataset'
ENDPOINT_EPISODE = 'episode'

# What each kind takes besides `kind` itself, in the order its declaration reads.
ENDPOINT_FIELDS = {EndpointKind.REMOTE: (ENDPOINT_URL,), EndpointKind.REPLAY: (ENDPOINT_DATASET, ENDPOINT_EPISODE)}

# One `endpoints` entry: a bare string is a served endpoint's URL, a mapping declares kind and fields.
EndpointSpec = str | dict[str, str | int | None]


def _endpoint_policy(name: str, spec: EndpointSpec, recording_dir: str | None) -> Policy:
    """The policy one `endpoints` entry names.

    The kind is declared, never inferred from the locator: a URL may be written without a scheme
    (`notebook:8000` is one), which no rule can tell apart from a relative path to a recording.
    """
    fields: dict[str, str | int | None] = {ENDPOINT_URL: spec} if isinstance(spec, str) else dict(spec)
    declared = fields.pop(ENDPOINT_KIND, EndpointKind.REMOTE)
    try:
        kind = EndpointKind(declared)
    except ValueError:
        known = ', '.join(str(k) for k in EndpointKind)
        raise ValueError(f'endpoint {name!r} declares {ENDPOINT_KIND}={declared!r}; the kinds are {known}') from None
    if unknown := sorted(fields.keys() - set(ENDPOINT_FIELDS[kind])):
        raise ValueError(
            f'endpoint {name!r} is {kind} and declares {unknown}, which that kind does not take; '
            f'a {kind} endpoint takes {list(ENDPOINT_FIELDS[kind])}'
        )

    if kind is EndpointKind.REMOTE:
        url = fields.get(ENDPOINT_URL)
        if not isinstance(url, str) or not url:
            raise ValueError(f'endpoint {name!r} is {kind} and names no {ENDPOINT_URL} to dial')
        # `recording_dir` taps the wire between the rig and the server, so only a served endpoint has one.
        return RemotePolicy(url, recording_dir=recording_dir)

    dataset = fields.get(ENDPOINT_DATASET)
    if not isinstance(dataset, str) or not dataset:
        raise ValueError(f'endpoint {name!r} is {kind} and names no {ENDPOINT_DATASET} to play back')
    episode = fields.get(ENDPOINT_EPISODE)
    # ``type`` not ``isinstance``: ``bool`` subclasses ``int``, so a JSON ``true`` would play episode 1.
    if episode is not None and type(episode) is not int:
        raise ValueError(f'endpoint {name!r} declares {ENDPOINT_EPISODE}={episode!r}, which is not an episode index')
    # An entry that names no episode plays the recording's first.
    return ReplayPolicy(dataset, episode=episode if episode is not None else 0)


@cfn.config(endpoints={}, weights={}, recording_dir=None, sampler=None, group_fields=None)
def production(
    endpoints: dict[str, EndpointSpec],
    weights: dict[str, float],
    recording_dir: str | None,
    sampler: Sampler | None,
    group_fields: list[str] | None,
):
    """Routes each episode to one of several named policies, sampled per episode.

    An endpoint is a served checkpoint given as its URL — `--policy.endpoints.groot=ws://desktop:8000`
    adds or repoints one without restating the others — or a mapping declaring what it is:

        --policy.endpoints='{"a": {"kind": "replay", "dataset": "s3://…/run", "episode": 1},
                             "b": "wss://host/api/v1/session"}'

    A `remote` endpoint takes a `url`, a `replay` endpoint a `dataset` to play back and optionally the
    `episode` within it. Whole-mapping form replaces the endpoints this config carries; the per-key form
    adds to them. `weights` name the same endpoints and set their sampling odds; endpoints left out of it
    weigh 1.0.

    Give a `dataset` in the whole-mapping form as an absolute path or a URI. A leading dot is
    configuronic's relative-import sigil inside an override value, so `"./run"` there is read as a
    config to import and the override raises before the run starts. A relative one is reached by the
    per-key form, which resolves against the value it replaces rather than against this config:

        --policy.endpoints='{"a": {"kind": "replay", "dataset": "unset"}}' \
        --policy.endpoints.a.dataset=./run

    The names are for addressing endpoints here, and do not reach the recording: an episode records the
    identity its policy reports — a served endpoint's checkpoint path, a replay's dataset and episode —
    which is what joins it back to the endpoint that produced it. Two replay endpoints on the same
    dataset and episode are refused: nothing tells their episodes apart, so a resumed run could not
    re-attach their counts either.
    """
    if not endpoints:
        raise ValueError('At least one endpoint must be given, e.g. --policy.endpoints.groot=ws://desktop:8000')
    if unknown := weights.keys() - endpoints.keys():
        raise ValueError(f'weights name unknown endpoints: {sorted(unknown)}; known are {sorted(endpoints)}')
    # Every Sampler but the default uniform one picks by episode counts alone, so weights would be dropped.
    if weights and sampler is not None:
        raise ValueError(f'weights cannot be combined with {type(sampler).__name__}, which samples by count')
    policies = [_endpoint_policy(name, spec, recording_dir) for name, spec in endpoints.items()]
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
