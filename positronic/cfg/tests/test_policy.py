import positronic.cfg.policy as cfg_policy


class _SpyRemotePolicy:
    """Captures what `production` hands each endpoint, without opening a connection."""

    def __init__(self, url: str, *, headers=None, recording_dir=None, **kwargs):
        self.url = url
        self.headers = headers
        self.recording_dir = recording_dir


def _endpoints_of(monkeypatch, preset, **overrides) -> list[_SpyRemotePolicy]:
    built: list[_SpyRemotePolicy] = []

    def spy(url, **kwargs):
        policy = _SpyRemotePolicy(url, **kwargs)
        built.append(policy)
        return policy

    monkeypatch.setattr(cfg_policy, 'RemotePolicy', spy)
    monkeypatch.setattr(cfg_policy, 'SampledPolicy', lambda *policies, **kwargs: policies)
    preset.override(**overrides).instantiate()
    return built


def test_headers_reach_every_sampled_endpoint(monkeypatch):
    headers = {'X-Proxy-Id': 'id', 'X-Proxy-Secret': 'secret'}
    built = _endpoints_of(
        monkeypatch, cfg_policy.production, endpoints={'a': 'ws://a:8000', 'b': 'ws://b:8000'}, headers=headers
    )
    assert [p.url for p in built] == ['ws://a:8000', 'ws://b:8000']
    assert [p.headers for p in built] == [headers, headers]


def test_no_headers_by_default(monkeypatch):
    built = _endpoints_of(monkeypatch, cfg_policy.production, endpoints={'a': 'ws://a:8000'})
    assert [p.headers for p in built] == [None]
