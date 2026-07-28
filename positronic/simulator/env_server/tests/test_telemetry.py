import json

from positronic import telemetry as positronic_telemetry
from positronic.simulator.env_server import telemetry


def test_env_spans_parse_with_positronic_reader(tmp_path):
    """The positronic-free env-server writer emits the same OTLP/JSON-lines shape the positronic reader parses,
    with physics/render nested under the server ``env.step`` span."""
    with telemetry.bind(tmp_path, 'run-env'):
        with telemetry.span(telemetry.SPAN_ENV_STEP):
            with telemetry.span('physics'):
                pass
            with telemetry.span('render'):
                pass

    spans = {rec.name: rec for rec in positronic_telemetry.read_spans(tmp_path / 'env.spans.jsonl')}
    assert set(spans) == {telemetry.SPAN_ENV_STEP, 'render', 'physics'}
    assert spans[telemetry.SPAN_ENV_STEP].parent_id is None
    assert spans['physics'].parent_id == spans[telemetry.SPAN_ENV_STEP].span_id
    assert spans['render'].parent_id == spans[telemetry.SPAN_ENV_STEP].span_id
    for rec in spans.values():
        assert len(rec.span_id) == 16
        assert rec.end_ns >= rec.start_ns

    meta = json.loads((tmp_path / 'env.meta.json').read_text())
    assert meta['process'] == 'env'
    assert meta['run_id'] == 'run-env'


def test_env_span_inert_when_unbound(tmp_path):
    with telemetry.span(telemetry.SPAN_ENV_STEP):
        pass
    assert not telemetry.active()
    assert not (tmp_path / 'env.spans.jsonl').exists()


def test_bind_from_env_reads_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv(telemetry.ENV_TELEMETRY_DIR, str(tmp_path))
    monkeypatch.setenv(telemetry.ENV_RUN_ID, 'run-fromenv')
    with telemetry.bind_from_env():
        assert telemetry.active()
        with telemetry.span(telemetry.SPAN_ENV_RESET):
            pass
    assert not telemetry.active()
    spans = list(positronic_telemetry.read_spans(tmp_path / 'env.spans.jsonl'))
    assert [rec.name for rec in spans] == [telemetry.SPAN_ENV_RESET]
