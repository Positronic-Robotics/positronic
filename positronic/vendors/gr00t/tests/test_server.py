from positronic.vendors.gr00t import server as gr00t_server


def _source(monkeypatch, checkpoints: list[str]) -> gr00t_server.Gr00tSource:
    monkeypatch.setattr(gr00t_server, 'list_checkpoints', lambda _dir, prefix='': checkpoints)
    return gr00t_server.Gr00tSource('s3://bucket/exp')


def test_zero_padded_checkpoints_are_served_under_the_id_they_advertise(monkeypatch):
    """The padding is the directory's, not the model's: a client asking for the advertised id must be
    recorded under that same id, or analysis splits one checkpoint in two."""
    source = _source(monkeypatch, ['checkpoint-005000', 'checkpoint-010000'])

    assert source.get_models() == ['5000', '10000']
    assert source.resolve('5000') == '5000'
    assert source.resolve(None) == '10000'
    # The raw suffix survives only where it is needed — reaching the directory.
    assert source._raw_for('5000') == '005000'
