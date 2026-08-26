import json

import pytest

from positronic.cfg.policy import file_headers

SECRET = 'ak-do-not-print-me'


def test_file_headers_reads_the_json_object(tmp_path):
    path = tmp_path / 'headers.json'
    path.write_text(json.dumps({'Modal-Key': SECRET, 'Modal-Secret': 'as-also-secret'}))

    assert file_headers(path=str(path)) == {'Modal-Key': SECRET, 'Modal-Secret': 'as-also-secret'}


def test_file_headers_expands_a_tilde_path(tmp_path, monkeypatch):
    monkeypatch.setenv('HOME', str(tmp_path))
    (tmp_path / 'headers.json').write_text(json.dumps({'X-Api-Key': SECRET}))

    assert file_headers(path='~/headers.json') == {'X-Api-Key': SECRET}


def test_file_headers_names_a_path_it_could_not_read(tmp_path):
    absent = tmp_path / 'absent.json'

    with pytest.raises(ValueError) as excinfo:
        file_headers(path=str(absent))

    assert str(absent) in str(excinfo.value)


@pytest.mark.parametrize(
    'content',
    [json.dumps([SECRET]), json.dumps({'Modal-Key': [SECRET]}), json.dumps({}), f'Modal-Key: {SECRET}', SECRET],
    ids=['array', 'non-string value', 'empty object', 'not json', 'bare token'],
)
def test_file_headers_refuses_what_is_not_a_header_set_without_quoting_it(tmp_path, content):
    """The refusal names the path and quotes nothing from the content, by message or by chain."""
    path = tmp_path / 'headers.json'
    path.write_text(content)

    with pytest.raises(ValueError) as excinfo:
        file_headers(path=str(path))

    assert str(path) in str(excinfo.value)
    assert SECRET not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__
