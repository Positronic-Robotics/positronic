import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_server import _model_url  # noqa: E402


@pytest.mark.parametrize(
    ('url', 'expected'),
    [
        ('localhost:8000', 'localhost:8000/api/v1/session/m'),
        ('https://host', 'https://host/api/v1/session/m'),
        ('https://host/', 'https://host/api/v1/session/m'),
        # A URL already naming a session addresses the endpoint, not a path to append under.
        ('https://host/api/v1/session', 'https://host/api/v1/session/m'),
        ('https://host/api/v1/session/other', 'https://host/api/v1/session/m'),
        # Session params belong to the connection, so they outlive the model the URL points at.
        ('https://host?fps=10', 'https://host/api/v1/session/m?fps=10'),
        ('ws://host:9000/api/v1/session?pad=false', 'ws://host:9000/api/v1/session/m?pad=false'),
    ],
)
def test_every_accepted_url_form_addresses_the_model(url, expected):
    assert _model_url(url, 'm') == expected


def test_a_path_shaped_model_id_keeps_its_separators():
    assert _model_url('https://host', 'GEAR/DreamZero') == 'https://host/api/v1/session/GEAR/DreamZero'


def test_a_model_id_that_would_end_the_path_is_encoded():
    assert _model_url('https://host', 's3://b/ckpt#1') == 'https://host/api/v1/session/s3%3A//b/ckpt%231'
