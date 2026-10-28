import json
from urllib.parse import parse_qsl

import pytest
from platform_client.policy_container import AUTH_TOKEN_ENV
from positronic_model_server.spec import parse_params

from positronic import keys
from positronic.cfg import policy as policy_cfg
from positronic.cfg.eval.sim.positronic import stack_cubes_collect
from positronic.cfg.policy import file_headers
from positronic.policy import RemotePolicy

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


@pytest.mark.parametrize('name', ['remote', 'authed_remote', 'nebius_remote', 'file_authed_remote'])
def test_a_remote_policy_config_carries_an_address_its_network_flags_reach(name):
    """`--policy.address.host` overrides a nested config, so every policy config holds one to override."""
    overridden = getattr(policy_cfg, name).override(**{'address.host': 'gpu-host', 'address.port': 443})

    address = overridden.kwargs['address'].instantiate()
    assert (address.host, address.port) == ('gpu-host', 443)


def test_the_collection_eval_records_every_camera_that_the_teleop_collection_policy_shows():
    recorded = stack_cubes_collect.kwargs['camera_dict']
    assert set(policy_cfg.TELEOP_COLLECT_CAMERAS) <= set(recorded)
    assert {keys.WRIST_IMAGE, keys.EXTERIOR_IMAGE} <= set(recorded)


@pytest.mark.parametrize(
    ('config', 'cameras'),
    [
        pytest.param(policy_cfg.teleop_collect, policy_cfg.TELEOP_COLLECT_CAMERAS, id='teleop_collect'),
        pytest.param(policy_cfg.teleop_molmo, policy_cfg.TELEOP_MOLMO_CAMERAS, id='teleop_molmo'),
    ],
)
def test_a_teleop_collection_policy_asks_the_service_for_its_cameras_with_the_person_at_the_back(config, cameras):
    query = config.instantiate()._client.session_url.split('?', 1)[1]
    assert parse_params(parse_qsl(query)) == {'operator_position': 'BACK', 'cameras': list(cameras)}


@pytest.mark.parametrize('name', ['teleop', 'teleop_network'])
def test_a_run_sets_one_session_parameter_of_a_teleop_policy(monkeypatch, name):
    monkeypatch.setenv(AUTH_TOKEN_ENV, 'token')
    config = getattr(policy_cfg, name).override(**{'address.query.operator_position': 'BACK'})
    query = config.instantiate()._client.session_url.split('?', 1)[1]
    assert parse_params(parse_qsl(query)) == {'operator_position': 'BACK'}


def test_the_teleop_policy_is_a_remote_policy_on_the_socket_of_the_service(tmp_path):
    socket = tmp_path / 'teleop.sock'
    teleop = policy_cfg.teleop.override(**{'address.uds': str(socket)}).instantiate()
    plain = policy_cfg.remote.override(
        wire='websocket_unix', address=policy_cfg.socket_address.override(uds=str(socket))
    ).instantiate()
    assert isinstance(teleop, RemotePolicy) and isinstance(plain, RemotePolicy)
    assert teleop._client.session_url == plain._client.session_url == f'ws+unix://{socket}/api/v1/session'
