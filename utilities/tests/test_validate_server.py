from utilities.validate_server import _build_inference_command


def _command(**overrides) -> list[str]:
    arguments = {
        'uv_path': 'uv',
        'eval_ref': '.sim.positronic.stack_cubes',
        'wire_name': 'websocket_tls',
        'address_args': ['--policy.host=gpu-host', '--policy.port=443'],
        'query': '',
        'policy_ref': '.authed_remote',
        'model_id': 'm',
        'output_dir': 's3://runs/m',
        'extra_args': [],
    }
    return _build_inference_command(**{**arguments, **overrides})


def test_a_socket_wire_names_its_socket_and_neither_a_host_nor_a_port():
    """The wire's own address reaches the eval subprocess, so a socket run names no host and no port."""
    command = _command(wire_name='websocket_unix', address_args=['--policy.uds=/run/policy.sock'])

    assert '--policy.wire=websocket_unix' in command
    assert '--policy.uds=/run/policy.sock' in command
    assert not [flag for flag in command if flag.startswith(('--policy.host', '--policy.port'))]


def test_the_command_names_the_wire_the_server_and_the_model_as_policy_flags():
    command = _command()
    assert command[:6] == ['uv', 'run', '--locked', 'positronic', 'eval', 'run']
    assert command[6:] == [
        '--eval=.sim.positronic.stack_cubes',
        '--policy=.authed_remote',
        '--policy.wire=websocket_tls',
        '--policy.host=gpu-host',
        '--policy.port=443',
        '--policy.model=m',
        '--output_dir=s3://runs/m',
    ]


def test_session_params_reach_the_command_only_where_there_are_any():
    assert '--policy.query=fps=10' in _command(query='fps=10')
    assert not any(part.startswith('--policy.query') for part in _command(query=''))


def test_a_path_shaped_model_id_reaches_the_command_as_written():
    """The policy percent-encodes the id itself, so the flag carries it as the server advertised it."""
    assert '--policy.model=GEAR/DreamZero' in _command(model_id='GEAR/DreamZero')
    assert '--policy.model=s3://b/ckpt#1' in _command(model_id='s3://b/ckpt#1')


def test_extra_arguments_follow_the_output_dir():
    assert _command(extra_args=['--eval.trial_count=3'])[-1] == '--eval.trial_count=3'
