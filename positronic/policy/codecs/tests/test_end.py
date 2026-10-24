import numpy as np
import pytest

from positronic import keys
from positronic.cfg import codecs as cfg_codecs
from positronic.drivers.roboarm.command import PositionControl
from positronic.policy.codecs import ACTION, EndCommand, ObservationCodec, PolicyCommandError, SetControlMode
from positronic.policy.codecs.action import AbsolutePositionAction
from positronic.policy.keys import END

POSE = np.array([0.1, -0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 0.5])


def _action():
    return AbsolutePositionAction(tgt_ee_pose_key='pose', tgt_grip_key='grip')


@pytest.mark.parametrize(
    'codec',
    [
        _action() & EndCommand(),
        ObservationCodec(state={}, images={}) & _action() & EndCommand(),
        (SetControlMode(PositionControl()) | _action()) & EndCommand(),
    ],
    ids=['action', 'observation_and_action', 'control_mode_then_action'],
)
def test_end_command_in_parallel_with_the_action_codec_carries_end_beside_the_commands(codec):
    plain, ending = codec.decode([{ACTION: POSE}, {ACTION: POSE, END: 'complete'}])
    assert END not in plain and keys.ROBOT_COMMAND in plain
    assert ending[END] == 'complete' and keys.ROBOT_COMMAND in ending


def test_an_action_codec_alone_drops_end():
    assert END not in _action().decode([{ACTION: POSE, END: 'complete'}])[0]


def test_end_command_maps_the_named_model_field_and_passes_nothing_else():
    assert EndCommand('done').decode([{ACTION: POSE, 'done': 'give_up'}, {ACTION: POSE}]) == [{END: 'give_up'}, {}]


def test_an_element_with_only_end_does_not_decode():
    with pytest.raises(KeyError):
        (_action() & EndCommand()).decode([{END: 'complete'}])
    with pytest.raises(PolicyCommandError):
        (_action() & EndCommand()).wrap(lambda obs: [{END: 'complete'}])({})


def test_end_command_composes_with_the_configured_server_codec():
    configured = cfg_codecs.compose_data.override(obs=cfg_codecs.eepose_obs, action=cfg_codecs.absolute_pos_action)
    codec = configured.instantiate()
    [element] = (codec & EndCommand()).decode([{ACTION: POSE, END: 'complete'}])
    assert element[END] == 'complete' and keys.ROBOT_COMMAND in element
