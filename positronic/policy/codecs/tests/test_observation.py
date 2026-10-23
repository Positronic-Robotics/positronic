import json
from typing import cast

import numpy as np
import pytest
from positronic_model_server.spec import ARGS, component

from pimm.time import RECEIVED_WALL, RECEIVED_WORLD
from positronic import keys as obs_keys
from positronic.cfg.codecs import image_observation
from positronic.dataset.episode import EpisodeContainer
from positronic.dataset.tests.utils import DummySignal
from positronic.dataset.vector import RECORDED_TIME
from positronic.drivers.recording import HARNESS_WORLD
from positronic.policy.codecs import ACTION, LEROBOT_FEATURES, BimanualJointsAction
from positronic.policy.codecs.observation import (
    PackFields,
    RenameObservationFields,
    StateObservationCodec,
    bimanual_joints_state,
)
from positronic.policy.compatibility import ObservationCodecV1
from positronic.policy.spec import from_spec


def test_state_observation_spec_round_trip():
    description = {
        'name': 'state_observation',
        'version': 1,
        'args': {'state': {'state': {'q': 2}}, 'task_field': 'prompt'},
    }
    codec = StateObservationCodec(state={'state': {'q': 2}}, task_field='prompt')
    assert codec.to_spec() == description
    restored = from_spec(description)
    assert isinstance(restored, StateObservationCodec)
    assert restored.to_spec() == description
    encoded = restored.encode({'q': [1, 2], obs_keys.TASK: 'pick'})
    assert encoded['prompt'] == 'pick'
    np.testing.assert_array_equal(encoded['state'], [1, 2])


@pytest.mark.parametrize('version', [None, 1])
@pytest.mark.parametrize('task', [None, '', 'pick'])
def test_observation_v1_keeps_images_state_prompt_and_training(version, task):
    description = {
        'name': 'observation_codec',
        'args': {
            'state': {'state': {'q': 2, 'grip': 1}},
            'images': {'image': ['camera', [4, 4]]},
            'task_field': 'prompt',
        },
    }
    if version is not None:
        description['version'] = version
    codec = from_spec(json.loads(json.dumps(description)))
    assert isinstance(codec, ObservationCodecV1)
    assert codec.to_spec() == {**description, 'version': 1}
    frame = np.arange(24, dtype=np.uint8).reshape(2, 4, 3)
    expected_image = np.pad(frame, ((1, 1), (0, 0), (0, 0)))
    inputs = {'q': np.array([1, 2]), 'grip': 0.25, 'camera': frame}
    if task is not None:
        inputs[obs_keys.TASK] = task
    encoded = codec.encode(inputs)
    assert set(encoded) == {'state', 'image'} | ({'prompt'} if task is not None else set())
    np.testing.assert_array_equal(encoded['state'], np.array([1, 2, 0.25], dtype=np.float32), strict=True)
    np.testing.assert_array_equal(encoded['image'], expected_image, strict=True)
    if task is not None:
        assert encoded['prompt'] == task

    signals = {name: DummySignal([100], [value]) for name, value in inputs.items() if name != obs_keys.TASK}
    if task is not None:
        signals[obs_keys.TASK] = task
    episode = EpisodeContainer(signals, meta={'label': 'test'})
    training = codec.training_encoder(episode)
    assert set(training) == {'state', 'image', 'task'}
    assert training.meta == episode.meta
    assert training['task'] == (task or '')
    np.testing.assert_array_equal(training['state'].values()[0], encoded['state'], strict=True)
    np.testing.assert_array_equal(training['image'].values()[0], encoded['image'], strict=True)
    assert codec.meta == {}
    assert codec.training_encoder.meta == {
        LEROBOT_FEATURES: {
            'state': {'shape': (3,), 'dtype': 'float32', 'names': ['q', 'grip']},
            'image': {'shape': (4, 4, 3), 'dtype': 'video', 'names': ['height', 'width', 'channel']},
        }
    }


@pytest.mark.parametrize('output', ['value', 'task'])
def test_observation_v1_preserves_overlapping_output_precedence(output):
    codec = from_spec({
        'name': 'observation_codec',
        'version': 1,
        'args': {'state': {output: {'q': 1}}, 'images': {output: ['camera', [2, 2]]}, 'task_field': output},
    })
    assert isinstance(codec, ObservationCodecV1)
    frame = np.full((2, 2, 3), 42, dtype=np.uint8)
    inputs = {'q': [0.5], 'camera': frame, obs_keys.TASK: 'pick'}
    np.testing.assert_array_equal(codec.encode(inputs)[output], [0.5])
    episode = EpisodeContainer({
        'q': DummySignal([100], [[0.5]]),
        'camera': DummySignal([100], [frame]),
        obs_keys.TASK: 'pick',
    })
    training = codec.training_encoder(episode)
    if output == 'task':
        assert training[output] == 'pick'
    else:
        np.testing.assert_array_equal(training[output].values()[0], frame)
    assert codec.training_encoder.meta[LEROBOT_FEATURES][output] == {
        'shape': (2, 2, 3),
        'dtype': 'video',
        'names': ['height', 'width', 'channel'],
    }


@pytest.mark.parametrize('dimensions', [0, 1, 2])
def test_packing_both_directions_preserves_values_and_shapes(dimensions):
    codec = PackFields(
        encode_layout={'state': {'joints': ['q']}, 'language': ['task'], 'video': {}},
        decode_layout={'q': [0, 'predictions', 'q'], 'task': [1, 'text']},
        unsqueeze_dims=dimensions,
        squeeze_dims=dimensions,
    )
    codec = from_spec(json.loads(json.dumps(codec.to_spec())))
    assert isinstance(codec, PackFields)
    values = np.arange(6, dtype=np.float32).reshape(2, 3).T
    encoded = codec.encode({'q': values, 'task': 'pick', 'unused': 42})
    assert set(encoded) == {'state', 'language', 'video'}
    assert encoded['video'] == {}
    assert encoded['state']['joints'].shape == (1,) * dimensions + values.shape
    decoded = codec.decode(({'predictions': {'q': encoded['state']['joints']}}, {'text': encoded['language']}))
    np.testing.assert_array_equal(decoded['q'], values, strict=True)
    assert decoded['task'] == 'pick'
    assert values.shape == (3, 2)


@pytest.mark.parametrize('encode', [False, True])
def test_packing_missing_and_empty_layouts_have_distinct_behavior(encode):
    value = ({'array': np.ones((1, 3))}, {'info': 1})
    identity = PackFields()
    assert (identity.encode(value) if encode else identity.decode(value)) is value
    codec = PackFields(encode_layout={} if encode else None, decode_layout=None if encode else {})
    assert (codec.encode(value) if encode else codec.decode(value)) == {}


@pytest.mark.parametrize('encode', [False, True])
def test_packing_owns_layouts_and_uses_literal_path_keys(encode):
    layout = {'nested': {'value': ['a.b/c', 0]}}
    codec = PackFields(encode_layout=layout if encode else None, decode_layout=None if encode else layout)
    layout['nested']['value'][0] = 'changed'
    description = codec.to_spec()
    description[ARGS]['encode_layout' if encode else 'decode_layout']['nested']['value'][0] = 'changed'
    apply = codec.encode if encode else codec.decode
    assert apply({'a.b/c': [7]}) == {'nested': {'value': 7}}
    with pytest.raises(KeyError, match='a.b/c'):
        apply({'a': {'b/c': [7]}})
    with pytest.raises(IndexError):
        apply({'a.b/c': []})


@pytest.mark.parametrize('value', [np.array(5, dtype=np.int16), 5, None, [1, 2], 'pick'])
def test_packing_root_values_round_trip(value):
    encoded = PackFields(encode_layout={'value': []}, unsqueeze_dims=2).encode(value)
    decoded = PackFields(decode_layout={'value': ['value']}, squeeze_dims=2).decode(encoded)['value']
    if isinstance(value, np.ndarray):
        np.testing.assert_array_equal(decoded, value, strict=True)
    else:
        assert decoded == value


def test_packing_removes_singleton_tuple_layers():
    assert PackFields(decode_layout={'text': []}, squeeze_dims=2).decode((('pick',),)) == {'text': 'pick'}


@pytest.mark.parametrize('value', [np.ones((2, 3)), np.ones((0, 3)), np.array(1), [], [1, 2], 1, 'x'])
def test_packing_rejects_non_singleton_dimensions(value):
    with pytest.raises(ValueError, match='leading size-one dimensions'):
        PackFields(decode_layout={'value': []}, squeeze_dims=1).decode(value)


@pytest.mark.parametrize('direction', ['encode', 'decode'])
@pytest.mark.parametrize('layout', [[], {'value': 'key'}, {'value': [True]}, {'value': [1.5]}, {1: ['key']}])
def test_packing_rejects_invalid_layouts(direction, layout):
    with pytest.raises(ValueError, match='Layout'):
        PackFields(**{f'{direction}_layout': layout})


@pytest.mark.parametrize('parameter', ['unsqueeze_dims', 'squeeze_dims'])
@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_packing_requires_nonnegative_dimension_counts(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        PackFields(**{parameter: value})


def test_packing_preserves_training_columns():
    episode = EpisodeContainer({'q': DummySignal([0], [[1]])}, meta={'label': 'test'})
    codec = PackFields(encode_layout={'state': ['q']}, decode_layout={'action': []})
    assert codec.training_encoder(episode) is episode


@pytest.mark.parametrize(
    'unsqueeze_dims, shape, prompt', [(0, (2, 3), 'pick'), (1, (1, 2, 3), ['pick']), (2, (1, 1, 2, 3), [['pick']])]
)
def test_observation_packing_preserves_values_and_adds_dimensions(unsqueeze_dims, shape, prompt):
    description = component(
        'pack_observation_fields',
        layout={'state': {'value': 'robot.state'}, 'language': {'text': 'task/prompt'}, 'video': {}},
        unsqueeze_dims=unsqueeze_dims,
    )
    codec = from_spec(json.loads(json.dumps(description)))
    assert isinstance(codec, PackFields)
    values = np.arange(6, dtype=np.float32).reshape(3, 2).T
    inputs = {'robot.state': values, 'task/prompt': 'pick', 'unused': 10}

    encoded = codec.encode(inputs)

    assert set(encoded) == {'state', 'language', 'video'}
    assert encoded['state']['value'].shape == shape
    assert encoded['state']['value'].dtype == np.float32
    np.testing.assert_array_equal(encoded['state']['value'].reshape(2, 3), values)
    assert encoded['language'] == {'text': prompt}
    assert encoded['video'] == {}
    assert inputs['robot.state'] is values
    assert values.shape == (2, 3)
    assert inputs['task/prompt'] == 'pick'
    assert (
        codec.to_spec()
        == PackFields(
            encode_layout={'state': {'value': ['robot.state']}, 'language': {'text': ['task/prompt']}, 'video': {}},
            unsqueeze_dims=unsqueeze_dims,
        ).to_spec()
    )


def test_observation_packing_handles_images_and_plain_values_without_coercion():
    codec = cast(
        PackFields,
        from_spec(
            component(
                'pack_observation_fields',
                version=1,
                layout={'image': 'rgb', 'scalar': 'zero', 'sequence': 'items', 'empty': 'none'},
                unsqueeze_dims=2,
            )
        ),
    )
    pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    encoded = codec.encode({'rgb': pixels, 'zero': 0.0, 'items': [1, 2], 'none': None})

    assert encoded['image'].shape == (1, 1, 2, 3, 3)
    assert encoded['image'].dtype == np.uint8
    np.testing.assert_array_equal(encoded['image'][0, 0], pixels)
    assert encoded['scalar'] == [[0.0]]
    assert encoded['sequence'] == [[[1, 2]]]
    assert encoded['empty'] == [[None]]


def test_observation_packing_keeps_its_layout_and_requires_selected_fields():
    layout = {'state': {'value': 'input'}}
    codec = cast(PackFields, from_spec(component('pack_observation_fields', version=1, layout=layout)))
    layout['state']['value'] = 'changed'
    description = codec.to_spec()
    description[ARGS]['encode_layout']['state']['value'][0] = 'changed'

    assert codec.encode({'input': 1}) == {'state': {'value': 1}}
    with pytest.raises(KeyError, match='input'):
        codec.encode({'unselected': 1})


@pytest.mark.parametrize('unsqueeze_dims', [-1, 1.5, True])
def test_observation_packing_rejects_invalid_dimensions(unsqueeze_dims):
    with pytest.raises(ValueError, match='unsqueeze_dims'):
        cast(
            PackFields,
            from_spec(component('pack_observation_fields', version=1, layout={}, unsqueeze_dims=unsqueeze_dims)),
        )


@pytest.mark.parametrize('layout', [{'state': {'value': 1}}, {'value': ['input']}, {1: 'input'}])
def test_observation_packing_rejects_invalid_layout_entries(layout):
    with pytest.raises(ValueError, match='Layout entries'):
        from_spec({'name': 'pack_observation_fields', 'version': 1, ARGS: {'layout': layout}})


def test_observation_packing_preserves_training_and_native_results():
    codec = cast(
        PackFields,
        from_spec(
            component('pack_observation_fields', version=1, layout={'state': {'value': 'input'}}, unsqueeze_dims=2)
        ),
    )
    episode = EpisodeContainer({'input': DummySignal([0], [1])}, meta={'label': 'test'})
    result = ({'action': np.ones((1, 3, 2))}, {'timing': 0.1})

    assert codec.training_encoder(episode) is episode
    assert codec.decode(result) is result
    assert codec.meta == {}


def test_observation_renaming_uses_literal_keys_and_preserves_values():
    codec = RenameObservationFields({'observation.state': 'observation/state', 'task': 'prompt'})
    state = np.array([1, 2], dtype=np.float32)
    camera = {'image': np.zeros((2, 3, 3), dtype=np.uint8)}
    inputs = {'observation.state': state, 'camera': camera}

    encoded = codec.encode(inputs)

    assert set(encoded) == {'observation/state', 'camera'}
    assert encoded['observation/state'] is state
    assert encoded['camera'] is camera
    assert set(inputs) == {'observation.state', 'camera'}


def test_observation_renaming_can_swap_names():
    codec = RenameObservationFields({'left': 'right', 'right': 'left'})
    assert codec.encode({'left': 1, 'right': 2}) == {'left': 2, 'right': 1}


@pytest.mark.parametrize('mapping', [{'a': 'b'}, {'a': 'result', 'b': 'result'}])
def test_observation_renaming_rejects_overwriting_fields(mapping):
    codec = RenameObservationFields(mapping)
    with pytest.raises(ValueError, match='collide'):
        codec.encode({'a': 1, 'b': 2})


def test_observation_renaming_preserves_training_and_native_results():
    codec = RenameObservationFields({'state': 'input'})
    episode = EpisodeContainer({'state': DummySignal([100], [[1]])}, meta={'label': 'test'})
    result = ({'state': np.ones((1, 3, 2))}, {'timing': 0.1})

    assert codec.training_encoder(episode) is episode
    assert codec.decode(result) is result


def test_observation_encode_images_and_state_shapes():
    # Image matches target size; ensures no resampling artifacts in assertions
    h, w = 6, 8
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    enc = StateObservationCodec(state={'observation.state': {'a': 2, 'b': 1}}) & image_observation(
        images={'observation.images.left': ('left.image', (w, h))}
    )
    obs = enc.encode({'left.image': img, 'a': [1, 2], 'b': 3.0})

    assert 'observation.images.left' in obs and 'observation.state' in obs
    left = obs['observation.images.left']
    state = obs['observation.state']

    assert left.shape == (h, w, 3)
    assert left.dtype == np.uint8
    assert np.all(left == 255)

    assert state.shape == (3,)
    np.testing.assert_allclose(state, np.array([1, 2, 3], dtype=np.float32))


def test_observation_encode_missing_state_inputs_raise():
    enc = StateObservationCodec(state={'observation.state': {'missing': 1}})
    with pytest.raises(KeyError):
        enc.encode({})


@pytest.mark.parametrize('axis', [HARNESS_WORLD, RECEIVED_WORLD, RECORDED_TIME])
def test_training_observations_align_on_the_recording_clock(axis):
    episode = EpisodeContainer({
        'a': DummySignal([[0, 100], [10, 200]], [[1], [2]], timelines=(axis, RECEIVED_WALL)),
        'b': DummySignal([[0, 150], [10, 250]], [[3], [4]], timelines=(axis, RECEIVED_WALL)),
    })
    codec = StateObservationCodec(state={'state': {'a': 1, 'b': 1}})
    state = codec.training_encoder(episode)['state']
    assert state.timelines == (axis,)
    assert list(state.timestamps(axis)) == [0, 10]
    np.testing.assert_array_equal(state.values(), [[1, 3], [2, 4]])


@pytest.mark.parametrize('timeline', [HARNESS_WORLD, RECEIVED_WORLD])
def test_training_observations_require_the_selected_clock_for_each_input(timeline):
    episode = EpisodeContainer({
        'a': DummySignal([[0, 100], [10, 200]], [[1], [2]], timelines=(timeline, RECORDED_TIME)),
        'b': DummySignal([100, 200], [[3], [4]], timelines=(RECORDED_TIME,)),
    })
    codec = StateObservationCodec(state={'state': {'a': 1, 'b': 1}})
    with pytest.raises(ValueError, match='Failed to apply transform') as error:
        codec.training_encoder(episode)['state']
    assert isinstance(error.value.__cause__, KeyError)
    assert error.value.__cause__.args == (timeline,)


@pytest.mark.parametrize(
    ('preferred', 'other'),
    [(HARNESS_WORLD, RECEIVED_WORLD), (HARNESS_WORLD, RECORDED_TIME), (RECEIVED_WORLD, RECORDED_TIME)],
)
def test_training_observations_use_the_preferred_clock(preferred, other):
    episode = EpisodeContainer({
        'a': DummySignal([[0, 100], [10, 200]], [[1], [2]], timelines=(preferred, other)),
        'b': DummySignal([[0, 150], [10, 250]], [[3], [4]], timelines=(preferred, other)),
    })
    codec = StateObservationCodec(state={'state': {'a': 1, 'b': 1}})
    state = codec.training_encoder(episode)['state']
    assert state.timelines == (preferred,)
    assert list(state.timestamps(preferred)) == [0, 10]
    np.testing.assert_array_equal(state.values(), [[1, 3], [2, 4]])


def test_training_observations_do_not_select_arbitrary_clocks():
    episode = EpisodeContainer({'a': DummySignal([0, 10], [[1], [2]], timelines=('device.clock',))})
    codec = StateObservationCodec(state={'state': {'a': 1}})
    with pytest.raises(ValueError, match='Failed to apply transform') as error:
        codec.training_encoder(episode)['state']
    assert isinstance(error.value.__cause__, KeyError)
    assert error.value.__cause__.args == (RECORDED_TIME,)


def test_observation_encode_task():
    enc = StateObservationCodec(state={'observation.state': {'a': 1}})
    obs = enc.encode({'a': 1.0, obs_keys.TASK: 'test_task'})
    assert obs[obs_keys.TASK] == 'test_task'

    obs_no_task = enc.encode({'a': 1.0})
    assert obs_keys.TASK not in obs_no_task


def test_bimanual_joints_state_packs_the_left_arm_first_with_gripper_widths():
    state = bimanual_joints_state({
        obs_keys.arm_channel(obs_keys.ROBOT_STATE, obs_keys.LEFT_ARM) + obs_keys.JOINTS_SUFFIX: np.arange(
            6, dtype=np.float64
        ),
        obs_keys.arm_channel(obs_keys.ROBOT_STATE, obs_keys.RIGHT_ARM) + obs_keys.JOINTS_SUFFIX: np.arange(
            6, 12, dtype=np.float64
        ),
        obs_keys.arm_channel(obs_keys.GRIP, obs_keys.LEFT_ARM): 0.0,
        obs_keys.arm_channel(obs_keys.GRIP, obs_keys.RIGHT_ARM): 0.75,
    })

    assert state.dtype == np.float32
    np.testing.assert_array_equal(state, np.concatenate([np.arange(6), [1.0], np.arange(6, 12), [0.25]]))
    # The action codec reads the same layout, so a state fed back as an action commands the observed pose.
    decoded = BimanualJointsAction().decode({ACTION: state})
    np.testing.assert_array_equal(
        decoded[obs_keys.arm_channel(obs_keys.ROBOT_COMMAND, obs_keys.RIGHT_ARM)].positions, np.arange(6, 12)
    )
    assert decoded[obs_keys.arm_channel(obs_keys.TARGET_GRIP, obs_keys.RIGHT_ARM)] == pytest.approx(0.75)
