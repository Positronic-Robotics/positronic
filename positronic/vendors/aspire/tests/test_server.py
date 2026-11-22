import hashlib
import json

import pytest
from positronic_model_server.serialization import deserialise, serialise

from positronic import keys
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame, parse_ee_frames
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd
from positronic.policy.keys import END
from positronic.policy.spec import from_spec
from positronic.vendors.aspire.bridge import BridgeSettings
from positronic.vendors.aspire.server import (
    META_CALIBRATION,
    META_PROGRAM,
    META_PROGRAM_SHA256,
    META_SESSION_RECORD,
    META_SETTINGS,
    PARAM_PROGRAM,
    POLICY_TYPE,
    CapProgramModel,
    SessionRecord,
    deployment,
)
from positronic.vendors.aspire.tests.test_bridge import LOOKS_DOWN, WRIST, Rig, wrist_rig

PROGRAM = """
result = close_gripper(arm='left')
if not result.ok:
    raise ValueError(result.error.message)
"""


@pytest.fixture
def programs(tmp_path):
    directory = tmp_path / 'programs'
    directory.mkdir()
    (directory / 'program.py').write_text(PROGRAM)
    return directory


@pytest.fixture
def model(programs, tmp_path):
    settings = BridgeSettings(fps=10.0, round_s=0.2, call_timeout_s=2.0, think_timeout_s=30.0)
    model = CapProgramModel(programs, settings, tmp_path)
    yield model
    model.close()


def open_session(model: CapProgramModel, session_id: str, program: str = 'program.py') -> dict:
    return model.open_session(session_id, {**model.parameters(), PARAM_PROGRAM: program})


def run_session(model: CapProgramModel, session_id: str, rig: Rig, program: str = 'program.py') -> list[dict]:
    open_session(model, session_id, program)
    for _ in range(100):
        chunk = model(rig.obs(), session_id=session_id)
        if END in chunk[-1]:
            return chunk
        rig.play(chunk)
    raise AssertionError('the program did not end')


def record(tmp_path, session_id: str) -> SessionRecord:
    return SessionRecord.model_validate_json((tmp_path / f'{session_id}.json').read_text())


def test_each_session_runs_the_program_and_records_how_it_ended(model, tmp_path):
    stalled = Rig()
    stalled.grip_limit['left'] = 0.5
    assert run_session(model, 'free', Rig())[-1][END] == PolicyEnd.COMPLETE.value
    assert run_session(model, 'stalled', stalled)[-1][END] == PolicyEnd.GIVE_UP.value

    free = record(tmp_path, 'free')
    assert free.program == 'program.py' and free.program_source == PROGRAM
    assert free.program_sha256 == hashlib.sha256(PROGRAM.encode()).hexdigest()
    assert free.end is PolicyEnd.COMPLETE and free.error is None
    assert [call.name for call in free.calls] == ['close_gripper']
    failed = record(tmp_path, 'stalled')
    assert failed.end is PolicyEnd.GIVE_UP
    assert 'ValueError' in failed.error


def test_a_session_that_ends_before_its_program_records_no_end(model, programs, tmp_path):
    (programs / 'wait.py').write_text("move_to_joints([1, 1, 1, 1, 1, 1], arm='left')")
    open_session(model, 'dropped', 'wait.py')
    model(Rig().obs(), session_id='dropped')
    model.end_session('dropped')
    assert record(tmp_path, 'dropped').end is None


def test_meta_states_the_hand_frames_it_reads_and_commands(model):
    meta = model.meta()
    assert meta[policy_keys.TYPE] == POLICY_TYPE
    frames = parse_ee_frames(meta[roboarm_keys.EE_FRAMES])
    assert frames == {
        signal: EEFrame(site=GRIPPER_SITE)
        for arm in keys.BIMANUAL_ARMS
        for signal in (
            f'{keys.arm_channel(keys.ROBOT_STATE, arm)}{keys.EE_POSE_SUFFIX}',
            f'{keys.arm_channel(keys.ROBOT_COMMAND, arm)}{keys.POSE_SUFFIX}',
        )
    }


def test_the_client_stack_plays_at_the_bridge_rate():
    stack = deployment(BridgeSettings(fps=25.0)).local
    assert from_spec(stack.to_spec()).meta()[policy_keys.ACTION_FPS] == 25.0


def test_each_session_runs_the_program_it_names(model, programs, tmp_path):
    (programs / 'nested').mkdir()
    (programs / 'nested' / 'open.py').write_text("open_gripper(arm='right')")
    meta = open_session(model, 'named', 'nested/open.py')
    assert meta[META_PROGRAM] == 'nested/open.py'
    assert meta[META_PROGRAM_SHA256] == hashlib.sha256(b"open_gripper(arm='right')").hexdigest()
    assert meta[META_SESSION_RECORD] == 'named.json'
    model.end_session('named')
    run_session(model, 'other', Rig())
    assert [call.name for call in record(tmp_path, 'other').calls] == ['close_gripper']


def test_a_session_runs_the_program_as_it_reads_when_the_session_opens(model, programs, tmp_path):
    run_session(model, 'first', Rig())
    (programs / 'program.py').write_text("open_gripper(arm='right')")
    run_session(model, 'second', Rig())
    assert [call.name for call in record(tmp_path, 'second').calls] == ['open_gripper']


@pytest.mark.parametrize('program', [None, '', 'missing.py', '../program.py', '/etc/hostname', '.'])
def test_a_session_that_names_no_program_in_the_directory_is_refused(model, programs, program):
    (programs.parent / 'program.py').write_text(PROGRAM)
    with pytest.raises(ValueError, match='program'):
        open_session(model, 'refused', program)


def test_a_program_that_does_not_compile_gives_up_with_its_error(model, programs, tmp_path):
    (programs / 'broken.py').write_text('close_gripper(')
    assert run_session(model, 'broken', Rig(), 'broken.py')[-1][END] == PolicyEnd.GIVE_UP.value
    broken = record(tmp_path, 'broken')
    assert 'SyntaxError' in broken.error and broken.calls == []


def test_a_one_arm_session_answers_on_the_unnamed_channels_and_its_meta_states_their_hand_frames(programs, tmp_path):
    (programs / 'one_arm.py').write_text('assert close_gripper().ok and go_home().ok\n')
    settings = BridgeSettings(fps=10.0, round_s=0.2, call_timeout_s=2.0, think_timeout_s=30.0, arms=keys.ONE_ARM)
    model = CapProgramModel(programs, settings, tmp_path)
    try:
        meta = model.meta()
        frames = parse_ee_frames(meta[roboarm_keys.EE_FRAMES])
        assert frames == dict.fromkeys((keys.EE_POSE, keys.TARGET_EE_POSE), EEFrame(site=GRIPPER_SITE))
        assert meta[f'{META_SETTINGS}.arms'] == keys.ONE_ARM
        chunk = run_session(model, 'one-arm', Rig(keys.ONE_ARM, joints=7), 'one_arm.py')
    finally:
        model.close()
    assert chunk[-1][END] == PolicyEnd.COMPLETE.value
    assert set(chunk[-1]) == {keys.ROBOT_COMMAND, keys.TARGET_GRIP, END}


def test_a_session_gets_the_views_of_the_calibration_file_and_the_meta_states_it(programs, tmp_path):
    calibration = tmp_path / 'calibration.json'
    calibration.write_text(json.dumps({WRIST: LOOKS_DOWN.model_dump()}))
    (programs / 'look.py').write_text(
        "view = get_observation().cameras['wrist']\n"
        'assert view.intrinsics[0][0] == 10.0 and view.depth_m[1][1] == 0.5\n'
        'assert abs(view.camera_pose.position[2] - 0.6) < 1e-9\n'
    )
    settings = BridgeSettings(fps=10.0, round_s=0.2, call_timeout_s=2.0, think_timeout_s=30.0, arms=keys.ONE_ARM)
    model = CapProgramModel(programs, settings, tmp_path, calibration=calibration)
    try:
        meta = model.meta()
        chunk = run_session(model, 'look', wrist_rig(), 'look.py')
    finally:
        model.close()
    assert chunk[-1][END] == PolicyEnd.COMPLETE.value, record(tmp_path, 'look').error
    assert meta[f'{META_CALIBRATION}.{WRIST}.pose'] == LOOKS_DOWN.pose
    assert meta[f'{META_CALIBRATION}.{WRIST}.intrinsics'] == LOOKS_DOWN.intrinsics
    assert meta[f'{META_CALIBRATION}.{WRIST}.hand'] == LOOKS_DOWN.hand
    # The handshake carries the meta.
    assert deserialise(serialise(meta)) == json.loads(json.dumps(meta))


@pytest.mark.parametrize(
    'calibration',
    [
        {WRIST: LOOKS_DOWN.model_dump() | {'frame': 'world'}},
        {WRIST: {'pose': LOOKS_DOWN.pose}},
        {WRIST: LOOKS_DOWN.model_dump() | {'pose': [0.0, 0.0, 0.3]}},
        [LOOKS_DOWN.model_dump()],
    ],
)
def test_a_calibration_file_that_does_not_map_each_camera_to_its_calibration_is_refused(
    programs, tmp_path, calibration
):
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(calibration))
    with pytest.raises(ValueError):
        CapProgramModel(programs, BridgeSettings(), tmp_path, calibration=path)


def test_a_calibration_file_on_the_hand_of_no_arm_is_refused_when_the_server_starts(programs, tmp_path):
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps({WRIST: LOOKS_DOWN.model_dump() | {'hand': 'tertiary'}}))
    with pytest.raises(ValueError, match='hand'):
        CapProgramModel(programs, BridgeSettings(), tmp_path, calibration=path)
