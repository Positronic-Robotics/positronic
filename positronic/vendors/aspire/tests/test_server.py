import hashlib

import pytest

from positronic import keys
from positronic.drivers.roboarm import keys as roboarm_keys
from positronic.drivers.roboarm.models import GRIPPER_SITE, EEFrame, parse_ee_frames
from positronic.policy import keys as policy_keys
from positronic.policy.base import PolicyEnd
from positronic.policy.keys import END
from positronic.policy.spec import from_spec
from positronic.vendors.aspire.bridge import BridgeSettings
from positronic.vendors.aspire.server import (
    META_PROGRAM_SHA256,
    POLICY_TYPE,
    CapProgramModel,
    SessionRecord,
    deployment,
)
from positronic.vendors.aspire.tests.test_bridge import Rig

PROGRAM = """
result = close_gripper(arm='left')
if not result.ok:
    raise ValueError(result.error.message)
"""


@pytest.fixture
def make_model(tmp_path):
    made = []

    def make(program: str) -> CapProgramModel:
        path = tmp_path / 'program.py'
        path.write_text(program)
        settings = BridgeSettings(fps=10.0, round_s=0.2, call_timeout_s=2.0, think_timeout_s=30.0)
        made.append(CapProgramModel(path, settings, tmp_path))
        return made[-1]

    yield make
    for model in made:
        model.close()


def run_session(model: CapProgramModel, session_id: str, rig: Rig) -> list[dict]:
    for _ in range(100):
        chunk = model(rig.obs(), session_id=session_id)
        if END in chunk[-1]:
            return chunk
        rig.play(chunk)
    raise AssertionError('the program did not end')


def record(tmp_path, session_id: str) -> SessionRecord:
    return SessionRecord.model_validate_json((tmp_path / f'{session_id}.json').read_text())


def test_each_session_runs_the_program_and_records_how_it_ended(make_model, tmp_path):
    model = make_model(PROGRAM)
    stalled = Rig()
    stalled.grip_limit['left'] = 0.5
    assert run_session(model, 'free', Rig())[-1][END] == PolicyEnd.COMPLETE.value
    assert run_session(model, 'stalled', stalled)[-1][END] == PolicyEnd.GIVE_UP.value

    sha = hashlib.sha256(PROGRAM.encode()).hexdigest()
    expected = SessionRecord(session_id='free', program_sha256=sha, end=PolicyEnd.COMPLETE, error=None)
    assert record(tmp_path, 'free') == expected
    failed = record(tmp_path, 'stalled')
    assert failed.end is PolicyEnd.GIVE_UP
    assert 'ValueError' in failed.error


def test_a_session_that_ends_before_its_program_records_no_end(make_model, tmp_path):
    model = make_model("move_to_joints([1, 1, 1, 1, 1, 1], arm='left')")
    model(Rig().obs(), session_id='dropped')
    model.end_session('dropped')
    assert record(tmp_path, 'dropped').end is None


def test_meta_states_the_hand_frames_it_reads_and_commands(make_model):
    meta = make_model(PROGRAM).meta()
    assert meta[policy_keys.TYPE] == POLICY_TYPE
    assert meta[META_PROGRAM_SHA256] == hashlib.sha256(PROGRAM.encode()).hexdigest()
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
