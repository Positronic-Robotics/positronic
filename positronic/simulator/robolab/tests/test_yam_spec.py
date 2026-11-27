"""The two-arm YAM robot of the RoboLab env server, checked without Isaac Lab: the URDF it builds, its kinematics
against the i2rt model, its labels, and the action that the wire commands map to."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from positronic import keys
from positronic.cfg import embodiment
from positronic.simulator.env_server import protocol
from positronic.simulator.robolab import keys as robolab_keys
from positronic.simulator.robolab import yam_spec as spec


@pytest.fixture(scope='module')
def urdf() -> ET.Element:
    return ET.fromstring(spec.bimanual_urdf())


@pytest.fixture(scope='module')
def model() -> mujoco.MjModel:
    compiler = '<mujoco><compiler discardvisual="false" fusestatic="false" strippath="false"/></mujoco>'
    text = spec.bimanual_urdf().replace('<robot name="yam_bimanual">', '<robot name="yam_bimanual">' + compiler, 1)
    return mujoco.MjModel.from_xml_string(text)


def _set_joints(model: mujoco.MjModel, data: mujoco.MjData, side: str, values: dict[str, float]) -> None:
    for name, value in values.items():
        data.qpos[model.jnt_qposadr[model.joint(f'{side}_{name}').id]] = value
    mujoco.mj_kinematics(model, data)


def test_the_arms_are_the_bimanual_arms_of_positronic():
    assert spec.SIDES == keys.BIMANUAL_ARMS


def test_the_urdf_is_one_tree_on_the_base_plate(urdf):
    links = {link.attrib['name'] for link in urdf.findall('link')}
    children = {joint.find('child').attrib['link'] for joint in urdf.findall('joint')}
    parents = {joint.find('parent').attrib['link'] for joint in urdf.findall('joint')}
    assert links - children == {'base_plate'}
    assert parents | children <= links
    for side in spec.SIDES:
        mount = urdf.find(f"joint[@name='{side}_mount']")
        assert mount.attrib['type'] == 'fixed'
        assert [float(v) for v in mount.find('origin').attrib['xyz'].split()] == pytest.approx(spec.mount_xyz(side))


def test_each_joint_has_the_spec_limits_and_joint8_mimics_joint7(urdf):
    for side in spec.SIDES:
        for i, name in enumerate(spec.ARM_JOINTS):
            limit = urdf.find(f"joint[@name='{side}_{name}']/limit")
            assert (float(limit.attrib['lower']), float(limit.attrib['upper'])) == pytest.approx(spec.ARM_LIMITS[i])
            assert float(limit.attrib['effort']) == spec.ARM_EFFORT[i]
        for name in (spec.FINGER_JOINT, spec.FINGER_MIMIC_JOINT):
            limit = urdf.find(f"joint[@name='{side}_{name}']/limit")
            assert (float(limit.attrib['lower']), float(limit.attrib['upper'])) == (
                spec.FINGER_CLOSED,
                spec.FINGER_OPEN,
            )
        mimic = urdf.find(f"joint[@name='{side}_{spec.FINGER_MIMIC_JOINT}']/mimic")
        assert mimic.attrib == {'joint': f'{side}_{spec.FINGER_JOINT}', 'multiplier': '1', 'offset': '0'}


def test_every_body_link_collides_and_the_gripper_link_does_not(urdf):
    for link in urdf.findall('link'):
        name = link.attrib['name']
        collisions = link.findall('collision')
        if name.endswith(f'_{spec.GRIPPER_SITE}'):
            assert not collisions
        elif name.endswith(spec.FINGER_LINKS):
            assert [c.find('geometry')[0].tag for c in collisions] == ['box', 'box']
        else:
            assert len(collisions) == 1, name
    names = [c.attrib['name'] for c in urdf.iter('collision')]
    assert len(names) == len(set(names))


def test_the_meshes_are_those_of_the_i2rt_model(urdf):
    meshes = {mesh.attrib['filename'] for mesh in urdf.iter('mesh')}
    assert meshes
    assert all((spec.I2RT_URDF.parent / name).is_file() for name in meshes)


def test_the_gripper_link_is_the_gripper_site_of_the_single_arm_model(model):
    data = mujoco.MjData(model)
    single = mujoco.MjModel.from_xml_path(str(spec.I2RT_URDF.with_name('yam.xml')))
    single_data = mujoco.MjData(single)
    site = single.site(spec.GRIPPER_SITE).id
    rng = np.random.default_rng(19)
    for _ in range(5):
        q = [rng.uniform(lo + 0.1, hi - 0.1) for lo, hi in spec.ARM_LIMITS]
        single_data.qpos[:6] = q
        mujoco.mj_kinematics(single, single_data)
        for side in spec.SIDES:
            data.qpos[:] = 0
            _set_joints(model, data, side, dict(zip(spec.ARM_JOINTS, q, strict=True)))
            body = model.body(f'{side}_{spec.GRIPPER_SITE}').id
            assert data.xpos[body] - spec.mount_xyz(side) == pytest.approx(single_data.site_xpos[site], abs=1e-7)
            assert data.xmat[body] == pytest.approx(single_data.site_xmat[site], abs=1e-7)


@pytest.mark.parametrize('side', spec.SIDES)
def test_the_pads_meet_on_the_wrist_axis_when_closed_and_open_95_mm(model, side):
    data = mujoco.MjData(model)
    wrist = model.body(f'{side}_{spec.WRIST_LINK}').id
    for travel, gap in ((spec.FINGER_CLOSED, 0.0), (spec.FINGER_OPEN, 0.095)):
        _set_joints(model, data, side, {spec.FINGER_JOINT: travel, spec.FINGER_MIMIC_JOINT: travel})
        rot, pos = data.xmat[wrist].reshape(3, 3), data.xpos[wrist]
        faces = []
        for finger, inner in (('tip_left', 1), ('tip_right', -1)):
            geom = model.geom(f'{side}_{finger}_pad').id
            center = (data.geom_xpos[geom] - pos) @ rot
            half = np.abs(rot.T @ data.geom_xmat[geom].reshape(3, 3)) @ model.geom_size[geom]
            faces.append(center[1] + inner * half[1])
        assert faces[1] - faces[0] == pytest.approx(gap, abs=1e-6)
        assert faces[0] + faces[1] == pytest.approx(0.0, abs=1e-6)


def test_the_robolab_labels_name_links_and_joints_of_the_urdf(urdf):
    links = {link.attrib['name'] for link in urdf.findall('link')}
    joints = {joint.attrib['name'] for joint in urdf.findall('joint')}
    fingers = [spec.finger_label(side, finger) for side in spec.SIDES for finger in spec.FINGER_LINKS]
    for label in fingers:
        assert spec.CONTACT_GRIPPER[label].removeprefix('{ENV_REGEX_NS}/robot/') in links
    assert set(spec.EE_RECORDER_BODIES.values()) <= links
    assert {entry['joint'] for entry in spec.GRIPPER_CLOSURE.values()} <= joints
    assert {joint for _, names in spec.ACTION_TERMS for joint in names} <= joints
    assert {camera.parent for camera in spec.CAMERAS.values()} <= links


def test_the_contact_groups_cover_every_finger_of_their_hands():
    """The groups are flat, as RoboLab takes them: each names fingers only, and the gripper group names all four."""
    fingers = {label for label, target in spec.CONTACT_GRIPPER.items() if isinstance(target, str)}
    assert len(fingers) == len(spec.SIDES) * len(spec.FINGER_LINKS)
    for side in spec.SIDES:
        assert spec.CONTACT_GRIPPER[side] == [spec.finger_label(side, finger) for finger in spec.FINGER_LINKS]
    assert set(spec.CONTACT_GRIPPER[spec.CONTACT_GRIPPERS]) == fingers
    assert len(spec.CONTACT_GRIPPER[spec.CONTACT_GRIPPERS]) == len(fingers)


def test_the_stand_is_under_the_plate_and_reaches_the_table_top():
    (x0, x1), (y0, y1), (_z0, z1) = spec.STAND_X, spec.STAND_Y, spec.STAND_Z
    # The stand front is the plate front, so the bound takes float round-off.
    assert x0 <= spec.PLATE_CENTER_X - spec.PLATE_SIZE[0] / 2
    assert spec.PLATE_CENTER_X + spec.PLATE_SIZE[0] / 2 <= x1 + 1e-9
    assert y0 <= -spec.PLATE_SIZE[1] / 2 and spec.PLATE_SIZE[1] / 2 <= y1
    assert z1 == 0.0
    assert 'PhysicsCollisionAPI' in spec.stand_usda()


def test_the_arm_mounts_in_the_rig_world_are_those_of_the_rig():
    mounts = embodiment.yam_bimanual_yambox.kwargs['mounts']
    for side in spec.SIDES:
        assert np.add(spec.mount_xyz(side), spec.ROOT_IN_RIG_WORLD) == pytest.approx(mounts[side])


def _joint_command(q) -> dict:
    return {protocol.COMMAND_TYPE: protocol.JOINT_POS, protocol.COMMAND_JOINT_POS: q}


def _wire(left: dict, right: dict, grips=(0.0, 0.0)) -> dict:
    wire = {}
    for side, command, grip in zip(spec.SIDES, (left, right), grips, strict=True):
        wire[protocol.arm_channel(protocol.ROBOT_COMMAND, side)] = command
        wire[protocol.arm_channel(protocol.TARGET_GRIP, side)] = grip
    return wire


_MEASURED = np.arange(12, dtype=np.float32).reshape(len(spec.SIDES), 6) / 10
_HOLD = {protocol.COMMAND_TYPE: protocol.HOLD}


def test_the_action_holds_each_arm_then_its_grip_in_the_term_order():
    left, right = np.full(6, 0.5, dtype=np.float32), np.full(6, -0.5, dtype=np.float32)

    action, grips = spec.action(_wire(_joint_command(left), _joint_command(right), (1.0, 0.0)), _MEASURED, np.zeros(2))

    assert spec.ACTION_DIM == action.shape[0] == 14
    assert [name for name, _ in spec.ACTION_TERMS] == ['left_arm', 'left_gripper', 'right_arm', 'right_gripper']
    np.testing.assert_array_equal(action, np.concatenate([left, [1.0], right, [0.0]]))
    np.testing.assert_array_equal(grips, [1.0, 0.0])


def test_a_joint_delta_moves_from_the_measured_joints_and_a_hold_keeps_them():
    delta = {protocol.COMMAND_TYPE: protocol.JOINT_DELTA, protocol.COMMAND_JOINT_DELTA: np.full(6, 0.25)}

    action = spec.action(_wire(delta, _HOLD), _MEASURED, np.zeros(2)).values

    np.testing.assert_allclose(action[:6], _MEASURED[0] + 0.25)
    np.testing.assert_array_equal(action[7:13], _MEASURED[1])


def _joint_delta(dq) -> dict:
    return {protocol.COMMAND_TYPE: protocol.JOINT_DELTA, protocol.COMMAND_JOINT_DELTA: dq}


@pytest.mark.parametrize(
    'command',
    [_joint_command(np.full(6, np.nan)), _joint_command(np.zeros(7)), _joint_delta(np.zeros(7)), _joint_delta([0.1])],
    ids=['not finite', 'wrong width', 'delta too wide', 'delta too narrow'],
)
def test_an_arm_holds_its_measured_joints_on_a_command_the_rig_refuses_and_a_warning_names_it(command, caplog):
    action = spec.action(_wire(command, _HOLD), _MEASURED, np.zeros(2)).values

    np.testing.assert_array_equal(action[:6], _MEASURED[0])
    (record,) = caplog.records
    assert f'the {spec.LEFT} arm holds' in record.getMessage()


def test_an_arm_keeps_its_last_grip_when_the_wire_grip_is_not_finite():
    action, grips = spec.action(_wire(_HOLD, _HOLD, (float('nan'), 1.0)), _MEASURED, np.array([1.0, 0.0]))

    assert (action[6], action[13]) == (1.0, 1.0)
    np.testing.assert_array_equal(grips, [1.0, 1.0])


# Joints well inside the limits of each arm, and the joints of a pose command near them.
_START = np.array([0.2, 1.2, 1.0, -0.3, 0.2, 0.3])
_TARGET = _START + np.array([0.15, 0.1, -0.1, 0.2, -0.15, 0.1])


def _gripper_pose(model: mujoco.MjModel, side: str, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pose of the ``<side>_gripper`` link of the URDF at the joints ``q`` of the arm, in the rig world frame."""
    data = mujoco.MjData(model)
    _set_joints(model, data, side, dict(zip(spec.ARM_JOINTS, q, strict=True)))
    body = model.body(f'{side}_{spec.GRIPPER_SITE}').id
    return data.xpos[body] + spec.ROOT_IN_RIG_WORLD, data.xmat[body].reshape(3, 3)


def _pose_command(pos: np.ndarray, rot: np.ndarray, kind: str = protocol.CARTESIAN) -> dict:
    field = protocol.COMMAND_POSE if kind == protocol.CARTESIAN else protocol.COMMAND_DELTA
    return {protocol.COMMAND_TYPE: kind, field: np.concatenate([pos, np.asarray(rot).reshape(-1)])}


@pytest.mark.parametrize('side', spec.SIDES)
def test_the_chain_of_each_arm_gives_the_pose_and_the_jacobian_of_its_gripper_link(model, side):
    chain = spec.arm_chains()[side]
    rng = np.random.default_rng(7)
    assert chain.joints == tuple(f'{side}_{joint}' for joint in spec.ARM_JOINTS)
    np.testing.assert_allclose(np.stack([chain.lower, chain.upper], axis=1), spec.ARM_LIMITS)
    for _ in range(5):
        q = np.array([rng.uniform(lo + 0.1, hi - 0.1) for lo, hi in spec.ARM_LIMITS])
        pos, rot, jac = chain.kinematics(q)
        expected_pos, expected_rot = _gripper_pose(model, side, q)
        np.testing.assert_allclose(pos + spec.ROOT_IN_RIG_WORLD, expected_pos, atol=1e-9)
        np.testing.assert_allclose(rot, expected_rot, atol=1e-9)
        for joint in range(len(q)):
            moved = q.copy()
            moved[joint] += 1e-6
            moved_pos, _, _ = chain.kinematics(moved)
            np.testing.assert_allclose((moved_pos - pos) / 1e-6, jac[:3, joint], atol=1e-5)


@pytest.mark.parametrize('side', spec.SIDES)
def test_a_pose_command_solves_to_joints_that_reach_the_pose_and_the_other_arm_moves(model, side):
    target_pos, target_rot = _gripper_pose(model, side, _TARGET)
    measured = np.stack([_START, _START]).astype(np.float32)
    joints = np.full(6, 0.4, dtype=np.float32)
    commands = {side: _pose_command(target_pos, target_rot)}
    other = next(arm for arm in spec.SIDES if arm != side)
    commands[other] = _joint_command(joints)

    values = spec.action(_wire(commands[spec.LEFT], commands[spec.RIGHT]), measured, np.zeros(2)).values

    arm = spec.SIDES.index(side)
    reached_pos, reached_rot = _gripper_pose(model, side, values[arm * 7 : arm * 7 + 6])
    np.testing.assert_allclose(reached_pos, target_pos, atol=1e-3)
    np.testing.assert_allclose(reached_rot, target_rot, atol=1e-2)
    other_arm = spec.SIDES.index(other)
    np.testing.assert_array_equal(values[other_arm * 7 : other_arm * 7 + 6], joints)


def test_a_cartesian_delta_moves_the_gripper_link_from_its_measured_pose(model):
    measured = np.stack([_START, _START]).astype(np.float32)
    start_pos, start_rot = _gripper_pose(model, spec.LEFT, _START)
    shift = np.array([0.01, -0.02, 0.015])
    delta = _pose_command(shift, np.eye(3), protocol.CARTESIAN_DELTA)

    values = spec.action(_wire(delta, _HOLD), measured, np.zeros(2)).values

    reached_pos, reached_rot = _gripper_pose(model, spec.LEFT, values[:6])
    np.testing.assert_allclose(reached_pos, start_pos + shift, atol=1e-3)
    np.testing.assert_allclose(reached_rot, start_rot, atol=1e-2)


def test_a_pose_out_of_reach_holds_its_arm_and_a_warning_names_it(caplog):
    far = _pose_command(np.array([2.0, 0.0, 2.0]), np.eye(3))

    values = spec.action(_wire(far, _HOLD), _MEASURED, np.zeros(2)).values

    np.testing.assert_array_equal(values[:6], _MEASURED[0])
    (record,) = caplog.records
    assert f'the {spec.LEFT} arm holds: no IK solution' in record.getMessage()


def test_a_channel_of_no_arm_is_refused():
    wire = {**_wire(_HOLD, _HOLD), protocol.ROBOT_COMMAND: _HOLD}

    with pytest.raises(ValueError, match='cannot act on channels'):
        spec.action(wire, _MEASURED, np.zeros(2))


@pytest.mark.parametrize(
    ('camera', 'fov_y_deg'),
    [(robolab_keys.TOP_CAMERA, 78.9), (robolab_keys.LEFT_WRIST_CAMERA, 85.9), (robolab_keys.RIGHT_WRIST_CAMERA, 85.7)],
)
def test_each_camera_has_square_pixels_and_the_fitted_field_of_view(camera, fov_y_deg):
    cam = spec.CAMERAS[camera]
    assert cam.horizontal_aperture / cam.vertical_aperture == pytest.approx(spec.CAMERA_WIDTH / spec.CAMERA_HEIGHT)
    fov_y = np.degrees(2 * np.arctan(cam.vertical_aperture / 2 / spec.CAMERA_FOCAL_LENGTH))
    assert fov_y == pytest.approx(fov_y_deg, abs=0.1)
