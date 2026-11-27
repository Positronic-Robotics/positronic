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

    action, grips, _ = spec.action(
        _wire(_joint_command(left), _joint_command(right), (1.0, 0.0)), _MEASURED, np.zeros(2)
    )

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
    action, grips, _ = spec.action(_wire(_HOLD, _HOLD, (float('nan'), 1.0)), _MEASURED, np.array([1.0, 0.0]))

    assert (action[6], action[13]) == (1.0, 1.0)
    np.testing.assert_array_equal(grips, [1.0, 1.0])


def _pose_command(kind: str) -> dict:
    field = protocol.COMMAND_POSE if kind == protocol.CARTESIAN else protocol.COMMAND_DELTA
    return {protocol.COMMAND_TYPE: kind, field: np.concatenate([np.full(3, 0.1), np.eye(3).reshape(-1)])}


@pytest.mark.parametrize('kind', [protocol.CARTESIAN, protocol.CARTESIAN_DELTA])
def test_a_pose_command_holds_its_arm_with_no_log_and_the_other_arm_moves(kind, caplog):
    target = np.full(6, 0.3, dtype=np.float32)

    action = spec.action(_wire(_pose_command(kind), _joint_command(target)), _MEASURED, np.zeros(2))

    np.testing.assert_array_equal(action.values[:6], _MEASURED[0])
    np.testing.assert_array_equal(action.values[7:13], target)
    assert action.pose_refused == (spec.LEFT,)
    assert not caplog.records


def test_an_episode_of_pose_commands_warns_once_per_arm_and_reports_the_count_per_clone_and_arm():
    refusals = spec.PoseRefusals(num_envs=2)
    wire = _wire(_pose_command(protocol.CARTESIAN), _pose_command(protocol.CARTESIAN_DELTA))

    warnings = [w for _ in range(50) for w in refusals.add(1, spec.action(wire, _MEASURED, np.zeros(2)).pose_refused)]

    assert len(warnings) == 2
    assert all(spec.JOINT_COMMANDS_ONLY in w for w in warnings)
    assert refusals.report() == [
        f'slot 1: the {side} arm held on a pose command on 50 steps, as {spec.JOINT_COMMANDS_ONLY}'
        for side in spec.SIDES
    ]


def test_a_report_of_one_clone_zeroes_only_the_counts_of_that_clone():
    refusals = spec.PoseRefusals(num_envs=2)
    for slot in (0, 1):
        refusals.add(slot, spec.SIDES[:1])

    assert refusals.report([0]) == [
        f'slot 0: the {spec.SIDES[0]} arm held on a pose command on 1 steps, as {spec.JOINT_COMMANDS_ONLY}'
    ]
    assert refusals.report() == [
        f'slot 1: the {spec.SIDES[0]} arm held on a pose command on 1 steps, as {spec.JOINT_COMMANDS_ONLY}'
    ]
    assert refusals.report() == []


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
