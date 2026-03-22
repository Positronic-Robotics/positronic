import pimm
from positronic.dataset import DatasetWriter
from positronic.dataset.ds_writer_agent import DsWriterAgent, Serializers, TimeMode

ROBOT_STATIC_META = {'joint_signal': 'robot_state.q', 'pose_signals': ['robot_state.ee_pose', 'robot_commands.pose']}


def wire(
    world: pimm.World,
    harness: pimm.ControlSystem,
    dataset_writer: DatasetWriter | None,
    cameras: dict[str, pimm.SignalEmitter] | None,
    robot_arm: pimm.ControlSystem | None,
    gripper: pimm.ControlSystem | None,
    gui: pimm.ControlSystem | None,
    time_mode: TimeMode = TimeMode.CLOCK,
):
    if robot_arm is not None:
        world.connect(harness.robot_commands, robot_arm.commands)
        world.connect(robot_arm.state, harness.robot_state)
        world.connect(robot_arm.robot_meta, harness.robot_meta_in)

    if gripper is not None:
        world.connect(harness.target_grip, gripper.target_grip)
        world.connect(gripper.grip, harness.gripper_state)

    for signal_name, emitter in cameras.items():
        world.connect(emitter, harness.frames[signal_name])

    ds_agent = None
    if dataset_writer is not None:
        ds_agent = DsWriterAgent(dataset_writer, time_mode=time_mode)
        for signal_name in cameras.keys():
            ds_agent.add_signal(signal_name, Serializers.camera_images)
        if robot_arm is not None:
            ds_agent.add_signal('robot_commands', Serializers.robot_command)
            ds_agent.add_signal('robot_state', Serializers.robot_state)
        if gripper is not None:
            ds_agent.add_signal('target_grip')
            ds_agent.add_signal('grip')

        for signal_name, emitter in cameras.items():
            world.connect(emitter, ds_agent.inputs[signal_name])
        if robot_arm is not None:
            world.connect(harness.robot_commands, ds_agent.inputs['robot_commands'])
            world.connect(robot_arm.state, ds_agent.inputs['robot_state'])
        if gripper is not None:
            world.connect(harness.target_grip, ds_agent.inputs['target_grip'])
            world.connect(gripper.grip, ds_agent.inputs['grip'])

    if gui is not None:
        for signal_name, emitter in cameras.items():
            world.connect(emitter, gui.cameras[signal_name])

    return ds_agent
