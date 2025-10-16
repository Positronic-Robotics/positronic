import pimm
from positronic.dataset import DatasetWriter
from positronic.dataset.ds_writer_agent import DsWriterAgent, Serializers, TimeMode


def wire(
    world: pimm.World,
    policy: pimm.ControlSystem,
    dataset_writer: DatasetWriter | None,
    cameras: dict[str, pimm.ControlSystem] | None,
    robot_arm: pimm.ControlSystem | None,
    gripper: pimm.ControlSystem | None,
    gui: pimm.ControlSystem | None,
    time_mode: TimeMode = TimeMode.CLOCK,
):
    world.connect(policy.robot_commands, robot_arm.commands)
    world.connect(robot_arm.state, policy.robot_state)

    if gripper is not None:
        world.connect(policy.target_grip, gripper.target_grip)
        world.connect(gripper.grip, policy.gripper_state)

    for camera_name, camera in cameras.items():
        world.connect(camera.frame, policy.frames[camera_name])

    ds_agent = None
    if dataset_writer is not None:
        ds_agent = DsWriterAgent(dataset_writer, time_mode=time_mode)
        for camera_name in cameras.keys():
            ds_agent.add_signal(camera_name, Serializers.camera_images)
        ds_agent.add_signal('target_grip')
        ds_agent.add_signal('robot_commands', Serializers.robot_command)
        # TODO: Controller positions must be binded outside of this function
        ds_agent.add_signal('robot_state', Serializers.robot_state)
        ds_agent.add_signal('grip')

        for camera_name, camera in cameras.items():
            world.connect(camera.frame, ds_agent.inputs[camera_name])
        # Controller positions must be binded outside of this function
        # TODO: DS commands must be binded outside of this function
        world.connect(policy.robot_commands, ds_agent.inputs['robot_commands'])
        world.connect(policy.target_grip, ds_agent.inputs['target_grip'])
        world.connect(robot_arm.state, ds_agent.inputs['robot_state'])
        if gripper is not None:
            world.connect(gripper.grip, ds_agent.inputs['grip'])

    if gui is not None:
        for camera_name, camera in cameras.items():
            world.connect(camera.frame, gui.cameras[camera_name])

    return ds_agent
