import pimm
from positronic.dataset import DatasetWriter
from positronic.dataset.ds_writer_agent import DsWriterAgent, Serializers, TimeMode


def wire(
    world: pimm.World,
    policy: pimm.ControlSystem,
    dataset_writer: DatasetWriter | None,
    cameras: dict[str, pimm.SignalEmitter] | None,
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

    ds_agent = None
    if dataset_writer is not None:
        ds_agent = DsWriterAgent(dataset_writer, time_mode=time_mode)

        for signal_name in cameras.keys():
            ds_agent.add_signal(signal_name, Serializers.camera_images)

        ds_agent.add_signal('target_grip')
        ds_agent.add_signal('robot_commands', Serializers.robot_command)
        # TODO: Controller positions must be bound outside of this function
        ds_agent.add_signal('robot_state', Serializers.robot_state)
        ds_agent.add_signal('grip')

        # Controller positions must be bound outside of this function
        # TODO: DS commands must be bound outside of this function
        world.connect(policy.robot_commands, ds_agent.inputs['robot_commands'])
        world.connect(policy.target_grip, ds_agent.inputs['target_grip'])
        world.connect(robot_arm.state, ds_agent.inputs['robot_state'])
        if gripper is not None:
            world.connect(gripper.grip, ds_agent.inputs['grip'])

    # Camera connections: one-to-one or one-to-many
    if cameras:
        for signal_name, emitter in cameras.items():
            if isinstance(emitter, pimm.FakeEmitter):
                continue

            # Collect all receivers for this camera signal
            receivers = [policy.frames[signal_name]]

            if ds_agent is not None:
                receivers.append(ds_agent.inputs[signal_name])
            if gui is not None:
                receivers.append(gui.cameras[signal_name])

            # filter out FakeReceiver
            receivers = [r for r in receivers if not isinstance(r, pimm.FakeReceiver)]
            if not receivers:
                raise ValueError(f"No valid receivers specified for camera signal '{signal_name}'.")

            # Connect emitter individually to each receiver
            for receiver in receivers:
                world.connect(emitter, receiver)

    return ds_agent
