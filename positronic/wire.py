import pimm
from positronic import eval_timing
from positronic.dataset import DatasetWriter
from positronic.dataset.ds_writer_agent import DsWriterAgent, TimeMode, TrajectoryOverrideSerializer
from positronic.dataset.serializers import Serializers, StatefulSerializer
from positronic.eval import ROBOT_STATIC_META, Embodiment, Observation

__all__ = ['ROBOT_STATIC_META', 'wire', 'wire_embodiment']


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
        ds_agent = DsWriterAgent(dataset_writer, time_mode=time_mode, timing=eval_timing.WriterHooks())
        for signal_name in cameras.keys():
            ds_agent.add_signal(signal_name, Serializers.camera_images)
        if robot_arm is not None:
            # Policies emit whole trajectories; flatten with last-writer-wins so the
            # recording is a dense per-command stream (teleop emits bare commands,
            # which pass straight through). See TrajectoryOverrideSerializer.
            ds_agent.add_signal('robot_command', TrajectoryOverrideSerializer(Serializers.robot_command))
            ds_agent.add_signal('robot_state', Serializers.robot_state)
        if gripper is not None:
            ds_agent.add_signal('target_grip', TrajectoryOverrideSerializer(None))
            ds_agent.add_signal('grip')

        for signal_name, emitter in cameras.items():
            world.connect(emitter, ds_agent.inputs[signal_name])
        if robot_arm is not None:
            world.connect(harness.robot_commands, ds_agent.inputs['robot_command'])
            world.connect(robot_arm.state, ds_agent.inputs['robot_state'])
        if gripper is not None:
            world.connect(harness.target_grip, ds_agent.inputs['target_grip'])
            world.connect(gripper.grip, ds_agent.inputs['grip'])

    if gui is not None:
        for signal_name, emitter in cameras.items():
            world.connect(emitter, gui.cameras[signal_name])

    return ds_agent


def wire_embodiment(
    world: pimm.World,
    harness: pimm.ControlSystem,
    embodiment: Embodiment,
    dataset_writer: DatasetWriter | None,
    time_mode: TimeMode = TimeMode.CLOCK,
    privileged: dict[str, Observation] | None = None,
    done: pimm.SignalEmitter | None = None,
):
    """Wire an embodiment to the Harness for the inference path.

    Connects device observation sources -> ``harness.observations`` and
    ``harness.commands`` -> device receivers, and records observations, command
    chunks, and the task's privileged ground-truth into the dataset. The task's ``done``
    terminating signal, when present, is connected to ``harness.done``. GUI camera wiring
    stays with the caller — it is a presentation concern, not part of the embodiment contract.
    """
    privileged = privileged or {}
    for name, obs in embodiment.observations.items():
        world.connect(obs.source, harness.observations[name])
    for name, cmd in embodiment.commands.items():
        world.connect(harness.commands[name], cmd.dest)
    if embodiment.meta_source is not None:
        world.connect(embodiment.meta_source, harness.robot_meta_in)
    if done is not None:
        world.connect(done, harness.done)

    ds_agent = None
    if dataset_writer is not None:
        ds_agent = DsWriterAgent(
            dataset_writer, time_mode=time_mode, virtual_time=embodiment.simulated, timing=eval_timing.WriterHooks()
        )
        for name, obs in embodiment.observations.items():
            if isinstance(obs.serializer, StatefulSerializer):
                raise TypeError(f"observation '{name}': stateful serializer can't be shared by policy and record paths")
            ds_agent.add_signal(name, obs.serializer)
            world.connect(obs.source, ds_agent.inputs[name])
        for name, cmd in embodiment.commands.items():
            # Policies emit whole trajectories; flatten with last-writer-wins so the
            # recording is a dense per-command stream. See TrajectoryOverrideSerializer.
            ds_agent.add_signal(name, TrajectoryOverrideSerializer(cmd.serializer))
            world.connect(harness.commands[name], ds_agent.inputs[name])
        for name, priv in privileged.items():
            ds_agent.add_signal(name, priv.serializer)
            world.connect(priv.source, ds_agent.inputs[name])

    return ds_agent
