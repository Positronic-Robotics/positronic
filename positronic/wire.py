import functools
from collections.abc import Mapping

import pimm
from positronic import keys, telemetry, telemetry_keys
from positronic.dataset.dataset import DatasetFactory
from positronic.dataset.ds_writer_agent import DsWriterAgent
from positronic.dataset.serializers import Serializers
from positronic.eval import ROBOT_STATIC_META, Embodiment, Observation
from positronic.policy.harness import Harness

__all__ = ['ROBOT_STATIC_META', 'wire', 'wire_embodiment']


def wire(  # noqa: C901
    world: pimm.World,
    harness: pimm.ControlSystem,
    dataset_factory: DatasetFactory | None,
    cameras: Mapping[str, pimm.SignalEmitter],
    robot_arm: pimm.ControlSystem | None,
    gripper: pimm.ControlSystem | None,
    gui: pimm.ControlSystem | None,
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
    if dataset_factory is not None:
        # A partial, never a lambda: the recorder may be spawned as a background process, and `World`
        # pickles a background control system whole.
        ds_agent = DsWriterAgent(
            dataset_factory, telemetry_span=functools.partial(telemetry.span, telemetry_keys.SPAN_RECORD_IO)
        )
        for signal_name in cameras.keys():
            ds_agent.add_signal(signal_name, Serializers.camera_images)
        if robot_arm is not None:
            ds_agent.add_signal(keys.ROBOT_COMMAND, Serializers.robot_command)
            ds_agent.add_signal(keys.ROBOT_STATE, Serializers.robot_state)
        if gripper is not None:
            ds_agent.add_signal(keys.TARGET_GRIP)
            ds_agent.add_signal(keys.GRIP)

        for signal_name, emitter in cameras.items():
            world.connect(emitter, ds_agent.inputs[signal_name])
        if robot_arm is not None:
            world.connect(harness.robot_commands, ds_agent.inputs[keys.ROBOT_COMMAND])
            world.connect(robot_arm.state, ds_agent.inputs[keys.ROBOT_STATE])
        if gripper is not None:
            world.connect(harness.target_grip, ds_agent.inputs[keys.TARGET_GRIP])
            world.connect(gripper.grip, ds_agent.inputs[keys.GRIP])

    if gui is not None:
        for signal_name, emitter in cameras.items():
            world.connect(emitter, gui.cameras[signal_name])

    return ds_agent


def wire_embodiment(
    world: pimm.World,
    harness: Harness,
    embodiment: Embodiment,
    *,
    privileged: dict[str, Observation] | None = None,
    done: pimm.SignalEmitter | None = None,
):
    """Connect policy inputs, commands, lifecycle handlers and recording-only privileged signals."""
    privileged = privileged or {}
    for name, obs in embodiment.observations.items():
        world.connect(obs.source, harness.observations[name])
    for name, cmd in embodiment.commands.items():
        world.connect(harness.commands[name], cmd.dest)
    for name, handler in embodiment.ready_handlers.items():
        world.connect(harness.ready[name], handler)
    for name, handler in embodiment.prepare_handlers.items():
        world.connect(harness.prepare[name], handler)
    if embodiment.meta_source is not None:
        world.connect(embodiment.meta_source, harness.robot_meta_in)
    if done is not None:
        world.connect(done, harness.done)

    for name, observation in privileged.items():
        world.connect(observation.source, harness.privileged[name])
