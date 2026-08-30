import functools
from collections.abc import Mapping

import pimm
from positronic import keys, telemetry, telemetry_keys
from positronic.dataset.ds_writer_agent import DatasetFactory, DsWriterAgent, TimeMode
from positronic.dataset.local_dataset import LocalDatasetWriter
from positronic.dataset.serializers import Serializers, StatefulSerializer
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
    if dataset_factory is not None:
        # A partial, never a lambda: the recorder may be spawned as a background process, and `World`
        # pickles a background control system whole.
        ds_agent = DsWriterAgent(
            dataset_factory,
            time_mode=time_mode,
            telemetry_span=functools.partial(telemetry.span, telemetry_keys.SPAN_RECORD_IO),
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


def _recorder(
    world: pimm.World, harness: Harness, embodiment: Embodiment, time_mode: TimeMode, privileged: dict[str, Observation]
) -> DsWriterAgent:
    """An embodiment's observations, command chunks and privileged ground-truth, recorded into the dataset
    each episode names."""
    ds_agent = DsWriterAgent(
        LocalDatasetWriter,
        time_mode=time_mode,
        virtual_time=embodiment.simulated,
        telemetry_span=functools.partial(telemetry.span, telemetry_keys.SPAN_RECORD_IO),
    )
    for name, obs in embodiment.observations.items():
        if isinstance(obs.serializer, StatefulSerializer):
            raise TypeError(f"observation '{name}': stateful serializer can't be shared by policy and record paths")
        ds_agent.add_signal(name, obs.serializer)
        world.connect(obs.source, ds_agent.inputs[name])
    for name, cmd in embodiment.commands.items():
        ds_agent.add_signal(name, cmd.serializer)
        world.connect(harness.commands[name], ds_agent.inputs[name])
    for name, priv in privileged.items():
        ds_agent.add_signal(name, priv.serializer)
        world.connect(priv.source, ds_agent.inputs[name])
    return ds_agent


def wire_embodiment(
    world: pimm.World,
    harness: Harness,
    embodiment: Embodiment,
    time_mode: TimeMode = TimeMode.CLOCK,
    *,
    record: bool = True,
    privileged: dict[str, Observation] | None = None,
    done: pimm.SignalEmitter | None = None,
):
    """Wire an embodiment to the Harness for the inference path.

    Connects device observation sources -> ``harness.observations``, ``harness.commands`` -> device
    receivers, ``harness.prepare`` -> everything a trial readies, and records observations, command chunks,
    and the eval's privileged ground-truth into the dataset each episode names. ``record`` off leaves the
    recorder out, so the episode commands reach nobody and the producers keep one consumer each. The ``done``
    terminating signal, when present, is connected to ``harness.done``. GUI camera wiring stays with the
    caller — it is a presentation concern, not part of the embodiment contract.
    """
    privileged = privileged or {}
    for name, obs in embodiment.observations.items():
        world.connect(obs.source, harness.observations[name])
    for name, cmd in embodiment.commands.items():
        world.connect(harness.commands[name], cmd.dest)
    for name, handler in embodiment.prepare_handlers.items():
        world.connect(harness.prepare[name], handler)
    if embodiment.meta_source is not None:
        world.connect(embodiment.meta_source, harness.robot_meta_in)
    if done is not None:
        world.connect(done, harness.done)

    if not record:
        return None
    return _recorder(world, harness, embodiment, time_mode, privileged)
