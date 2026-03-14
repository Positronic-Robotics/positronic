import select
import sys
from collections.abc import Iterator

import pimm
from positronic.drivers.motors.feetech import MotorBus
from positronic.drivers.roboarm import command as roboarm_command


class LeaderFollower(pimm.ControlSystem):
    """Reads positions from a leader arm and sends joint commands to a follower.

    This control system enables teleoperation by reading the leader arm's joint positions
    and emitting them as commands for the follower arm. The gripper position is emitted
    separately for recording.
    """

    def __init__(self, leader_bus: MotorBus):
        self.leader_bus = leader_bus

        # Emitters for controlling the follower
        self.robot_commands: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)
        self.target_grip: pimm.SignalEmitter[float] = pimm.ControlSystemEmitter(self)

        # Emitter for dataset agent commands
        self.ds_agent_commands: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)

        # Sound feedback
        self.sound: pimm.SignalEmitter = pimm.ControlSystemEmitter(self)

        # Unused but required for compatibility with wire()
        self.robot_state: pimm.SignalReceiver = pimm.FakeReceiver(self)
        self.gripper_state: pimm.SignalReceiver = pimm.FakeReceiver(self)
        self.frames: pimm.ReceiverDict = pimm.ReceiverDict(self, fake=True)
        self.controller_positions: pimm.SignalReceiver = pimm.FakeReceiver(self)

    def run(self, should_stop: pimm.SignalReceiver, clock: pimm.Clock) -> Iterator[pimm.Sleep]:
        from positronic.dataset.ds_writer_agent import DsWriterCommand, DsWriterCommandType

        start_wav_path = 'positronic/assets/sounds/recording-has-started.wav'
        end_wav_path = 'positronic/assets/sounds/recording-has-stopped.wav'

        self.leader_bus.connect()
        # Disable torque so leader can be moved freely
        self.leader_bus.set_torque_mode(False)

        rate_limit = pimm.RateLimiter(hz=100, clock=clock)
        recording = False
        last_record_toggle = clock.now_ns()
        debounce_ns = 500_000_000  # 500ms debounce

        print('================================================================')
        print('Leader-Follower teleoperation ready!')
        print('Press ENTER to toggle recording, Ctrl+C to stop')
        print('================================================================')

        while not should_stop.value:
            # Check for keyboard input (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()  # consume the line
                now = clock.now_ns()
                if (now - last_record_toggle) > debounce_ns:
                    op = DsWriterCommandType.START_EPISODE if not recording else DsWriterCommandType.STOP_EPISODE
                    self.ds_agent_commands.emit(DsWriterCommand(op, {}))
                    self.sound.emit(start_wav_path if not recording else end_wav_path)
                    recording = not recording
                    last_record_toggle = now
                    print(f'Recording: {"STARTED" if recording else "STOPPED"}')

            # Read leader position and send to follower
            leader_pos = self.leader_bus.position
            joint_pos = leader_pos[:-1]  # All joints except gripper
            grip_pos = leader_pos[-1]  # Last motor is gripper

            self.robot_commands.emit(roboarm_command.NormalizedJointPosition(joint_pos))
            self.target_grip.emit(grip_pos)

            yield pimm.Sleep(rate_limit.wait_time())

        self.leader_bus.disconnect()
