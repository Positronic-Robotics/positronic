import configuronic as cfgc
import pimm.cfg.hardware.roboarm
import ironic2 as ir
import geom
from pimm.drivers.roboarm import command


@cfgc.config(robot=pimm.cfg.hardware.roboarm.kinova)
def main(robot):
    with ir.World() as world:
        command_emmiter, robot.commands = world.mp_pipe()
        robot.state, state_reader = world.shared_memory()

        world.start_in_subprocess(robot.run)

        while True:
            text_command = input("Enter a command: ")

            match text_command.split(' '):
                case ['reset']:
                    command_emmiter.emit(command.Reset())
                case ['move', x, y, z, qw, qx, qy, qz]:
                    pos = [float(x) for x in [x, y, z]]
                    quat = geom.Rotation.from_quat([float(qw), float(qx), float(qy), float(qz)])
                    command_emmiter.emit(command.CartesianMove(geom.Transform3D(translation=pos, rotation=quat)))
                case ['joint_move', *args]:
                    args = [float(x) for x in args]
                    command_emmiter.emit(command.JointMove(positions=args))
                case ['info']:
                    print("Q", state_reader.value.q)
                    print("DQ", state_reader.value.dq)
                    print("EE", state_reader.value.ee_pose.translation, state_reader.value.ee_pose.rotation.as_quat)
                    print("Status", state_reader.value.status)
                case ['quit']:
                    break
                case arg:
                    print(f"Unknown command: {arg}")


if __name__ == '__main__':
    cfgc.cli(main)