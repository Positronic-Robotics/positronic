import configuronic as cfn
import positronic.cfg.hardware.roboarm
import pimm
from positronic import geom
from positronic.drivers.roboarm import command


@cfn.config(robot=positronic.cfg.hardware.roboarm.kinova)
def main(robot):
    with pimm.World() as world:
        command_emmiter = world.pair(robot.commands)
        state_receiver = world.pair(robot.state)

        world.start([], background=robot)
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
                    print("Q", state_receiver.value.q)
                    print("DQ", state_receiver.value.dq)
                    print("EE", state_receiver.value.ee_pose.translation, state_receiver.value.ee_pose.rotation.as_quat)
                    print("Status", state_receiver.value.status)
                case ['quit']:
                    break
                case arg:
                    print(f"Unknown command: {arg}")


if __name__ == '__main__':
    cfn.cli(main)
