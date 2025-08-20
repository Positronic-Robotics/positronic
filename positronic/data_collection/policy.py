import pimm
from positronic import geom
from positronic.drivers.roboarm import command as roboarm_command


def webxr_robot_policy(controller_pos: pimm.SignalReader[dict], operator_position: geom.Transform3D | None = None):
    prev_controller_pos: geom.Transform3D | None = None
    def _convert_controller_pos_to_robot_policy(controller_pos: dict):
        nonlocal prev_controller_pos

        controller_pos = controller_pos['right']
        if controller_pos is None:
            return roboarm_command.NoOp()

        if prev_controller_pos is None:
            prev_controller_pos = controller_pos
            return roboarm_command.NoOp()

        delta = geom.Transform3D(
            controller_pos['translation'] - prev_controller_pos.translation,
            geom.Rotation.from_quat(controller_pos['rotation']).inv * prev_controller_pos.rotation,
        )

        prev_controller_pos = controller_pos

        if operator_position is not None:
            delta = operator_position.inv * delta * operator_position  # TODO: check the math here

        return roboarm_command.RelativeCartesianMove(delta)


    return pimm.map(controller_pos, _convert_controller_pos_to_robot_policy)