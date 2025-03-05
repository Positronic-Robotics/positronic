# flake8: noqa

# In order to use Kinova API, you need to install the Kinova Kortex
# Go to https://artifactory.kinovaapps.com/ui/native/generic-public/kortex/API/2.7.0/
# Download the wheel file and install it using pip install kortex_api-2.7.0-py3-none-any.whl


# Not that this is not actually working.
# The main problem that I was not able to solve is forward kinematics that aligned
# with inverse kinematics.
#
# The robot's forward kinematics throws exception when called, complaining about
# unavability of sequencer. Most likely this is related to the fact that the robot
# is running in low level servoing mode, and forward kinematics is on single-
# level servoing. Switching to single-level servoing before calling forward
# kinematics did not fix the problem either
#
# The other approach I tried is to learn the forward kinematics from data. Please
# refer to kinova_kinematics.ipynb for details. I managed to get it working, but
# the respective inverse kinematics returned very unstable results, and the arm
# was moving like crazy.

import logging
import threading
import time

import numpy as np
from control.world import MainThreadWorld
import ruckig
from scipy.optimize import least_squares

import collections
# This is required because kortex_api relies on the older Python version
collections.MutableMapping = collections.abc.MutableMapping
collections.MutableSequence = collections.abc.MutableSequence
collections.MutableSet = collections.abc.MutableSet

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2
from kortex_api.Exceptions.KServerException import KServerException

from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport

from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2

from control import ControlSystem, World, control_system
from control.utils import control_system_fn
from geom import Rotation, Transform3D, degrees_to_radians, radians_to_degrees

logger = logging.getLogger(__name__)

_KINOVA_CHAIN = [
    Transform3D([-7.732256199233234e-05, 0.0004820869071409106, 0.1384558230638504],
                Rotation(0.9999598860740662, -0.0014847038546577096, -0.0004586570430546999, 0.008817019872367382)),
    Transform3D([-0.0020223343744874, -0.11016443371772766, 0.14530658721923828],
                Rotation(0.7064821720123291, -0.7075875997543335, 0.014157610014081001, -0.0015306948916986585)),
    Transform3D([-0.012653934769332409, -0.6051681637763977, 0.0995088443160057],
                Rotation(0.7058353424072266, -0.7080992460250854, 0.019063977524638176, 0.005337915848940611)),
    Transform3D([0.00476058479398489, 0.14438331127166748, 0.18294934928417206],
                Rotation(0.70661860704422, 0.7065088152885437, -0.03622652217745781, 0.014936398714780807)),
    Transform3D([-0.031046129763126373, -0.4425973892211914, 0.13292159140110016],
                Rotation(0.7047972083091736, -0.7014123201370239, 0.09579798579216003, 0.045874472707509995)),
    Transform3D([0.028989970684051514, 0.12411590665578842, 0.129145547747612],
                Rotation(0.703943133354187, 0.6839388608932495, -0.1895177811384201, 0.027831479907035828)),
    Transform3D([-0.051960721611976624, -0.15933051705360413, 0.0792686715722084],
                Rotation(0.9999746084213257, 0.0043069422245025635, -0.0023099626414477825, 0.005186134018003941)),
    Transform3D([0.00040886219358071685, -0.000173802807694301, 0.04850476235151291],
                Rotation(0.9999657869338989, 0.0035003339871764183, -0.0015276813646778464, 0.007339356932789087)),
]

_KINOVA_MATRICES = np.array([t.as_matrix for t in _KINOVA_CHAIN])


def _forward_kinematics_matrix(joints):
    joints = degrees_to_radians(joints)
    joints[0] = -joints[0]  # This is on purpose, the first joint is inverted

    result_matrix = np.eye(4)
    for i in range(8):
        T_joint = _KINOVA_MATRICES[i]
        if i < 7:
            theta = joints[i]
            R_joint = np.array([[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta),
                                                                        np.cos(theta), 0, 0], [0, 0, 1, 0],
                                [0, 0, 0, 1]])
            T_joint = T_joint @ R_joint
        result_matrix = result_matrix @ T_joint
    return result_matrix


def _forward_kinematics(joints):
    return Transform3D.from_matrix(_forward_kinematics_matrix(joints))


def _inverse_kinematics(target, initial_position=None):
    if initial_position is None:
        initial_position = np.zeros(7)
    initial_position = np.array(initial_position, dtype=np.float64)
    initial_position = np.mod(initial_position + 180, 360) - 180
    target = target.as_matrix

    def objective_function(joint_angles):
        fk_result = _forward_kinematics_matrix(joint_angles)
        position_error = fk_result[:3, 3] - target[:3, 3]
        # orientation_error = 0.1 * (fk_result[:3, :3] - target[:3, :3])
        # regularization = 0.0001 * degrees_to_radians(np.linalg.norm(joint_angles - initial_position))
        # return np.concatenate((position_error, orientation_error.flatten()))
        return position_error

    result = least_squares(objective_function, initial_position, bounds=(-180, 180))
    return result.x


def _normalise_angles(target, base, period=360):
    res = target - base
    res[res > period / 2] -= period
    res[res < -period / 2] += period
    return base + res


def _feedback_to_joints(feedback: BaseCyclic_pb2.Feedback):
    return np.array([a.position for a in feedback.actuators])


def _feedback_to_pos(feedback: BaseCyclic_pb2.Feedback):
    return Transform3D(
        np.array([feedback.base.tool_pose_x, feedback.base.tool_pose_y, feedback.base.tool_pose_z]),
        Rotation.from_euler(
            [feedback.base.tool_pose_theta_x, feedback.base.tool_pose_theta_y, feedback.base.tool_pose_theta_z]))


class KinovaController:

    def __init__(self, ip: str):
        self.stop_event = threading.Event()
        self.target_joints = None
        self.target_lock = threading.Lock()

        self.output_lock = threading.Lock()
        self.output = None

        self.base = None
        self.last_joints = None
        self.last_pos = None

    def inverse_kinematics(self, target_pos):
        res = _inverse_kinematics(target_pos, self.last_joints)
        res = np.mod(res + np.pi, 2 * np.pi) - np.pi
        return radians_to_degrees(res)
        # region old inverse kinematics
        # euler = radians_to_degrees(target_pos.quaternion.as_euler)
        # ik_data = Base_pb2.IKData()
        # ik_data.cartesian_pose.x = target_pos.translation[0]
        # ik_data.cartesian_pose.y = target_pos.translation[1]
        # ik_data.cartesian_pose.z = target_pos.translation[2]
        # ik_data.cartesian_pose.theta_x = euler[0]
        # ik_data.cartesian_pose.theta_y = euler[1]
        # ik_data.cartesian_pose.theta_z = euler[2]

        # for a in self.last_joints:
        #     ja = ik_data.guess.joint_angles.add()
        #     ja.value = a

        # try:
        #     joint_angels = self.base.ComputeInverseKinematics(ik_data)
        #     return np.array([a.value for a in joint_angels.joint_angles])
        # except KServerException as e:
        #    logger.error(f"Failed to compute inverse kinematics: {e}")
        #    return None
        # endregion

    def forward_kinematics(self, joints):
        return _forward_kinematics(joints)

    def run(self):
        # region kortex setup
        transport = TCPTransport()
        router = RouterClient(transport, RouterClient.basicErrorCallback)
        transport.connect('192.168.1.10', 10000)

        session_info = Session_pb2.CreateSessionInfo(username="admin",
                                                     password="admin",
                                                     session_inactivity_timeout=1000 * 1000,
                                                     connection_inactivity_timeout=1000 * 1000)

        sessionManager = SessionManager(router)
        sessionManager.CreateSession(session_info)

        self.base = BaseClient(router)
        self.base.SetServoingMode(Base_pb2.ServoingModeInformation(servoing_mode=Base_pb2.SINGLE_LEVEL_SERVOING))
        self.base.ClearFaults()

        udp_transport = UDPTransport()
        upd_router = RouterClient(udp_transport, RouterClient.basicErrorCallback)
        udp_transport.connect('192.168.1.10', 10001)

        udp_session = SessionManager(upd_router)
        udp_session.CreateSession(session_info)
        base_cyclic = BaseCyclicClient(upd_router)
        # endregion
        # region home position
        action_list = self.base.ReadAllActions(Base_pb2.RequestedActionType(action_type=Base_pb2.REACH_JOINT_ANGLES))
        action_handle = [a.handle for a in action_list.action_list if a.name == "Home"][0]

        def check(n, e):
            if n.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
                print(f"Home position reached: {Base_pb2.ActionEvent.Name(n.action_event)}")
                e.set()

        e = threading.Event()
        notification_handle = self.base.OnNotificationActionTopic(lambda n: check(n, e), Base_pb2.NotificationOptions())

        self.base.ExecuteActionFromReference(action_handle)
        finished = e.wait(30)
        if not finished:
            raise Exception("Home position not reached")
        self.base.Unsubscribe(notification_handle)
        # endregion

        control = ruckig.RuckigNoThrow(7, 1 / 1000)  # 1000 Hz
        inp_ctrl = ruckig.InputParameter(7)

        inp_ctrl.target_velocity = [0.0] * 7
        inp_ctrl.target_acceleration = [0.0] * 7

        inp_ctrl.max_position = [np.inf, 128.9, np.inf, 147.8, np.inf, 120.3, np.inf]
        inp_ctrl.min_position = [-np.inf, -128.9, -np.inf, -147.8, -np.inf, -120.3, -np.inf]
        inp_ctrl.max_velocity = [25] * 7  # 50
        inp_ctrl.max_acceleration = [57.3] * 4 + [570] * 3
        inp_ctrl.max_jerk = [450] * 7

        target_joints = None
        command = BaseCyclic_pb2.Command()
        for i in range(7):
            act = command.actuators.add()

        send_option = RouterClientSendOptions()
        send_option.timeout_ms = 100

        feedback = base_cyclic.RefreshFeedback()
        self.last_joints = _feedback_to_joints(feedback)
        inp_ctrl.current_position = self.last_joints
        inp_ctrl.current_velocity = ([a.velocity for a in feedback.actuators])
        with self.output_lock:
            self.output = self.last_joints, _feedback_to_pos(feedback)
            print(f"Initial joints: {self.last_joints}")

        out = ruckig.OutputParameter(7)
        ruckig.Result.Working
        start = time.monotonic()
        count = 0

        self.base.SetServoingMode(Base_pb2.ServoingModeInformation(servoing_mode=Base_pb2.LOW_LEVEL_SERVOING))
        try:
            while not self.stop_event.is_set():
                with self.target_lock:
                    if self.target_joints is not None:
                        if target_joints is None:
                            start = time.monotonic()

                        target_joints = self.target_joints
                        self.target_joints = None

                if target_joints is None:
                    time.sleep(0.1)
                    continue

                inp_ctrl.target_position = _normalise_angles(target_joints, inp_ctrl.current_position)

                control.update(inp_ctrl, out)
                out.pass_to_input(inp_ctrl)

                command.frame_id = (command.frame_id + 1) % 65536
                for i in range(7):
                    command.actuators[i].command_id = command.frame_id
                    # command.actuators[i].position = feedback.actuators[i].position  # out.new_position[i]
                    command.actuators[i].position = out.new_position[i]

                feedback = base_cyclic.Refresh(command, 0, send_option)
                self.last_joints = _feedback_to_joints(feedback)
                pos = self.forward_kinematics(self.last_joints)
                with self.output_lock:
                    self.output = self.last_joints, pos

                count += 1
                if count % 1000 == 999:
                    print(f'Running {count / (time.monotonic() - start):.2f} hz')
                    # logger.debug(f'Running {count / (time.monotonic() - start):.2f} hz')
                    count, start = 0, time.monotonic()

        finally:
            self.base.SetServoingMode(Base_pb2.ServoingModeInformation(servoing_mode=Base_pb2.SINGLE_LEVEL_SERVOING))
            self.base.Stop()


# @control_system(inputs=["target_position"], outputs=["position", "joint_positions"])
# class Kinova(ControlSystem):
#     def __init__(self, world: World, ip: str):
#         super().__init__(world)
#         self.ip = ip

#     def run(self):
#         # TODO: If we ever return to working with Kinova, merge the controller into this class
#         controller = KinovaController(self.ip)

#         pos, joints = None, None
#         control_thread = threading.Thread(target=controller.run)
#         control_thread.start()

#         try:
#             for _ts, value in self.ins.target_position.read(1 / 40):
#                 target_joints = controller.inverse_kinematics(value)
#                 if target_joints is not None:
#                     with controller.target_lock:
#                         controller.target_joints = target_joints
#                         print(f"Target joints: {target_joints}")

#                 with controller.output_lock:
#                     if controller.output is not None:
#                         joints, pos = controller.output
#                         joints = np.array(joints)
#                         controller.output = None

#                 if joints is not None:
#                     pos = controller.forward_kinematics(joints)
#                     # print(f"Joints: {np.array2string(joints, precision=3, suppress_small=True)} - Position: {pos}")
#                     self.outs.position.write(pos)
#                     self.outs.joint_positions.write(degrees_to_radians(joints))
#                     joints = None
#         finally:
#             controller.stop_event.set()
#             control_thread.join()

# def _main():
#     world = MainThreadWorld()

#     kinova = Kinova(world, '192.168.1.10')

#     @control_system_fn(inputs=["robot_pos", "joints"], outputs=["target_pos"])
#     def controller(ins, outs):
#         _, robot_pos = ins.robot_pos.read()
#         start_time = time.time()
#         last_command_time = None
#         for input_name, _ts, value in ins.joints.read(1 / 50):
#             # if input_name == "robot_pos":
#             #     print(robot_pos)

#             if last_command_time is None or time.time() - last_command_time > 0.2:
#                 t = (time.time() - start_time) / 8 * 2 * np.pi      # One period every 10 seconds
#                 delta = np.array([0., np.cos(t), np.sin(t)]) * 0.20  # Radius of 20 cm
#                 target = Transform3D(robot_pos.translation + delta, robot_pos.quaternion)
#                 last_command_time = time.time()
#                 outs.target_pos.write(target)

#             if time.time() - start_time > 30:
#                 break

#     manager = controller(world)
#     manager.ins.robot_pos = kinova.outs.position
#     manager.ins.joints = kinova.outs.joint_positions
#     kinova.ins.target_position = manager.outs.target_pos

#     world.run()

if __name__ == "__main__":
    _main()
