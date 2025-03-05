# In order to use Kinova API, you need to install the Kinova Kortex
# Go to https://artifactory.kinovaapps.com/ui/native/generic-public/kortex/API/2.7.0/
# Download the wheel file and install it using pip install kortex_api-2.7.0-py3-none-any.whl

import math
import threading
import time

import numpy as np
import pinocchio as pin
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.ControlConfigClientRpc import ControlConfigClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import ActuatorCyclic_pb2, ActuatorConfig_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2, ControlConfig_pb2, Session_pb2
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport

_TCP_PORT = 10000
_UDP_PORT = 10001

K_r = np.diag([0.3, 0.3, 0.3, 0.3, 0.18, 0.18, 0.18])
K_l = np.diag([75.0, 75.0, 75.0, 75.0, 40.0, 40.0, 40.0])
K_lp = np.diag([5.0, 5.0, 5.0, 5.0, 4.0, 4.0, 4.0])
K_p = np.diag([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0])
K_d = np.diag([3.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0])
K_r_inv = np.linalg.inv(K_r)
K_r_K_l = K_r @ K_l
_DT = 0.001


def _wrap_joint_angle(q):
    return q + np.mod(q + np.pi, 2 * np.pi) - np.pi


def _wrap_joint_angle_nearest(q, q_base):
    return q_base + np.mod(q - q_base + np.pi, 2 * np.pi) - np.pi


class DeviceConnection:

    def __init__(self,
                 ip_address: str,
                 port: int,
                 transport: TCPTransport | UDPTransport,
                 credentials: tuple[str, str] = ('admin', 'admin')):
        self.ip_address = ip_address
        self.port = port
        self.credentials = credentials
        self.session_manager = None
        self.transport = transport
        self.router = RouterClient(self.transport, RouterClient.basicErrorCallback)

    def __enter__(self):
        self.transport.connect(self.ip_address, self.port)
        if self.credentials[0] != '':
            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = self.credentials[0]
            session_info.password = self.credentials[1]
            session_info.session_inactivity_timeout = 10000  # (milliseconds)
            session_info.connection_inactivity_timeout = 2000  # (milliseconds)
            self.session_manager = SessionManager(self.router)
            self.session_manager.CreateSession(session_info)
        return self.router

    def __exit__(self, *_):
        if self.session_manager is not None:
            router_options = RouterClientSendOptions()
            router_options.timeout_ms = 1000
            self.session_manager.CloseSession(router_options)
        self.transport.disconnect()


class JointCompliantController:

    class LowPassFilter:

        def __init__(self, alpha, initial_value):
            assert 0 < alpha <= 1, 'Alpha must be between 0 and 1'
            self.alpha = alpha
            self.y = initial_value

        def filter(self, x):
            self.y = self.alpha * x + (1 - self.alpha) * self.y
            return self.y

    def __init__(self, actuator_count):
        self.q_s = None
        self.q_d = None
        self.dq_d = None
        self.q_n = None
        self.dq_n = None
        self.tau_filter = None

        self.actuator_count = actuator_count
        self.last_command_time = None
        self.otg = None
        self.otg_inp = None
        self.otg_out = None
        self.otg_res = None

        self.target_qpos = None

    def set_target_qpos(self, qpos):
        self.target_qpos = qpos

    def compute_torque(self, q, dq, tau, g):
        if self.q_s is None:
            self.q_s = q.copy()
            self.q_d = q.copy()
            self.dq_d = np.zeros_like(q)
            self.q_n = q.copy()
            self.dq_n = dq.copy()
            self.tau_filter = JointCompliantController.LowPassFilter(0.01, tau.copy())

            self.last_command_time = time.monotonic()
            self.otg = Ruckig(self.actuator_count, _DT)
            self.otg_inp = InputParameter(self.actuator_count)
            self.otg_out = OutputParameter(self.actuator_count)
            self.otg_inp.max_velocity = 4 * [math.radians(80)] + 3 * [math.radians(140)]
            self.otg_inp.max_acceleration = 4 * [math.radians(240)] + 3 * [math.radians(450)]
            self.otg_inp.current_position = q.copy()
            self.otg_inp.current_velocity = dq.copy()
            self.otg_inp.target_position = q.copy()
            self.otg_inp.target_velocity = np.zeros(self.actuator_count)
            self.otg_res = Result.Finished

        self.q_s = _wrap_joint_angle_nearest(q, self.q_s)
        dq_s = dq.copy()  # TODO: It seems that we don't need copy here
        tau_s_f = self.tau_filter.filter(tau)

        if self.target_qpos is not None:
            qpos = _wrap_joint_angle_nearest(self.target_qpos, self.q_s)
            self.otg_inp.target_position = qpos
            self.otg_res = Result.Working

            self.target_qpos = None

        if self.otg_res == Result.Working:
            self.otg_res = self.otg.update(self.otg_inp, self.otg_out)
            self.otg_out.pass_to_input(self.otg_inp)
            self.q_d[:] = self.otg_out.new_position
            self.dq_d[:] = self.otg_out.new_velocity

        tau_task = -K_p @ (self.q_n - self.q_d) - K_d @ (self.dq_n - self.dq_d) + g

        # Nominal motor plant
        ddq_n = K_r_inv @ (tau_task - tau_s_f)
        self.dq_n += ddq_n * _DT
        self.q_n += self.dq_n * _DT

        tau_f = K_r_K_l @ ((self.dq_n - dq_s) + K_lp @ (self.q_n - self.q_s))  # Nominal friction

        return tau_task + tau_f


class KinovaController:

    def __init__(self, ip):
        self.tcp_connection = DeviceConnection(ip, _TCP_PORT, TCPTransport())
        self.udp_connection = DeviceConnection(ip, _UDP_PORT, UDPTransport())

        self.base = BaseClient(self.tcp_connection.__enter__())
        self.base_cyclic = BaseCyclicClient(self.udp_connection.__enter__())
        self.actuator_config = ActuatorConfigClient(self.base.router)
        self.actuator_count = self.base.GetActuatorCount().count
        self.control_config = ControlConfigClient(self.base.router)
        device_manager = DeviceManagerClient(self.base.router)
        device_handles = device_manager.ReadAllDevices()
        self.actuator_device_ids = [
            handle.device_identifier for handle in device_handles.device_handle
            if handle.device_type in [Common_pb2.BIG_ACTUATOR, Common_pb2.SMALL_ACTUATOR]
        ]
        self.send_options = RouterClientSendOptions()
        self.send_options.timeout_ms = 3

        # Make sure actuators are in position mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        # Make sure arm is in high-level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        if self.base.GetArmState().active_state == Base_pb2.ARMSTATE_IN_FAULT:
            self.base.ClearFaults()
            while self.base.GetArmState().active_state != Base_pb2.ARMSTATE_SERVOING_READY:
                time.sleep(0.1)

        # Note: Torque commands are converted to current commands since
        # Kinova's torque controller is unable to achieve commanded torques.
        # See relevant GitHub issue: https://github.com/Kinovarobotics/kortex/issues/38
        self.torque_constant = np.array([11.0, 11.0, 11.0, 11.0, 7.6, 7.6, 7.6])
        self.current_limit_max = np.array([10.0, 10.0, 10.0, 10.0, 6.0, 6.0, 6.0])
        self.current_limit_min = -self.current_limit_max

        self.model = pin.buildModelFromUrdf('positronic/drivers/roboarm/kinova/model.urdf')
        self.data = self.model.createData()
        self.q_pin = np.zeros(self.model.nq)
        self.tool_frame_id = self.model.getFrameId('tool_frame')

        self.control_thread = None
        self.stop_event = threading.Event()
        self.joint_controller = None
        self.joint_controller_mutex = threading.Lock()

    def __enter__(self):
        print('Creating control thread')
        self.control_thread = threading.Thread(target=self.control_thread_loop)
        self.control_thread.start()
        return self

    def __exit__(self, *_):
        print('Shutting down control thread')
        self.stop_event.set()
        self.control_thread.join()

        self.tcp_connection.__exit__()
        self.udp_connection.__exit__()

    def _execute_reference_action(self, action_name):
        assert self.control_thread is None or not self.control_thread.is_alive(
        ), 'Arm must be in high-level servoing mode'

        # Make sure arm is in high-level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        # Retrieve reference action
        action_type = Base_pb2.RequestedActionType()
        action_type.action_type = Base_pb2.REACH_JOINT_ANGLES
        action_list = self.base.ReadAllActions(action_type)
        action_handle = None
        for action in action_list.action_list:
            if action.name == action_name:
                action_handle = action.handle
        if action_handle is None:
            return

        # Execute action
        end_or_abort_event = threading.Event()

        def check_for_end_or_abort(e):

            def check(notification, e=e):
                if notification.action_event in [Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT]:
                    e.set()

            return check

        notification_handle = self.base.OnNotificationActionTopic(check_for_end_or_abort(end_or_abort_event),
                                                                  Base_pb2.NotificationOptions())
        self.base.ExecuteActionFromReference(action_handle)
        end_or_abort_event.wait(20)
        self.base.Unsubscribe(notification_handle)

    def home(self):
        self._execute_reference_action('Home')

    def _update_state(self, base_feedback):
        q = np.zeros(self.actuator_count)
        dq = np.zeros(self.actuator_count)
        tau = np.zeros(self.actuator_count)
        for i in range(self.actuator_count):
            q[i] = base_feedback.actuators[i].position
            dq[i] = base_feedback.actuators[i].velocity
            tau[i] = base_feedback.actuators[i].torque

        np.deg2rad(q, out=q)
        np.deg2rad(dq, out=dq)
        np.negative(tau, out=tau)  # Raw torque readings are negative relative to actuator direction

        # Pinocchio joint configuration
        q_pin = np.array([
            math.cos(q[0]),
            math.sin(q[0]),
            q[1],
            math.cos(q[2]),
            math.sin(q[2]),
            q[3],
            math.cos(q[4]),
            math.sin(q[4]),
            q[5],
            math.cos(q[6]),
            math.sin(q[6]),
        ])

        gravity = pin.computeGeneralizedGravity(self.model, self.data, q_pin)
        return q, dq, tau, gravity

    def control_thread_loop(self):
        print('Starting control thread')
        base_command = BaseCyclic_pb2.Command()
        for _ in range(self.actuator_count):
            base_command.actuators.add()

        base_feedback = self.base_cyclic.RefreshFeedback()
        for i in range(self.actuator_count):
            base_command.actuators[i].flags = ActuatorCyclic_pb2.SERVO_ENABLE
            base_command.actuators[i].position = base_feedback.actuators[i].position
            base_command.actuators[i].current_motor = base_feedback.actuators[i].current_motor

        print('Initial current command:')
        print(', '.join(f'{base_command.actuators[i].current_motor: .5f}' for i in range(self.actuator_count)))

        # Set arm to low-level servoing mode
        self.base.SetServoingMode(Base_pb2.ServoingModeInformation(servoing_mode=Base_pb2.LOW_LEVEL_SERVOING))

        # Send first frame and update robot state
        base_feedback = self.base_cyclic.Refresh(base_command, 0, self.send_options)

        # Set actuators to current control mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('CURRENT')
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        with self.joint_controller_mutex:
            self.joint_controller = JointCompliantController(self.actuator_count)
            self.joint_controller.compute_torque(*self._update_state(base_feedback))

        last_ts = time.monotonic()
        count = 0
        while not self.stop_event.is_set():
            now_ts = time.monotonic()
            step_time = now_ts - last_ts
            if step_time > 0.004:  # 4 ms
                print(f'Warning: Step time {1000 * step_time:.3f} ms')

            if step_time >= 0.0005:  # TODO: Do we really need this check?
                last_ts = now_ts
                with self.joint_controller_mutex:
                    torque_command = self.joint_controller.compute_torque(*self._update_state(base_feedback))
                current_command = np.divide(torque_command, self.torque_constant)
                np.clip(current_command, self.current_limit_min, self.current_limit_max, out=current_command)
                if count % 20 == 0:
                    cur_cmd = ', '.join(f'{c: .5f}' for c in current_command)
                    q_s = ', '.join(f'{q: .3f}' for q in self.joint_controller.q_s)
                    q_t = ', '.join(f'{q: .3f}' for q in self.joint_controller.q_d)
                    print(f'{cur_cmd}|{q_s}|{q_t}')

                # Increment frame ID to ensure actuators can reject out-of-order frames
                base_command.frame_id = (base_command.frame_id + 1) % 65536

                for i in range(self.actuator_count):
                    base_command.actuators[i].current_motor = current_command[i]
                    # base_command.actuators[i].current_motor = base_feedback.actuators[i].current_motor
                    base_command.actuators[i].position = base_feedback.actuators[i].position
                    base_command.actuators[i].command_id = base_command.frame_id

                base_feedback = self.base_cyclic.Refresh(base_command, 0, self.send_options)
                count += 1

                # if count > 100000:
                #     break

        # Set actuators back to position control mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        # Set arm back to high-level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

    @property
    def current_qpos(self):
        with self.joint_controller_mutex:
            if self.joint_controller is None:
                return None
            return self.joint_controller.q_s

    def set_target_qpos(self, qpos):
        with self.joint_controller_mutex:
            if self.joint_controller is None:
                return
            self.joint_controller.set_target_qpos(qpos)


if __name__ == '__main__':
    arm = KinovaController('192.168.1.10')
    # arm._execute_reference_action('Retract')
    arm.home()
    with arm:
        time.sleep(0.5)
        q_retract = np.array([0.0, -0.34906585, 3.14159265, -2.54818071, 0.0, -0.87266463, 1.57079633])
        arm.set_target_qpos(q_retract)
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                arm.stop_event.set()
                break

    print('Done')
