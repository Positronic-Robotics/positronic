import time

import numpy as np

from kortex_api.autogen.client_stubs.ActuatorConfigClientRpc import ActuatorConfigClient
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
from kortex_api.autogen.messages import (ActuatorCyclic_pb2, ActuatorConfig_pb2, Base_pb2, BaseCyclic_pb2, Common_pb2,
                                         Session_pb2)
from kortex_api.RouterClient import RouterClient, RouterClientSendOptions
from kortex_api.SessionManager import SessionManager
from kortex_api.TCPTransport import TCPTransport
from kortex_api.UDPTransport import UDPTransport


_TCP_PORT = 10000
_UDP_PORT = 10001


class DeviceConnection:
    """Manages TCP/UDP connections to a Kinova device."""

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


class KinovaAPI:
    """Low-level interface to the Kinova arm hardware."""

    def __init__(self, ip):
        self.ip = ip
        self.tcp_connection = DeviceConnection(ip, _TCP_PORT, TCPTransport())
        self.udp_connection = DeviceConnection(ip, _UDP_PORT, UDPTransport())

        self.base = None
        self.base_cyclic = None
        self.actuator_config = None
        self.actuator_count = None
        self.actuator_device_ids = None

        self.current_limit_max = np.array([10.0, 10.0, 10.0, 10.0, 6.0, 6.0, 6.0])
        self.current_limit_min = -self.current_limit_max

        self.base_feedback, self.base_command = None, None

    def __enter__(self):
        tcp_router = self.tcp_connection.__enter__()
        udp_router = self.udp_connection.__enter__()
        self.base = BaseClient(tcp_router)
        self.base_cyclic = BaseCyclicClient(udp_router)
        self.actuator_config = ActuatorConfigClient(self.base.router)
        self.actuator_count = self.base.GetActuatorCount().count

        device_manager = DeviceManagerClient(self.base.router)
        device_handles = device_manager.ReadAllDevices()
        self.actuator_device_ids = [
            handle.device_identifier for handle in device_handles.device_handle
            if handle.device_type in [Common_pb2.BIG_ACTUATOR, Common_pb2.SMALL_ACTUATOR]
        ]

        # Configure communication
        self.send_options = RouterClientSendOptions()
        self.send_options.timeout_ms = 30

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
        # Initialize control loop
        self.base.SetServoingMode(Base_pb2.ServoingModeInformation(servoing_mode=Base_pb2.LOW_LEVEL_SERVOING))

        self.base_feedback = self.base_cyclic.RefreshFeedback()
        self.base_command = BaseCyclic_pb2.Command()
        for i in range(self.actuator_count):
            self.base_command.actuators.add(flags=ActuatorCyclic_pb2.SERVO_ENABLE)

        self.apply_current_command(None)

        # Set actuators to current control mode
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('CURRENT')
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        return self

    def __exit__(self, *_):
        control_mode_message = ActuatorConfig_pb2.ControlModeInformation()
        control_mode_message.control_mode = ActuatorConfig_pb2.ControlMode.Value('POSITION')
        for device_id in self.actuator_device_ids:
            self.actuator_config.SetControlMode(control_mode_message, device_id)

        # Set arm back to high-level servoing mode
        base_servo_mode = Base_pb2.ServoingModeInformation()
        base_servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(base_servo_mode)

        self.tcp_connection.__exit__()
        self.udp_connection.__exit__()

    def apply_current_command(self, current_command):
        if current_command is not None:
            np.clip(current_command, self.current_limit_min, self.current_limit_max, out=current_command)

        # Increment frame ID to ensure actuators can reject out-of-order frames
        self.base_command.frame_id = (self.base_command.frame_id + 1) % 65536
        for i in range(self.actuator_count):
            self.base_command.actuators[i].current_motor = (current_command[i] if current_command is not None else
                                                            self.base_feedback.actuators[i].current_motor)
            self.base_command.actuators[i].position = self.base_feedback.actuators[i].position
            self.base_command.actuators[i].command_id = self.base_command.frame_id

        self.base_feedback = self.base_cyclic.Refresh(self.base_command, 0, self.send_options)

        q, dq, tau = np.zeros(self.actuator_count), np.zeros(self.actuator_count), np.zeros(self.actuator_count)

        for i in range(self.actuator_count):
            q[i] = self.base_feedback.actuators[i].position
            dq[i] = self.base_feedback.actuators[i].velocity
            tau[i] = self.base_feedback.actuators[i].torque

        np.deg2rad(q, out=q)
        np.deg2rad(dq, out=dq)
        np.negative(tau, out=tau)  # Raw torque readings are negative relative to actuator direction
        return q, dq, tau
