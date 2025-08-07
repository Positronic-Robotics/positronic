from typing import Iterator
import ironic2 as ir
import numpy as np

import scservo_sdk as scs

PROTOCOL_VERSION = 0
TIMEOUT_MS = 1000

leader_calibration = {
    "homing_offsets": np.array([-109, 912, -194, -82, -865, -1427]),
    "range_mins": np.array([664, 1006, 688, 853, 225, 2036]),
    "range_maxs": np.array([3286, 3375, 2895, 3150, 4037, 3260]),
}

follower_calibration = {
    "homing_offsets": np.array([-855, 1089, -497, -140, 875, -1010]),
    "range_mins": np.array([889, 731, 969, 888, 23, 2034]),
    "range_maxs": np.array([3589, 3100, 3174, 3177, 3865, 3513]),
}


SCS_SERIES_CONTROL_TABLE = {
    "Model": (3, 2),
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Angle_Limit": (9, 2),
    "Max_Angle_Limit": (11, 2),
    "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1),
    "Min_Voltage_Limit": (15, 1),
    "Max_Torque_Limit": (16, 2),
    "Phase": (18, 1),
    "Unloading_Condition": (19, 1),
    "LED_Alarm_Condition": (20, 1),
    "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1),
    "I_Coefficient": (23, 1),
    "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1),
    "CCW_Dead_Zone": (27, 1),
    "Protection_Current": (28, 2),
    "Angular_Resolution": (30, 1),
    "Offset": (31, 2),
    "Mode": (33, 1),
    "Protective_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
    "Speed_closed_loop_P_proportional_coefficient": (37, 1),
    "Over_Current_Protection_Time": (38, 1),
    "Velocity_closed_loop_I_integral_coefficient": (39, 1),
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Speed": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),
    "Present_Speed": (58, 2),
    "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),
    "Present_Temperature": (63, 1),
    "Status": (65, 1),
    "Moving": (66, 1),
    "Present_Current": (69, 2),
    # Not in the Memory Table
    "Maximum_Acceleration": (85, 2),
}


def read_from_motor(port_handler, packet_handler, motor_indices: list[int], data_name: str) -> np.ndarray:
    """
    Read data from multiple motors using group sync read.

    Args:
        port_handler: The port handler for the serial connection
        packet_handler: The packet handler for communication protocol
        motor_indices: List of motor IDs to read from
        data_name: Name of the data to read (must be in SCS_SERIES_CONTROL_TABLE)

    Returns:
        np.ndarray: Array of values read from the motors

    Raises:
        KeyError: If data_name is not in the control table
        ConnectionError: If communication fails
    """
    if data_name not in SCS_SERIES_CONTROL_TABLE:
        raise KeyError(f"Data name '{data_name}' not found in control table")

    addr, bytes = SCS_SERIES_CONTROL_TABLE[data_name]
    group = scs.GroupSyncRead(port_handler, packet_handler, addr, bytes)

    for idx in motor_indices:
        group.addParam(idx)

    # Try to read with retries
    NUM_READ_RETRY = 20
    for _ in range(NUM_READ_RETRY):
        comm = group.txRxPacket()
        if comm == scs.COMM_SUCCESS:
            break

    if comm != scs.COMM_SUCCESS:
        raise ConnectionError(
            f"Read failed due to communication error on port {port_handler.port_name} for indices {motor_indices}: "
            f"{packet_handler.getTxRxResult(comm)}"
        )

    values = []
    for idx in motor_indices:
        value = group.getData(idx, addr, bytes)
        values.append(value)

    values = np.array(values)

    # Convert to signed int for position data
    CONVERT_UINT32_TO_INT32_REQUIRED = ["Goal_Position", "Present_Position"]
    if data_name in CONVERT_UINT32_TO_INT32_REQUIRED:
        values = values.astype(np.int32)

    return values


def convert_to_bytes(value, bytes):
    """
    Convert a value to the appropriate byte format for feetech motors.

    Args:
        value: The value to convert
        bytes: Number of bytes (1, 2, or 4)

    Returns:
        list: List of bytes representing the value

    Raises:
        NotImplementedError: If bytes is not 1, 2, or 4
    """
    # Note: No need to convert back into unsigned int, since this byte preprocessing
    # already handles it for us.
    if bytes == 1:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
        ]
    elif bytes == 2:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
        ]
    elif bytes == 4:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
        ]
    else:
        raise NotImplementedError(
            f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but "
            f"{bytes} is provided instead."
        )
    return data


def write_to_motor(port_handler, packet_handler, motor_indices: list[int], data_name: str, values: np.ndarray):
    """
    Write data to multiple motors using group sync write.

    Args:
        port_handler: The port handler for the serial connection
        packet_handler: The packet handler for communication protocol
        motor_indices: List of motor IDs to write to
        data_name: Name of the data to write (must be in SCS_SERIES_CONTROL_TABLE)
        values: Array of values to write to the motors

    Raises:
        KeyError: If data_name is not in the control table
        ConnectionError: If communication fails
        ValueError: If the number of values doesn't match the number of motor indices
    """
    if data_name not in SCS_SERIES_CONTROL_TABLE:
        raise KeyError(f"Data name '{data_name}' not found in control table")

    if len(values) != len(motor_indices):
        raise ValueError(f"Number of values ({len(values)}) must match number of motor indices ({len(motor_indices)})")

    addr, bytes = SCS_SERIES_CONTROL_TABLE[data_name]
    group = scs.GroupSyncWrite(port_handler, packet_handler, addr, bytes)

    for idx, value in zip(motor_indices, values, strict=True):
        data = convert_to_bytes(int(value), bytes)
        group.addParam(idx, data)

    # Try to write with retries
    NUM_WRITE_RETRY = 20
    for _ in range(NUM_WRITE_RETRY):
        comm = group.txPacket()
        if comm == scs.COMM_SUCCESS:
            break

    if comm != scs.COMM_SUCCESS:
        raise ConnectionError(
            f"Write failed due to communication error on port {port_handler.port_name} for indices {motor_indices}: "
            f"{packet_handler.getTxRxResult(comm)}"
        )


class MotorBus:
    target_position: ir.SignalReader = ir.NoOpReader()
    position: ir.SignalEmitter = ir.NoOpEmitter()

    def __init__(self, port: str, calibration: dict[str, np.ndarray], processing_freq: float = 1000.0):
        self.port = port
        self.motor_indices = [1, 2, 3, 4, 5, 6]
        self.processing_freq = processing_freq
        self.calibration = calibration

    def connect(self):
        port_handler = scs.PortHandler(self.port)
        packet_handler = scs.PacketHandler(PROTOCOL_VERSION)

        if not port_handler.openPort():
            raise OSError(f"Failed to open port '{self.port}'.")
        port_handler.setPacketTimeoutMillis(TIMEOUT_MS)
        return port_handler, packet_handler

    def run(self, should_stop: ir.SignalReader, clock: ir.Clock) -> Iterator[ir.Sleep]:
        rate_limit = ir.RateLimiter(hz=self.processing_freq, clock=clock)
        fps = ir
        port_handler, packet_handler = self.connect()

        while not should_stop.value:
            position = read_from_motor(port_handler, packet_handler, self.motor_indices, "Present_Position")
            position = self.apply_calibration(position)
            target_position = self.target_position.read()
            if target_position is not None:
                target_position = self.revert_calibration(target_position.data)
                write_to_motor(port_handler, packet_handler, self.motor_indices, "Goal_Position", target_position)
            self.position.emit(position)
            yield ir.Sleep(rate_limit.wait_time())

        port_handler.closePort()


    def apply_calibration(self, values: np.ndarray):
        # convert raw values to 0-1 range
        return (values - self.calibration["range_mins"]) / (self.calibration["range_maxs"] - self.calibration["range_mins"])

    def revert_calibration(self, values: np.ndarray, clip: bool = True):
        values = values * (self.calibration["range_maxs"] - self.calibration["range_mins"]) + self.calibration["range_mins"]
        if clip:
            values = np.clip(values, self.calibration["range_mins"], self.calibration["range_maxs"])
        return values