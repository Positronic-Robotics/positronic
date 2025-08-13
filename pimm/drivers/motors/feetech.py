"""This code is adopted from https://github.com/huggingface/lerobot/blob/0878c6880fa4fbadf0742751cf7b015f2d63a769/src/lerobot/motors/feetech/feetech.py"""  # noqa: E501
import numpy as np
import scservo_sdk as scs


PROTOCOL_VERSION = 0
TIMEOUT_MS = 1000

# Sign-Magnitude encoding bits
STS_SMS_SERIES_ENCODINGS_TABLE = {
    "Homing_Offset": 11,
    "Goal_Velocity": 15,
    "Present_Velocity": 15,
}

STS_SMS_SERIES_CONTROL_TABLE = {
    # EPROM
    "Firmware_Major_Version": (0, 1),  # read-only
    "Firmware_Minor_Version": (1, 1),  # read-only
    "Model_Number": (3, 2),  # read-only
    "ID": (5, 1),
    "Baud_Rate": (6, 1),
    "Return_Delay_Time": (7, 1),
    "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2),
    "Max_Position_Limit": (11, 2),
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
    "Homing_Offset": (31, 2),
    "Operating_Mode": (33, 1),
    "Protective_Torque": (34, 1),
    "Protection_Time": (35, 1),
    "Overload_Torque": (36, 1),
    "Velocity_closed_loop_P_proportional_coefficient": (37, 1),
    "Over_Current_Protection_Time": (38, 1),
    "Velocity_closed_loop_I_integral_coefficient": (39, 1),
    # SRAM
    "Torque_Enable": (40, 1),
    "Acceleration": (41, 1),
    "Goal_Position": (42, 2),
    "Goal_Time": (44, 2),
    "Goal_Velocity": (46, 2),
    "Torque_Limit": (48, 2),
    "Lock": (55, 1),
    "Present_Position": (56, 2),  # read-only
    "Present_Velocity": (58, 2),  # read-only
    "Present_Load": (60, 2),  # read-only
    "Present_Voltage": (62, 1),  # read-only
    "Present_Temperature": (63, 1),  # read-only
    "Status": (65, 1),  # read-only
    "Moving": (66, 1),  # read-only
    "Present_Current": (69, 2),  # read-only
    "Goal_Position_2": (71, 2),  # read-only
    # Factory
    "Moving_Velocity": (80, 1),
    "Moving_Velocity_Threshold": (80, 1),
    "DTs": (81, 1),  # (ms)
    "Velocity_Unit_factor": (82, 1),
    "Hts": (83, 1),  # (ns) valid for firmware >= 2.54, other versions keep 0
    "Maximum_Velocity_Limit": (84, 1),
    "Maximum_Acceleration": (85, 1),
    "Acceleration_Multiplier ": (86, 1),  # Acceleration multiplier in effect when acceleration is 0
}

CONVERT_UINT32_TO_INT32_REQUIRED = ["Goal_Position", "Present_Position"]


def encode_sign_magnitude(value: int, sign_bit_index: int):
    """
    https://en.wikipedia.org/wiki/Signed_number_representations#Sign%E2%80%93magnitude
    """
    max_magnitude = (1 << sign_bit_index) - 1
    magnitude = abs(value)
    if magnitude > max_magnitude:
        raise ValueError(f"Magnitude {magnitude} exceeds {max_magnitude} (max for {sign_bit_index=})")

    direction_bit = 1 if value < 0 else 0
    return (direction_bit << sign_bit_index) | magnitude


def decode_sign_magnitude(encoded_value: int, sign_bit_index: int):
    """
    https://en.wikipedia.org/wiki/Signed_number_representations#Sign%E2%80%93magnitude
    """
    direction_bit = (encoded_value >> sign_bit_index) & 1
    magnitude_mask = (1 << sign_bit_index) - 1
    magnitude = encoded_value & magnitude_mask
    return -magnitude if direction_bit else magnitude


def read_from_motor(port_handler, packet_handler, motor_indices: list[int], data_name: str) -> np.ndarray:
    """
    Read data from multiple motors using group sync read.

    Args:
        port_handler: The port handler for the serial connection
        packet_handler: The packet handler for communication protocol
        motor_indices: List of motor IDs to read from
        data_name: Name of the data to read (must be in STS_SMS_SERIES_CONTROL_TABLE)

    Returns:
        np.ndarray: Array of values read from the motors

    Raises:
        KeyError: If data_name is not in the control table
        ConnectionError: If communication fails
    """
    if data_name not in STS_SMS_SERIES_CONTROL_TABLE:
        raise KeyError(f"Data name '{data_name}' not found in control table")

    addr, message_bytes = STS_SMS_SERIES_CONTROL_TABLE[data_name]
    group = scs.GroupSyncRead(port_handler, packet_handler, addr, message_bytes)

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
        value = group.getData(idx, addr, message_bytes)
        values.append(value)

    if data_name in STS_SMS_SERIES_ENCODINGS_TABLE:
        sign_bit_index = STS_SMS_SERIES_ENCODINGS_TABLE[data_name]
        values = [decode_sign_magnitude(value, sign_bit_index) for value in values]
    values = np.array(values)

    # Convert to signed int for position data
    if data_name in CONVERT_UINT32_TO_INT32_REQUIRED:
        values = values.astype(np.int32)

    return values


def convert_to_bytes(value, n_bytes):
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
    if n_bytes == 1:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
        ]
    elif n_bytes == 2:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
        ]
    elif n_bytes == 4:
        data = [
            scs.SCS_LOBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_LOWORD(value)),
            scs.SCS_LOBYTE(scs.SCS_HIWORD(value)),
            scs.SCS_HIBYTE(scs.SCS_HIWORD(value)),
        ]
    else:
        raise NotImplementedError(
            f"Value of the number of bytes to be sent is expected to be in [1, 2, 4], but "
            f"{n_bytes} is provided instead."
        )
    return data


def write_to_motor(port_handler, packet_handler, motor_indices: list[int], data_name: str, values: np.ndarray):
    """
    Write data to multiple motors using group sync write.

    Args:
        port_handler: The port handler for the serial connection
        packet_handler: The packet handler for communication protocol
        motor_indices: List of motor IDs to write to
        data_name: Name of the data to write (must be in STS_SMS_SERIES_CONTROL_TABLE)
        values: Array of values to write to the motors

    Raises:
        KeyError: If data_name is not in the control table
        ConnectionError: If communication fails
        ValueError: If the number of values doesn't match the number of motor indices
    """
    if data_name not in STS_SMS_SERIES_CONTROL_TABLE:
        raise KeyError(f"Data name '{data_name}' not found in control table")

    if len(values) != len(motor_indices):
        raise ValueError(f"Number of values ({len(values)}) must match number of motor indices ({len(motor_indices)})")

    addr, message_bytes = STS_SMS_SERIES_CONTROL_TABLE[data_name]
    group = scs.GroupSyncWrite(port_handler, packet_handler, addr, message_bytes)

    for idx, value in zip(motor_indices, values, strict=True):
        if data_name in STS_SMS_SERIES_ENCODINGS_TABLE:
            sign_bit_index = STS_SMS_SERIES_ENCODINGS_TABLE[data_name]
            value = encode_sign_magnitude(value, sign_bit_index)
        data = convert_to_bytes(int(value), message_bytes)
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
    def __init__(self, port: str, calibration: dict[str, np.ndarray] | None = None, processing_freq: float = 1000.0):
        self.port = port
        self.motor_indices = [1, 2, 3, 4, 5, 6]
        self.processing_freq = processing_freq
        self.calibration = calibration
        self.port_handler = None
        self.packet_handler = None

    def connect(self):
        assert self.port_handler is None and self.packet_handler is None, "Already connected"
        self.port_handler = scs.PortHandler(self.port)
        self.packet_handler = scs.PacketHandler(PROTOCOL_VERSION)

        if not self.port_handler.openPort():
            raise OSError(f"Failed to open port '{self.port}'.")
        self.port_handler.setPacketTimeoutMillis(TIMEOUT_MS)

    def _read(self, data_name: str) -> np.ndarray:
        return read_from_motor(self.port_handler, self.packet_handler, self.motor_indices, data_name)

    def _write(self, data_name: str, values: np.ndarray):
        write_to_motor(self.port_handler, self.packet_handler, self.motor_indices, data_name, values)

    @property
    def position(self) -> np.ndarray:
        position = self._read("Present_Position")
        position = self.apply_calibration(position)
        return position

    @property
    def velocity(self) -> np.ndarray:
        velocity = self._read("Present_Velocity")
        velocity = self.apply_calibration(velocity)
        return velocity

    @property
    def torque_mode(self) -> bool:
        return self._read("Torque_Enable") == 1

    def set_torque_mode(self, enabled: bool):
        values = np.ones(len(self.motor_indices)) if enabled else np.zeros(len(self.motor_indices))
        self._write("Torque_Enable", values)
        self._write("Lock", values)

    def set_target_position(self, positions: np.ndarray):
        positions = self.revert_calibration(positions)
        self._write("Goal_Position", positions)

    def disconnect(self):
        assert self.port_handler is not None and self.packet_handler is not None, "Not connected"
        # Disable torque and lock to prevent motors degrading
        self.set_torque_mode(False)
        self.port_handler.closePort()
        self.port_handler = None
        self.packet_handler = None

    def apply_calibration(self, values: np.ndarray):
        if self.calibration is None:
            return values
        # convert raw values to 0-1 range
        return (values - self.calibration["mins"]) / (self.calibration["maxs"] - self.calibration["mins"])

    def revert_calibration(self, values: np.ndarray, clip: bool = True):
        if self.calibration is None:
            return values
        values = values * (self.calibration["maxs"] - self.calibration["mins"]) + self.calibration["mins"]
        if clip:
            values = np.clip(values, self.calibration["mins"], self.calibration["maxs"])
        return values

    def stats(self):
        result = {}
        for data_name in STS_SMS_SERIES_CONTROL_TABLE:
            values = self._read(data_name)
            result[data_name] = values
        return result
