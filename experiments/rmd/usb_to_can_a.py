import serial
import threading
import queue
import argparse

CANUSB_FRAME_STANDARD = 0x01
CANUSB_FRAME_EXTENDED = 0x02
CANUSB_MODE_NORMAL = 0x00

CMD_START_BYTE = 0xaa
CMD_ID = 0x55

CAN_SPEED_MAP = {
    1000000: 0x01,
    800000:  0x02,
    500000:  0x03,
    400000:  0x04,
    250000:  0x05,
    200000:  0x06,
    125000:  0x07,
    100000:  0x08,
    50000:   0x09,
    20000:   0x0A,
    10000:   0x0B,
    5000:    0x0C,
}

class CANUSBAdapter:
    def __init__(self, port, baudrate=2000000, can_speed=0x03):  # 500k default
        self.port = port
        self.baudrate = baudrate
        self.can_speed = can_speed
        self.ser = serial.Serial(port, baudrate, timeout=0.1)
        self.running = True
        self.tx_queue = queue.Queue()
        self.rx_thread = threading.Thread(target=self.listen)
        self.tx_thread = threading.Thread(target=self.transmit)
        self.rx_thread.start()
        self.tx_thread.start()
        self.lock = threading.Lock()
        self.setup_device()

    def setup_device(self):
        cmd = [0xaa, 0x55, 0x12, self.can_speed, CANUSB_FRAME_STANDARD] + [0]*9 + [CANUSB_MODE_NORMAL, 0x01, 0, 0, 0, 0]
        checksum = sum(cmd[2:]) & 0xFF
        cmd.append(checksum)
        with self.lock:
            self.ser.write(bytearray(cmd))

    def send_can_frame(self, can_id, data):
        if not (0 <= len(data) <= 8):
            raise ValueError("CAN data must be 0-8 bytes")
        id_lsb = can_id & 0xFF
        id_msb = (can_id >> 8) & 0xFF
        header = 0xC0 | (len(data) & 0x0F)
        frame = [0xAA, header, id_lsb, id_msb] + list(data) + [0x55]
        self.tx_queue.put(bytearray(frame))
        # print(f"Sending CAN ID: {can_id:04X}, Data: {[hex(b) for b in data]}")

    def transmit(self):
        while self.running or not self.tx_queue.empty():
            try:
                frame = self.tx_queue.get(timeout=0.5)
                with self.lock:
                    self.ser.write(frame)
            except queue.Empty:
                continue

    def listen(self):
        buffer = bytearray()
        while self.running:
            data = self.ser.read(64)
            buffer.extend(data)
            while len(buffer) >= 5:
                if buffer[0] != 0xAA:
                    buffer.pop(0)
                    continue
                if (buffer[1] >> 4) == 0xC:
                    dlc = buffer[1] & 0x0F
                    expected_len = 5 + dlc
                    if len(buffer) < expected_len:
                        break
                    frame = buffer[:expected_len]
                    buffer = buffer[expected_len:]
                    self.process_frame(frame)
                else:
                    # Handle or skip unknown frame
                    buffer.pop(0)

    def process_frame(self, frame):
        can_id = frame[3] << 8 | frame[2]
        data = frame[4:-1]
        return can_id, data

    def stop(self):
        self.running = False
        self.rx_thread.join()
        self.tx_thread.join()
        self.ser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CANUSB Adapter Utility")
    parser.add_argument(
        "-d", "--device", type=str, default="/dev/ttyUSB0",
        help="Serial device path (default: /dev/ttyUSB0)"
    )
    parser.add_argument(
        "-s", "--speed", type=int, choices=CAN_SPEED_MAP.keys(), default=1000000,
        help="CAN speed in bit/s (default: 1000000)"
    )
    args = parser.parse_args()

    can_speed = CAN_SPEED_MAP[args.speed]
    adapter = CANUSBAdapter(args.device, can_speed=can_speed)
    print(f"Using device: {args.device}, CAN speed: {args.speed} bit/s")
    try:
        while True:
            user_input = input("Enter CAN ID and data (e.g. 123 112233): ")
            if not user_input:
                continue
            parts = user_input.strip().split()
            if len(parts) != 2:
                continue
            can_id = int(parts[0], 16)
            data = bytes.fromhex(parts[1])
            adapter.send_can_frame(can_id, data)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        adapter.stop()
