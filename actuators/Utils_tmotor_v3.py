import can
import time
import struct
import math
from dataclasses import dataclass
from typing import List

RAD2DEG = 180.0 / math.pi
DEG2RAD = math.pi / 180.0

MIT_MODE_ID = 8
ORIGIN_SET_ID = 5

# Values taken from
# AK Series Module Product Manual V3.0.1
# Page 39, column for AK80-9 motors
P_MIN = -12.56  # rad
P_MAX = 12.56  # rad
V_MIN = -65.0  # rad/s
V_MAX = 65.0  # rad/s
T_MIN = -18.0  # N-m
T_MAX = 18.0  # N-m
KP_MIN = 0
KP_MAX = 500.0
KD_MIN = 0
KD_MAX = 5.0

# Technically, this just makes int() since python doesn't use uint
# so it's NOT ROBUST TO NEGATIVE NUMBERS, but clamping to min/max
# means only positives come in so it's fine.
def _float_to_uint(x, x_min, x_max, bits):
    span = float(x_max - x_min)
    # int() makes the rounding unconventional but the resolution is
    # high enough that we're just not going to care?
    return int(((x - x_min) / span) * ((1 << bits) - 1))


def _clamp(x, x_min, x_max):
    return max(x_min, min(x_max, x))


def _read_cubemars_message(msg: can.Message) -> List[float]:
    pos_int = msg.data[0] << 8 | msg.data[1]
    vel_int = msg.data[2] << 8 | msg.data[3]
    current_int = msg.data[4] << 8 | msg.data[5]

    # Python doesn't seem to handle the sign bit properly when we just shift the bits
    # and OR them together, so using struct to force it to treat int as bytes
    pos = struct.unpack(">h", pos_int.to_bytes(2, "big"))[0] * 0.1 * DEG2RAD
    vel = struct.unpack(">h", vel_int.to_bytes(2, "big"))[0] * 10.0 * DEG2RAD
    current = struct.unpack(">h", current_int.to_bytes(2, "big"))[0] * 0.01
    temp = msg.data[6]
    errs = msg.data[7]
    return [pos, vel, current, temp, errs]


def _create_mit_message(can_id, pos, vel, kp, kd, torque) -> can.Message:
    pos_uint16 = _float_to_uint(_clamp(pos, P_MIN, P_MAX), P_MIN, P_MAX, 16)
    torque_uint12 = _float_to_uint(_clamp(torque, T_MIN, T_MAX), T_MIN, T_MAX, 12)
    vel_uint12 = _float_to_uint(_clamp(vel, V_MIN, V_MAX), V_MIN, V_MAX, 12)
    kp_uint12 = _float_to_uint(_clamp(kp, KP_MIN, KP_MAX), KP_MIN, KP_MAX, 12)
    kd_uint12 = _float_to_uint(_clamp(kd, KD_MIN, KD_MAX), KD_MIN, KD_MAX, 12)

    buffer = [
        kp_uint12 >> 4,  # KP High 8 bits
        ((kp_uint12 & 0xF) << 4) | (kd_uint12 >> 8),  # KP Low 4 bits, Kd High 4 bits
        kd_uint12 & 0xFF,  # Kd low 8 bits
        pos_uint16 >> 8,  # position high 8 bits
        pos_uint16 & 0xFF,  # position low 8 bits
        vel_uint12 >> 4,  # speed high 8 bits
        ((vel_uint12 & 0xF) << 4)
        | (torque_uint12 >> 8),  # speed low 4 bits torque high 4 bits
        torque_uint12 & 0xFF,  # torque low 8 bits
    ]

    arbitration_id = MIT_MODE_ID << 8 | can_id
    result = can.Message(
        arbitration_id=arbitration_id,
        data=buffer,
        is_extended_id=True,
        # is_rx=False,
        # check=True
    )
    # print(f"Composed message {result}")
    return result


def _create_set_origin_message(can_id: int) -> can.Message:
    buffer = [0] * 8
    arbitration_id = ORIGIN_SET_ID << 8 | can_id
    return can.Message(arbitration_id=arbitration_id, data=buffer, is_extended_id=True)

@dataclass
class MotorData:
    """Stores the most recent state of the current motor. This data is typically updated by a CAN Listener class.

    This contains the parameters relevant for control, (i.e. commanded and current position, velocity, torque, etc.), as well as the motor limits.
    The same data structure is used for all motors, but the limits are specific to the motor type. Additionally, some fields are not used for all motor types, and thus
    it is advised to use the getter methods for each motor instead of the dataclass directly.
    """
    motor_id: int
    motor_type: str
    current_position: float = 0.0
    current_velocity: float = 0.0
    current_torque: float = 0.0
    current_temperature: float = 0.0
    commanded_position: float = 0
    commanded_velocity: float = 0
    commanded_torque: float = 0
    kp: float = 0
    kd: float = 0
    torque_limits: tuple = (0,0)
    rated_torque_limits: tuple = (0,0)
    velocity_limits: tuple = (0,0)
    position_limits: tuple = (0,0)
    kp_limits: tuple = (0,0)
    kd_limits: tuple = (0,0)
    timestamp: float = -1
    last_command_time: float = -1
    initialized: bool = False
    responding: bool = False
    unique_hardware_id: int = -1
    running_torque: list = None
    rms_torque: float = 0
    rms_time_prev: float = 0
    motor_mode: float = 0
    internal_params = {}


class TMotorV3(can.Listener):
    def __init__(self, can_id: int, motor_type: str, invert: bool = False):
        self.can_id = can_id
        self.motor_type = motor_type
        self.invert = -1 if invert else 1
        self._bus = None
        self.data = MotorData(
            motor_id=self.can_id,
            motor_type=self.motor_type,
            current_position=0,
            current_velocity=0,
            current_torque=0,
            commanded_position=0,
            commanded_velocity=0,
            commanded_torque=0,
            kp=0,
            kd=0,
            timestamp=-1,
            running_torque=(),
            rms_torque=0,
            rms_time_prev=0,
        )

        self._connection_established = False
        self._reconnection_start_time = 0
        self.prev_command_time = 0

    def on_message_received(self, msg: can.Message) -> None:
        # print(f"Received message: {msg}") # Uncomment for debugging
        if msg.arbitration_id == (
            0x2900 + self.can_id
        ):  # + (1 << 32): # Message is from the motor, shift the check by
            pos, vel, current, temp, errs = _read_cubemars_message(msg)
            self.data.current_position = pos
            self.data.current_velocity = vel
            self.data.current_torque = current  # This is not necessarily correct, the torque != to current in all cases
            self.data.current_temperature = temp
            self.data.timestamp = time.perf_counter()
            # Ignoring errors for now

    def call_response_latency(self):
        return self.data.last_command_time - self.data.timestamp

    def set_torque(self, torque: float) -> None:
        torque = self.invert * torque
        self.data.commanded_torque = torque
        self.data.commanded_position = 0
        self.data.commanded_velocity = 0
        self.data.kp = 0
        self.data.kd = 0
        msg = _create_mit_message(self.can_id, 0, 0, 0, 0, self.data.commanded_torque)
        self._bus.send(msg)

    def set_position(
        self, position: float, kp: float, kd: float, degree: bool = False
    ) -> None:
        if degree:
            position *= DEG2RAD
            kp *= RAD2DEG
            kd *= RAD2DEG
        position = self.invert * position
        self.data.commanded_position = position
        self.data.kp = kp
        self.data.kd = kd
        self.data.commanded_torque = 0
        self.data.commanded_velocity = 0
        msg = _create_mit_message(
            self.can_id,
            self.data.commanded_position,
            0,
            self.data.kp,
            self.data.kd,
            self.data.commanded_torque,
        )
        self._bus.send(msg)

    def set_velocity(self, velocity: float, kd: float, degree: bool = False) -> None:
        if degree:
            velocity *= DEG2RAD
            kd *= RAD2DEG
        velocity = self.invert * velocity
        self.data.commanded_velocity = velocity
        self.data.kd = kd
        self.data.commanded_torque = 0
        self.data.commanded_position = 0
        msg = _create_mit_message(
            self.can_id,
            0,
            self.data.commanded_velocity,
            0,
            self.data.kd,
            self.data.commanded_torque,
        )
        self._bus.send(msg)

    def get_data(self) -> MotorData:
        return self.data

    def get_torque(self) -> float:
        return self.data.current_torque * self.invert

    def get_position(self, degrees: bool) -> float:
        if degrees:
            return self.data.current_position * self.invert * 180 / math.pi
        return self.data.current_position * self.invert

    def get_velocity(self, degrees: bool) -> float:
        if degrees:
            return self.data.current_velocity * self.invert * 180 / math.pi
        return self.data.current_velocity * self.invert

    def get_temperature(self) -> float:
        return self.data.current_temperature

    def zero_encoder(self):
        msg = _create_set_origin_message(self.can_id)
        self._bus.send(msg)

    def _enable(self) -> None:
        zero_trq_msg = _create_mit_message(self.can_id, 0, 0, 0, 0, 0)
        self._bus.send(zero_trq_msg)

    def _disable(self) -> None:
        zero_trq_msg = _create_mit_message(self.can_id, 0, 0, 0, 0, 0)
        self._bus.send(zero_trq_msg)

    def _set_zero_torque(self):
        self.data.commanded_torque = 0.0
        self.data.commanded_position = 0.0
        self.data.commanded_velocity = 0.0
        self.data.kp = 0.0
        self.data.kd = 0.0
        msg = _create_mit_message(self.can_id, 0, 0, 0, 0, 0)
        self._bus.send(msg)


if __name__ == "__main__":
    msg = _create_mit_message(1, 6, 0, 2, 2, 0)
