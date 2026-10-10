# controllers/base.py
#
# Hip-exo sensor and result types.
#
# Sensors carries BOTH sides because every hip controller here is bilateral:
# the student TCN takes both hip angles and both thigh angular velocities in a
# single input tensor and emits both sides' impedance commands at once. (The
# knee stack ran the same single-leg model twice as a batch of 2; the hip model
# is genuinely bilateral and cannot be split per side.)
#
# Two Muse IMUs (imu_type: muse) supply each thigh's IMU vector, quaternion and
# selected-axis angle; imu_type: none leaves them zero/NaN (encoder-only).
#
# IMU 6-vector layout: [Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z],
#   accel in m/s^2, gyro in deg/s, Muse body frame.
#
# Motor values are RAW, in each motor's own frame: degrees and degrees/second.
# Sign correction lives in the controllers, driven by config keys, so the trial
# log keeps the unmodified hardware values.
from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class Sensors:
    imu_thigh_L: np.ndarray   # left thigh  [ax, ay, az, gx, gy, gz]
    imu_thigh_R: np.ndarray   # right thigh
    pos_L: float              # left hip encoder angle  (deg, motor frame)
    pos_R: float              # right hip encoder angle (deg, motor frame)
    vel_L: float              # left hip encoder velocity  (deg/s, motor frame)
    vel_R: float              # right hip encoder velocity (deg/s, motor frame)
    thigh_angle_L: float = float("nan")  # Muse selected-axis angle, rad, neutral offset removed
    thigh_angle_R: float = float("nan")
    quaternion_L: tuple = (float("nan"),) * 4  # Muse w,x,y,z
    quaternion_R: tuple = (float("nan"),) * 4
    imu_age_L: float = float("nan")       # seconds since host reception
    imu_age_R: float = float("nan")
    imu_control_valid: bool = True        # Muse freshness + latched fault guard


@dataclass(slots=True)
class CtrlResult:
    model_out_R: float
    model_out_L: float
    applied_R: float
    applied_L: float
    extra: dict


class BaseController:
    name: str = "base"

    def start(self): pass

    def step(self, s: Sensors) -> CtrlResult: raise NotImplementedError

    def close(self): pass


class RollingWindow:
    """Fixed-length FIFO over the last axis. push_last() shifts left and writes
    the newest sample into [..., -1], returning the whole (..., T) buffer."""

    def __init__(self, shape, dtype=np.float32):
        shape = tuple(int(x) for x in shape)
        if len(shape) < 2:
            raise ValueError("RollingWindow shape must be at least 2D: (..., T)")
        self.buf = np.zeros(shape, dtype=dtype)
        self._last_slice_shape = self.buf.shape[:-1]

    def push_last(self, arr_last):
        arr_last = np.asarray(arr_last, dtype=self.buf.dtype)
        arr_last = arr_last.reshape(self._last_slice_shape)

        self.buf[..., :-1] = self.buf[..., 1:]
        self.buf[..., -1] = arr_last
        return self.buf
