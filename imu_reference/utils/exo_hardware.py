"""
exo_hardware.py — hip exoskeleton hardware interface.

Encoder zeroing: ActuatorGroup.__init__ calls zero_encoder() on every actuator.
The wearer must be standing in the neutral reference pose when this object is
constructed, or every angle for the whole trial is offset.
"""

import time

import numpy as np

from actuators import ActuatorGroup, TMotorV3
from controllers.base import Sensors
from utils.Header_MuseIMU import DEFAULTS as MUSE_DEFAULTS

IMU_ZEROS = np.zeros(6, dtype=np.float32)


class HipExo:
    """Owns the two hip motors and optional Muse IMUs and returns Sensors snapshots."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.can_id_L = int(cfg.get("can_id_L", 1))
        self.can_id_R = int(cfg.get("can_id_R", 2))
        self.motor_type = str(cfg.get("motor_type", "AK80-9"))
        self.can_channel = str(cfg.get("can_channel", "can0"))
        self.imu_type = str(cfg.get("imu_type", "muse")).lower()
        if self.imu_type not in ("none", "muse"):
            raise ValueError("imu_type must be muse or none")
        self.imus = None
        self.motor_read_fail_count = 0
        self._muse_armed = False
        self.muse_command_hold_s = float(cfg.get(
            "muse_command_hold_timeout_s", MUSE_DEFAULTS["muse_command_hold_timeout_s"]))
        if not np.isfinite(self.muse_command_hold_s) or self.muse_command_hold_s < 0:
            raise ValueError("muse_command_hold_timeout_s must be finite and non-negative")
        self._last_valid_torque = (0.0, 0.0)  # right, left
        self._muse_invalid_since = None
        self.muse_command_hold_active = False

        # Validate/connect sensors before enabling motors.
        if self.imu_type == "muse":
            from utils.Header_MuseIMU import MuseIMUs
            self.imus = MuseIMUs(cfg)
            try:
                self.imus.start()
            except BaseException:
                self.imus.close()
                raise
        else:
            print("[HipExo] IMU acquisition disabled; using motor encoders only.")

        # ---- neutral pose: IMU calibration + motor zeroing ----
        # Both are measured from the same standing pose, so one Enter runs them
        # back to back: the Muse neutral angles first (the wearer stands still
        # for muse_calibration_duration_s), then ActuatorGroup zeroes both
        # encoders as it enables the motors. ActuatorGroup._load_can_drivers()
        # reloads the CAN kernel modules and brings the interface up, so no
        # separate configure step is needed.
        calibrate_imus = self.imu_type == "muse" and self.imus.auto_calibrate
        try:
            _ = input("Stand still in the NEUTRAL pose, then press Enter to " +
                      ("calibrate the IMUs and " if calibrate_imus else "") +
                      "ZERO the motors: ")
            if self.imu_type == "muse":
                self.imus.calibrate_neutral()
            print(f"[HipExo] Keep still: zeroing motors L=CAN {self.can_id_L}, "
                  f"R=CAN {self.can_id_R} ({self.motor_type}) on {self.can_channel}...")
            self.mtr_comms = ActuatorGroup(
                [TMotorV3(self.can_id_L, self.motor_type),
                 TMotorV3(self.can_id_R, self.motor_type)],
                can_args={"bustype": "socketcan", "channel": self.can_channel},
                enable_on_startup=True,
                exit_manually=True,
            )
            print("[HipExo] Neutral pose recorded; the wearer can move.")
        except BaseException:
            if self.imu_type == "muse":
                self.imus.close()
            raise

    # ------------------------------------------------------------------
    def read(self) -> Sensors:
        """One synchronous snapshot of motors and IMUs, in raw motor/sensor
        frames. Sign correction belongs to the controller, not here, so the
        trial log keeps the unmodified values."""
        try:
            pos_L = float(self.mtr_comms.get_position(self.can_id_L, degrees=True))
            vel_L = float(self.mtr_comms.get_velocity(self.can_id_L, degrees=True))
            pos_R = float(self.mtr_comms.get_position(self.can_id_R, degrees=True))
            vel_R = float(self.mtr_comms.get_velocity(self.can_id_R, degrees=True))
        except Exception:
            # ActuatorGroup already swallows CanOperationError internally; this
            # is a backstop so a motor hiccup cannot abort the loop.
            self.motor_read_fail_count += 1
            pos_L = vel_L = pos_R = vel_R = 0.0

        if self.imu_type == "muse":
            if not self._muse_armed:
                self.imus.arm()
                self._muse_armed = True
            samples, ages, valid = self.imus.snapshot()
            left, right = samples["left"], samples["right"]
            return Sensors(
                imu_thigh_L=np.asarray(left.imu, dtype=np.float32) if left else IMU_ZEROS.copy(),
                imu_thigh_R=np.asarray(right.imu, dtype=np.float32) if right else IMU_ZEROS.copy(),
                pos_L=pos_L, pos_R=pos_R, vel_L=vel_L, vel_R=vel_R,
                thigh_angle_L=left.thigh_angle if left else float("nan"),
                thigh_angle_R=right.thigh_angle if right else float("nan"),
                quaternion_L=left.quaternion if left else (float("nan"),) * 4,
                quaternion_R=right.quaternion if right else (float("nan"),) * 4,
                imu_age_L=ages["left"], imu_age_R=ages["right"],
                imu_control_valid=valid,
            )

        return Sensors(
            imu_thigh_L=IMU_ZEROS.copy(),
            imu_thigh_R=IMU_ZEROS.copy(),
            pos_L=pos_L, pos_R=pos_R, vel_L=vel_L, vel_R=vel_R,
        )

    # ------------------------------------------------------------------
    def set_torques(self, tau_R: float, tau_L: float) -> tuple:
        """Command both hips, in Nm, already in each motor's own frame.

        No sign flipping happens here. The controller's torque_sign_r /
        torque_sign_l config keys are the single place mounting direction is
        handled, so there is exactly one thing to change if a motor turns out
        to be mounted backwards.
        """
        # Recheck at actuation, since inference/loop work may have taken time
        # after read(). Return guarded commands so the log records the request
        # actually sent to CAN (not a pre-inhibit value).
        tau_R, tau_L = float(tau_R), float(tau_L)
        if self.imu_type == "muse":
            now = time.monotonic()
            valid = self._muse_armed and self.imus.snapshot()[2]
            if valid:
                self._muse_invalid_since = None
                self.muse_command_hold_active = False
                self._last_valid_torque = (tau_R, tau_L)
            else:
                if self._muse_invalid_since is None:
                    self._muse_invalid_since = now
                elapsed = now - self._muse_invalid_since
                if elapsed <= self.muse_command_hold_s:
                    tau_R, tau_L = self._last_valid_torque
                    self.muse_command_hold_active = True
                else:
                    tau_R = tau_L = 0.0
                    self.muse_command_hold_active = False
        try:
            self.mtr_comms.set_torque(self.can_id_L, float(tau_L))
            self.mtr_comms.set_torque(self.can_id_R, float(tau_R))
        except Exception as e:
            print(f"[HipExo] set_torque failed: {e}")
        return tau_R, tau_L

    def zero_torques(self) -> None:
        try:
            self.mtr_comms.set_torque(self.can_id_L, 0.0)
            self.mtr_comms.set_torque(self.can_id_R, 0.0)
        except Exception as e:
            print(f"[HipExo] zero_torques failed: {e}")

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Zero torque, disable the motors, and drop the CAN bus. Safe to call
        more than once — shutdown paths can overlap."""
        try:
            self.zero_torques()
            time.sleep(0.05)
        except Exception:
            pass

        try:
            self.mtr_comms.disable_actuators()
        except Exception as e:
            print(f"[HipExo] disable_actuators failed: {e}")

        try:
            self.mtr_comms.notifier.stop()
        except Exception as e:
            print(f"[HipExo] notifier.stop failed: {e}")

        try:
            self.mtr_comms.bus.shutdown()
        except Exception as e:
            print(f"[HipExo] bus.shutdown failed: {e}")

        if self.imu_type == "muse":
            self.imus.close()

        if self.motor_read_fail_count:
            print(f"[HipExo] motor read failures during trial: {self.motor_read_fail_count}")
