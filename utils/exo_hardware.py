"""
exo_hardware.py — hip exoskeleton hardware interface (two TMotors, encoders only).

Encoder zeroing: ActuatorGroup.__init__ calls zero_encoder() on every actuator.
The wearer must be standing in the neutral reference pose when this object is
constructed, or every angle for the whole trial is offset.
"""

import time

from actuators import ActuatorGroup, TMotorV3
from controllers.base import Sensors


class HipExo:
    """Owns the two hip motors and returns Sensors snapshots."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

        self.can_id_L = int(cfg.get("can_id_L", 1))
        self.can_id_R = int(cfg.get("can_id_R", 2))
        self.motor_type = str(cfg.get("motor_type", "AK80-9"))
        self.can_channel = str(cfg.get("can_channel", "can0"))
        self.motor_read_fail_count = 0

        # ---- neutral pose: motor zeroing ----
        # ActuatorGroup zeroes both encoders as it enables the motors.
        # ActuatorGroup._load_can_drivers() reloads the CAN kernel modules and
        # brings the interface up, so no separate configure step is needed.
        _ = input("Stand still in the NEUTRAL pose, then press Enter to ZERO the motors: ")
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

    # ------------------------------------------------------------------
    def read(self) -> Sensors:
        """One synchronous snapshot of both motors, in raw motor frames. Sign
        correction belongs to the controller, not here, so the trial log keeps
        the unmodified values."""
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

        return Sensors(pos_L=pos_L, pos_R=pos_R, vel_L=vel_L, vel_R=vel_R)

    # ------------------------------------------------------------------
    def set_torques(self, tau_R: float, tau_L: float) -> tuple:
        """Command both hips, in Nm, already in each motor's own frame.

        No sign flipping happens here. The controller's torque_sign_r /
        torque_sign_l config keys are the single place mounting direction is
        handled.
        """
        tau_R, tau_L = float(tau_R), float(tau_L)
        try:
            self.mtr_comms.set_torque(self.can_id_L, tau_L)
            self.mtr_comms.set_torque(self.can_id_R, tau_R)
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

        if self.motor_read_fail_count:
            print(f"[HipExo] motor read failures during trial: {self.motor_read_fail_count}")
