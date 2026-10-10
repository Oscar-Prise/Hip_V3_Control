"""
dofc_hip — Delayed Output Feedback Control for the hip exoskeleton.

Model-free controller that needs no trained policy, only the motor encoders.

Control law:
    y(t)    = sin(theta_hip_r) - sin(theta_hip_l)      antisymmetric gait signal
    tau(t)  = kappa * y(t - delay)
    right   = -tau,   left = +tau

The delay is what makes this work: feeding back a delayed antisymmetric hip
signal entrains the wearer's gait instead of fighting it. kappa sets assistance
magnitude and delay sets the phase, so those are the two knobs to tune.

This is the hip-native version of Knee-exo-control/controllers/dofc.py, which
already used the hip sin() formulation.
"""

from collections import deque

import numpy as np

from .base import BaseController, CtrlResult, Sensors


class dofc_hip(BaseController):
    name = "dofc_hip"

    def __init__(self, config: dict):
        self.fs = float(config["fs"])
        self.dt = 1.0 / self.fs

        # Input low-pass on the hip angles.
        self.lpf_tau = float(config.get("lpf_tau", 0.05))
        if self.lpf_tau <= 0:
            self.alpha = 1.0
        else:
            self.alpha = max(0.0, min(1.0, self.dt / self.lpf_tau))

        self.delay = float(config["delay"])
        self.delay_samples = max(1, int(round(self.delay * self.fs)))
        self.y_buffer = deque([0.0] * self.delay_samples, maxlen=self.delay_samples)

        self.kappa = float(config["kappa"])   # Nm per unit of sin() difference

        # Encoder -> flexion-positive hip angle. Left motor is mounted reversed
        # on this rig; the config keys are the source of truth.
        self.hip_sign_r = float(config.get("hip_sign_r", 1.0))
        self.hip_sign_l = float(config.get("hip_sign_l", -1.0))
        self.hip_offset_deg_r = float(config.get("hip_offset_deg_r", 0.0))
        self.hip_offset_deg_l = float(config.get("hip_offset_deg_l", 0.0))

        self.torque_limit = float(config.get("hip_max_torque", 22.0))

        self.torque_sign_r = float(config.get("torque_sign_r", 1.0))
        self.torque_sign_l = float(config.get("torque_sign_l", -1.0))

        self.r_filt = 0.0
        self.l_filt = 0.0

    def start(self):
        self.r_filt = 0.0
        self.l_filt = 0.0
        self.y_buffer.clear()
        for _ in range(self.delay_samples):
            self.y_buffer.append(0.0)

    def step(self, s: Sensors) -> CtrlResult:
        r_raw = self.hip_sign_r * np.deg2rad(float(s.pos_R) - self.hip_offset_deg_r)
        l_raw = self.hip_sign_l * np.deg2rad(float(s.pos_L) - self.hip_offset_deg_l)

        self.r_filt = (1.0 - self.alpha) * self.r_filt + self.alpha * r_raw
        self.l_filt = (1.0 - self.alpha) * self.l_filt + self.alpha * l_raw

        y_now = float(np.sin(self.r_filt) - np.sin(self.l_filt))

        y_delayed = float(self.y_buffer[0])
        self.y_buffer.append(y_now)

        tau = self.kappa * y_delayed
        tau = float(np.clip(tau, -self.torque_limit, self.torque_limit))

        # torque_sign_* maps the controller frame onto each motor's own frame.
        # main_hip.py sends these through
        # unmodified, so mounting direction is handled in one place only.
        tau_r = self.torque_sign_r * (-tau)
        tau_l = self.torque_sign_l * (+tau)

        return CtrlResult(
            model_out_R=tau_r,
            model_out_L=tau_l,
            applied_R=tau_r,
            applied_L=tau_l,
            extra={
                "hip_angle_r": float(self.r_filt),
                "hip_angle_l": float(self.l_filt),
                "dofc_y": y_now,
                "dofc_y_delayed": y_delayed,
            },
        )

    def close(self):
        pass
