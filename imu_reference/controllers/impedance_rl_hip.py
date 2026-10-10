"""
impedance_rl_hip — bilateral hip impedance controller driven by the distilled
TCN student policy.

This is the hip counterpart of Knee-exo-control/controllers/impedance_rl.py.
The impedance law itself is transcribed from the training-side controller
(ref/hip_impedance_controllerv2.py) so that what runs on hardware matches what
the policy was trained against.

--------------------------------------------------------------------------
MODEL I/O  (see ref/cfg_tcn_hip_lg_hipang_thighvel_bilateral.yaml)
--------------------------------------------------------------------------
Input  : (1, C, T)   ONE sample, both legs in the same tensor. C = input_size,
                     which selects the sensor modality (see INPUT MODALITY):

    input_size 4 — selected angles + bilateral thigh gyros
        ch0  thigh_angle_r    (rad, selected Muse axis; hip_angle_r in legacy hip mode)
        ch1  thigh_angle_l    (rad, selected Muse axis; hip_angle_l in legacy hip mode)
        ch2  thigh_angvel_r   (rad/s, right convention)
        ch3  thigh_angvel_l   (rad/s, sign-corrected to right convention)

    input_size 2 — bilateral thigh gyros only
        ch0  thigh_angvel_r   (rad/s, right convention)
        ch1  thigh_angvel_l   (rad/s, sign-corrected to right convention)

Output : (4,)
    y[0] K_r     stiffness command, [-1, 1] -> _map_K -> [K_min, K_max] Nm/rad
    y[1] dq_r    gait-phase command, [-1, 1] -> Gaussian flex/trans/ext weights
    y[2] K_l
    y[3] dq_l

This is the structural break from the knee stack. The knee ran a single-leg
model twice as a batch of 2; here one bilateral model sees both legs at once,
so the TensorRT worker runs batch size 1 (see trt_worker.py).

--------------------------------------------------------------------------
INPUT MODALITY
--------------------------------------------------------------------------
`input_size` in the YAML picks which student policy is running: 4 for the
angle + thigh-gyro model, 2 for the gyro-only model. `model_angle_source` selects
"hip" (encoders) or "thigh" (Muse, selected axis). It changes exactly one
thing — which signals are packed into the sensor window. The hip encoder is
still read, sign-corrected, filtered and fed to the impedance law in BOTH
modes; a gyro-only policy is blind to the hip angle, the control law is not.

The pick is not taken on trust. Each engine ships a meta.json (sensor_channels,
sensor_channel_names, sensor_mean, sensor_std) and the constructor refuses to
start unless it agrees with `input_size` channel for channel, in order. That is
the guard against the failure this stack has already been bitten by once: a
config pointed at the wrong artifact, running a real policy on the wrong
signals, producing smooth gait-locked output the whole time. Switching modality
therefore means switching trt_engine_path AND input_size together.

--------------------------------------------------------------------------
SIGN CONVENTIONS
--------------------------------------------------------------------------
Everything fed to the model is expressed in the RIGHT-leg convention, matching
the simulator: positive hip angle = FLEXION, positive torque = FLEXION assist.

The defaults follow the hip rig used by continual-online-personalization, where
the RIGHT motor is mounted reversed. Its Controller.py does exactly two things
about this:
    mtr_pos_R *= -1; mtr_vel_R *= -1     # mirror right encoder onto left's frame
    set_torque(CAN_id_R, -motor_cmd_R)   # "motor is mounted in reverse direction"
so the shared convention is the LEFT motor's native frame, and the Hip_v3_CTRL
README pins that down as flexion(+) / extension(-) — the same sense the model
uses. Hence hip_sign_l = +1, hip_sign_r = -1, torque_sign_l = +1,
torque_sign_r = -1.

Every multiplier is a config key, not a literal, so a mounting surprise is a
YAML edit rather than a code change:
    hip_sign_r / hip_sign_l              encoder angle + velocity
    muse_axis_sign_r / muse_axis_sign_l  thigh angle (applied in Header_MuseIMU)
    muse_gyro_sign_r / muse_gyro_sign_l  thigh gyro
    torque_sign_r / torque_sign_l        controller frame -> motor frame

These are the ONLY place mounting direction is handled. main_hip.py sends the
commands through unmodified, so there is no second negation to reason about.

--------------------------------------------------------------------------
IMPEDANCE LAW (from ref/hip_impedance_controllerv2.py)
--------------------------------------------------------------------------
    w_flex, w_trans, w_ext = gaussian weights over the gait command
    tau_flex  = -K * flex_K_mult * (hip - q_flex)
    tau_ext   = -K *               (hip - q_ext)
    tau       = w_flex*tau_flex + w_trans*0 + w_ext*tau_ext
    out       = hip_max_torque * tanh(tau / hip_max_torque)

Two deliberate differences from the knee law, both inherited from the hip
training controller:
  * NO velocity damping term. The knee law added -B*knee_u; the hip controller
    the policy trained against has no B at all, so adding one here would be a
    train/deploy mismatch.
  * flex_K_mult = 1.0. The knee used 0.3 to guard against buckling; the hip has
    no equivalent buckling risk and the training controller matched flexion and
    extension stiffness.
"""

import json
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty, Full

import numpy as np

from .base import BaseController, CtrlResult, RollingWindow, Sensors
from .trt_worker import TRTWorker
from utils.Header_MuseIMU import DEFAULTS as MUSE_DEFAULTS

# The engine loads and warms up in ~0.3 s on the Orin Nano.
TRT_READY_TIMEOUT_S = 20.0


# Every sensor this controller can put in front of a policy, in the order the
# raw sample is assembled each step. CHANNEL_LAYOUTS selects from it.
CANONICAL_CHANNELS = ("hip_angle_r", "hip_angle_l",
                      "thigh_angvel_r", "thigh_angvel_l",
                      "thigh_angle_r", "thigh_angle_l")

# input_size -> the model's channel order, matching the distillation dataset.
# Validated against meta.json["sensor_channel_names"] at construction.
CHANNEL_LAYOUTS = {
    4: ("hip_angle_r", "hip_angle_l", "thigh_angvel_r", "thigh_angvel_l"),
    2: ("thigh_angvel_r", "thigh_angvel_l"),
}


class impedance_rl_hip(BaseController):
    name = "impedance_rl_hip"

    def __init__(self, config: dict):
        # ---------------- model / timing ----------------
        self.engine_path = config["trt_engine_path"]
        self.T = int(config["frame_length"])
        self.fs = int(config["fs"])
        self.dt = 1.0 / self.fs

        self.input_size = int(config["input_size"])
        self.output_size = int(config["output_size"])
        if str(config.get("imu_type", "muse")).lower() != "muse":
            raise ValueError("impedance_rl_hip needs imu_type: muse (thigh IMUs feed the policy)")
        self.model_angle_source = str(config.get("model_angle_source", "thigh")).lower()
        self.latch_imu_fault = bool(config.get(
            "muse_latch_stale_fault", MUSE_DEFAULTS["muse_latch_stale_fault"]))
        if self.model_angle_source not in ("hip", "thigh"):
            raise ValueError("model_angle_source must be hip or thigh")
        self._imu_fault = False

        if self.input_size not in CHANNEL_LAYOUTS:
            raise ValueError(
                f"impedance_rl_hip supports input_size "
                f"{sorted(CHANNEL_LAYOUTS)}, got {self.input_size}. "
                f"4 = hip angles + thigh gyros, 2 = thigh gyros only."
            )
        self.channel_names = CHANNEL_LAYOUTS[self.input_size]
        if self.input_size == 4 and self.model_angle_source == "thigh":
            self.channel_names = ("thigh_angle_r", "thigh_angle_l",
                                  "thigh_angvel_r", "thigh_angvel_l")
        # model channel -> index into the canonical raw sample built every step.
        self._channel_src = np.array(
            [CANONICAL_CHANNELS.index(c) for c in self.channel_names],
            dtype=np.intp,
        )

        if self.output_size != 4:
            raise ValueError(
                f"impedance_rl_hip expects output_size=4 (K_r, dq_r, K_l, dq_l), "
                f"got {self.output_size}"
            )

        # ONE sample, both legs. Not a batch of 2 like the knee stack.
        self.in_shape = (1, self.input_size, self.T)
        self.out_shape = (self.output_size,)

        # ---------------- sensor interpretation ----------------
        # Right motor is mounted reversed on this rig — see the docstring.
        self.hip_sign_r = float(config.get("hip_sign_r", -1.0))
        self.hip_sign_l = float(config.get("hip_sign_l", 1.0))
        # Thigh velocity = Muse gyro (deg/s) on muse_gyro_axis times each side's
        # sign. The Euler angle and body-frame gyro axes need not match: a
        # side-mounted unit can expose Y-pitch while flexion rate is on body Z.
        muse_axis = str(config.get("muse_axis", "x")).lower()
        muse_gyro_axis = str(config.get("muse_gyro_axis", muse_axis)).lower()
        if muse_gyro_axis not in ("x", "y", "z"):
            raise ValueError("muse_gyro_axis must be x, y or z")
        fallback_sign = float(config.get("muse_axis_sign", 1.0))
        self.thigh_vel_sign_r = float(config.get(
            "muse_gyro_sign_r", config.get("muse_axis_sign_r", fallback_sign)))
        self.thigh_vel_sign_l = float(config.get(
            "muse_gyro_sign_l", config.get("muse_axis_sign_l", fallback_sign)))
        if any(sign not in (-1.0, 1.0) for sign in (self.thigh_vel_sign_r, self.thigh_vel_sign_l)):
            raise ValueError("Muse gyro signs must be 1.0 or -1.0")
        # IMU 6-vector is [Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z].
        self.thigh_gyro_idx = {"x": 3, "y": 4, "z": 5}[muse_gyro_axis]
        self.torque_sign_r = float(config.get("torque_sign_r", -1.0))
        self.torque_sign_l = float(config.get("torque_sign_l", 1.0))

        # Encoder zero offset (deg), applied before deg2rad and before sign
        # correction. Non-zero when the motor is homed somewhere other than the
        # neutral standing hip angle.
        self.hip_offset_deg_r = float(config.get("hip_offset_deg_r", 0.0))
        self.hip_offset_deg_l = float(config.get("hip_offset_deg_l", 0.0))

        # ---------------- input normalization ----------------
        # The student was trained on z-scored windows: TCN_distillation.py applies
        # meta.json's per-channel sensor_mean/sensor_std in ShardIterableDataset
        # before any window reaches the model. Those stats live in meta.json, NOT
        # in the TCN config -- reading the config alone and defaulting to identity
        # is what silently ran this policy on unnormalized inputs (angles ~3.5x too
        # small and off-centre, velocities ~2x too large).
        #
        # No identity fallback on purpose: a missing meta file must fail loudly,
        # because the failure is otherwise invisible. The policy still produces
        # smooth, gait-locked K/dq output when fed raw physical units -- it is just
        # a different function than the one that was trained.
        meta_path = Path(config.get("norm_meta_path")
                         or Path(self.engine_path).with_suffix(".meta.json"))
        with open(meta_path) as f:
            norm_meta = json.load(f)

        # Modality cross-check: the engine's own sidecar decides whether
        # input_size is telling the truth. A stale input_size against a swapped
        # engine is otherwise invisible until the torque feels wrong.
        meta_channels = norm_meta.get("sensor_channels")
        if meta_channels is not None and int(meta_channels) != self.input_size:
            raise ValueError(
                f"{meta_path}: this engine takes {int(meta_channels)} input "
                f"channels, config says input_size={self.input_size}. Switch "
                f"trt_engine_path and input_size together."
            )
        meta_names = norm_meta.get("sensor_channel_names")
        if self.input_size == 4 and self.model_angle_source == "thigh" and meta_names is None:
            raise ValueError("Thigh-angle policy metadata must declare sensor_channel_names; "
                             "do not reuse hip-angle policy weights or normalization")
        if meta_names is not None and tuple(meta_names) != self.channel_names:
            raise ValueError(
                f"{meta_path}: engine channel order {tuple(meta_names)} does "
                f"not match the layout this controller builds for input_size="
                f"{self.input_size}: {self.channel_names}"
            )
        if int(norm_meta.get("sensor_window", self.T)) != self.T:
            raise ValueError(f"{meta_path}: sensor_window does not match frame_length")
        if not Path(self.engine_path).is_file():
            raise FileNotFoundError(f"Policy engine not found: {self.engine_path}")

        self.sensor_mean = np.asarray(norm_meta["sensor_mean"], dtype=np.float32)
        self.sensor_std = np.asarray(norm_meta["sensor_std"], dtype=np.float32)
        if self.sensor_mean.shape != (self.input_size,) or self.sensor_std.shape != (self.input_size,):
            raise ValueError(
                f"{meta_path}: sensor_mean/sensor_std must each have "
                f"{self.input_size} entries, got {self.sensor_mean.shape} / "
                f"{self.sensor_std.shape}"
            )
        if not np.all(self.sensor_std > 1e-6):
            raise ValueError(
                f"{meta_path}: sensor_std must be strictly positive, got {self.sensor_std}"
            )
        if not (np.all(np.isfinite(self.sensor_mean)) and np.all(np.isfinite(self.sensor_std))):
            raise ValueError(f"{meta_path}: normalization must be finite")

        # ---------------- impedance params (ref/cfg_1.yaml) ----------------
        self.K_max = float(config.get("K_max", 60.0))          # Nm/rad
        self.K_min = float(config.get("K_min", 0.0))           # Nm/rad
        self.hip_max_torque = float(config.get("hip_max_torque", 22.0))  # Nm
        self.gait_sigma = float(config.get("gait_sigma", 0.3))
        self.rate_max = float(config.get("rate_max", 20.0))    # gait cmd slew, 1/s

        # Matched flexion/extension stiffness — see module docstring.
        self.flex_K_mult = float(config.get("flex_K_mult", 1.0))

        # Reference angles (deg -> rad). Positive = flexion.
        # q_ref_min is the FLEXION target, q_ref_max the EXTENSION target.
        self.q_flex = np.deg2rad(float(config.get("q_ref_min", 40.0)))
        self.q_ext = np.deg2rad(float(config.get("q_ref_max", -20.0)))

        # ---------------- filter time constants ----------------
        self.hip_filter_tau = float(config.get("hip_filter_tau", 0.05))
        self.impedance_filter_tau = float(config.get("impedance_filter_tau", 0.05))
        self.gait_filter_tau = float(config.get("gait_filter_tau", 0.1))
        self.output_filter_tau = float(config.get("output_filter_tau", 0.05))

        # ---------------- body-mass scaling ----------------
        # The hip impedance law emits absolute Nm (hip_max_torque is Nm, not
        # Nm/kg), so this defaults to a no-op ratio of 1.0. Set BOTH keys to
        # scale a policy trained on one body mass onto a different subject.
        subject_mass = config.get("subject_mass", None)
        model_mass = config.get("model_mass", None)
        if subject_mass is not None and model_mass is not None:
            self.mass_ratio = float(subject_mass) / float(model_mass)
        else:
            self.mass_ratio = 1.0

        # ---------------- rolling window / model state ----------------
        self.x = RollingWindow((self.input_size, self.T))
        # Neutral start: K command at -1 (K_min), gait command at 0 (transition,
        # which produces zero torque) so nothing kicks before the first inference.
        self.last_out = np.array([-1.0, 0.0, -1.0, 0.0], dtype=np.float32)

        # ---------------- TRT worker ----------------
        self.in_q = mp.Queue(maxsize=1)
        self.out_q = mp.Queue(maxsize=1)
        self.worker = TRTWorker(self.in_q, self.out_q, self.engine_path,
                                self.in_shape, self.out_shape)
        self.worker.daemon = True

        # ---------------- filtered signal state ----------------
        self.hip_r_filt = 0.0
        self.hip_l_filt = 0.0
        self.hip_r_u_filt = 0.0          # encoder-derived hip velocity (diagnostic)
        self.hip_l_u_filt = 0.0

        self.K_r_raw = 0.0
        self.K_l_raw = 0.0
        self.K_r_filt = 0.0
        self.K_l_filt = 0.0

        self.gait_r_filt = 0.0
        self.gait_l_filt = 0.0
        self.gait_r_raw_prev = 0.0
        self.gait_l_raw_prev = 0.0

        self.tau_r_filt = 0.0
        self.tau_l_filt = 0.0

    # ------------------------------------------------------------------
    # small helpers
    # ------------------------------------------------------------------
    def _alpha(self, tau: float) -> float:
        if tau <= 0.0:
            return 1.0
        return min(max(self.dt / tau, 0.0), 1.0)

    def _lpf(self, x_prev: float, x_raw: float, tau: float) -> float:
        a = self._alpha(tau)
        return float(x_prev + a * (x_raw - x_prev))

    def _rate_limit(self, current: float, prev: float, rate_max: float) -> float:
        max_step = rate_max * self.dt
        return float(prev + np.clip(current - prev, -max_step, +max_step))

    # ------------------------------------------------------------------
    # impedance law — transcribed from ref/hip_impedance_controllerv2.py
    # ------------------------------------------------------------------
    def _map_K(self, u: float) -> float:
        """[-1, 1] policy output -> [K_min, K_max] Nm/rad."""
        u = float(np.clip(u, -1.0, 1.0))
        return float(self.K_min + (self.K_max - self.K_min) * (u + 1.0) / 2.0)

    def _gait_weights(self, g: float):
        """Gaussian soft-assignment of the gait command to
        (flexion, transition, extension) modes. Centers and the normalization
        match the training controller exactly."""
        g = float(np.clip(g, -1.0, 1.0))
        c = np.array([-0.85, 0.0, 0.85], dtype=np.float32)
        s = max(self.gait_sigma, 1e-6)
        w = np.exp(-0.5 * ((g - c) / s) ** 2)
        w = w / np.sum(w)
        return float(w[0]), float(w[1]), float(w[2])

    def _torque_one_leg(self, hip_f: float, K_f: float, gait_f: float):
        """Quasi-stiffness about a flexion/extension reference angle.
        No damping term — see module docstring."""
        w_flex, w_trans, w_ext = self._gait_weights(gait_f)
        dq_flex = float(hip_f - self.q_flex)
        dq_ext = float(hip_f - self.q_ext)

        tau_trans = 0.0
        tau_flex = -K_f * self.flex_K_mult * dq_flex
        tau_ext = -K_f * dq_ext
        tau = w_trans * tau_trans + w_flex * tau_flex + w_ext * tau_ext
        out = self.hip_max_torque * np.tanh(tau / self.hip_max_torque)

        diag = dict(
            K_f=K_f, gait_f=gait_f, dq_flex=dq_flex, dq_ext=dq_ext,
            w_flex=w_flex, w_trans=w_trans, w_ext=w_ext,
            tau_flex=tau_flex, tau_ext=tau_ext, tau_raw=tau, tau_out=float(out),
        )
        return float(out), diag

    # ------------------------------------------------------------------
    # inference plumbing
    # ------------------------------------------------------------------
    def _get_latest_inference(self):
        latest = None
        try:
            while True:
                latest = self.out_q.get_nowait()
        except Empty:
            pass
        return latest

    def _try_put_latest(self, payload) -> None:
        try:
            self.in_q.put_nowait(payload)
        except Full:
            pass

    # ------------------------------------------------------------------
    # main control step
    # ------------------------------------------------------------------
    def step(self, s: Sensors) -> CtrlResult:
        finite = np.all(np.isfinite(s.imu_thigh_L)) and np.all(np.isfinite(s.imu_thigh_R))
        if self.model_angle_source == "thigh":
            finite = finite and np.isfinite(s.thigh_angle_R) and np.isfinite(s.thigh_angle_L)
        current_fault = not s.imu_control_valid or not finite
        self._imu_fault = ((self._imu_fault or current_fault)
                           if self.latch_imu_fault else current_fault)
        if self._imu_fault:
            # No stale history enters the network, and no previous policy
            # torque resumes automatically when BLE recovers.
            return CtrlResult(0.0, 0.0, 0.0, 0.0, {"imu_fault": 1.0})
        # ---- raw sensors -> right-leg convention, SI units ----
        hip_r_raw = self.hip_sign_r * np.deg2rad(float(s.pos_R) - self.hip_offset_deg_r)
        hip_l_raw = self.hip_sign_l * np.deg2rad(float(s.pos_L) - self.hip_offset_deg_l)

        # Encoder-derived joint velocity. Diagnostic only: the model is fed the
        # IMU thigh velocity, because that is what the distillation dataset used.
        hip_r_u_raw = self.hip_sign_r * np.deg2rad(float(s.vel_R))
        hip_l_u_raw = self.hip_sign_l * np.deg2rad(float(s.vel_L))

        gi = self.thigh_gyro_idx
        thigh_vel_r_raw = np.deg2rad(self.thigh_vel_sign_r * float(s.imu_thigh_R[gi]))
        thigh_vel_l_raw = np.deg2rad(self.thigh_vel_sign_l * float(s.imu_thigh_L[gi]))

        # ---- low-pass the control-path signals ----
        self.hip_r_filt = self._lpf(self.hip_r_filt, hip_r_raw, self.hip_filter_tau)
        self.hip_l_filt = self._lpf(self.hip_l_filt, hip_l_raw, self.hip_filter_tau)
        self.hip_r_u_filt = self._lpf(self.hip_r_u_filt, hip_r_u_raw, self.hip_filter_tau)
        self.hip_l_u_filt = self._lpf(self.hip_l_u_filt, hip_l_u_raw, self.hip_filter_tau)

        # ---- build the bilateral model input ----
        # Channel order is fixed by the distillation dataset and checked against
        # meta.json at construction:
        #   input_size 4: [selected_angle_r, selected_angle_l, thigh_angvel_r, thigh_angvel_l]
        #   input_size 2: [thigh_angvel_r, thigh_angvel_l]
        # EVERY channel is raw, for one reason: that is the bandwidth the
        # model was trained on. H1018AMPHipExoWrapper._read_sensor_channel feeds
        # the sensor window the unfiltered dof_position_array()[3] / femur body
        # ang_vel; the filtered hip angle (hip_r_filt) exists only inside the
        # impedance law, and the training env filters no sensor channel. The knee stack fed its model a filtered joint angle,
        # but that is not the precedent to follow here.
        raw_all = np.array([hip_r_raw, hip_l_raw,
                            thigh_vel_r_raw, thigh_vel_l_raw,
                            s.thigh_angle_R, s.thigh_angle_L], dtype=np.float32)
        x_last = (raw_all[self._channel_src] - self.sensor_mean) / self.sensor_std
        seq = self.x.push_last(x_last)                       # (C, T)
        x = seq.reshape(self.in_shape).astype(np.float32, copy=False)   # (1, C, T)

        # ---- non-blocking TRT inference: consume newest, submit newest ----
        new_result = self._get_latest_inference()
        if new_result is not None:
            self.last_out = np.asarray(new_result, dtype=np.float32).reshape(self.out_shape).copy()
        self._try_put_latest({"x": x})

        y = self.last_out.copy()

        # ---- decode the bilateral action ----
        K_r_raw = self._map_K(y[0])
        K_l_raw = self._map_K(y[2])

        gait_r_cmd = float(np.clip(y[1], -1.0, 1.0))
        gait_l_cmd = float(np.clip(y[3], -1.0, 1.0))

        # Slew-limit the gait command before filtering (training controller does
        # the same, in this order).
        gait_r_cmd = self._rate_limit(gait_r_cmd, self.gait_r_raw_prev, self.rate_max)
        gait_l_cmd = self._rate_limit(gait_l_cmd, self.gait_l_raw_prev, self.rate_max)

        self.K_r_raw = K_r_raw
        self.K_l_raw = K_l_raw
        self.K_r_filt = self._lpf(self.K_r_filt, K_r_raw, self.impedance_filter_tau)
        self.K_l_filt = self._lpf(self.K_l_filt, K_l_raw, self.impedance_filter_tau)

        self.gait_r_filt = self._lpf(self.gait_r_filt, gait_r_cmd, self.gait_filter_tau)
        self.gait_l_filt = self._lpf(self.gait_l_filt, gait_l_cmd, self.gait_filter_tau)
        self.gait_r_raw_prev = gait_r_cmd
        self.gait_l_raw_prev = gait_l_cmd

        # ---- impedance law ----
        tau_r, diag_r = self._torque_one_leg(self.hip_r_filt, self.K_r_filt, self.gait_r_filt)
        tau_l, diag_l = self._torque_one_leg(self.hip_l_filt, self.K_l_filt, self.gait_l_filt)

        tau_r *= self.mass_ratio
        tau_l *= self.mass_ratio

        # ---- output smoothing ----
        self.tau_r_filt = self._lpf(self.tau_r_filt, tau_r, self.output_filter_tau)
        self.tau_l_filt = self._lpf(self.tau_l_filt, tau_l, self.output_filter_tau)

        # ---- back to each side's motor frame ----
        out_r = self.torque_sign_r * self.tau_r_filt
        out_l = self.torque_sign_l * self.tau_l_filt

        return CtrlResult(
            model_out_R=float(out_r),
            model_out_L=float(out_l),
            applied_R=float(out_r),
            applied_L=float(out_l),
            extra={
                "hip_angle_r": float(self.hip_r_filt),
                "hip_angle_l": float(self.hip_l_filt),
                "hip_angle_r_u": float(self.hip_r_u_filt),
                "hip_angle_l_u": float(self.hip_l_u_filt),
                "K_r": float(self.K_r_filt),
                "K_l": float(self.K_l_filt),
                "Soft_ctrl_r": float(self.gait_r_filt),
                "Soft_ctrl_l": float(self.gait_l_filt),
                "w_flex_r": diag_r["w_flex"], "w_ext_r": diag_r["w_ext"],
                "w_flex_l": diag_l["w_flex"], "w_ext_l": diag_l["w_ext"],
                "tau_raw_r": diag_r["tau_raw"], "tau_raw_l": diag_l["tau_raw"],
            },
        )

    # ------------------------------------------------------------------
    def start(self):
        """Start inference and wait until it is ready, before any motor is enabled."""
        self.worker.start()
        engine = Path(self.engine_path)
        engine = f"{engine.parent.name}/{engine.name}"
        deadline = time.monotonic() + TRT_READY_TIMEOUT_S
        while not self.worker.ready.wait(0.05):
            if not self.worker.is_alive():
                raise RuntimeError(
                    f"TensorRT worker exited during startup (exit code "
                    f"{self.worker.exitcode}); see its error above. Engine: {engine}")
            if time.monotonic() >= deadline:
                self.worker.terminate()
                raise RuntimeError(
                    f"TensorRT worker not ready after {TRT_READY_TIMEOUT_S:g} s. "
                    f"Engine: {engine}")

    def close(self):
        try:
            self.in_q.put_nowait(None)
        except Exception:
            pass

        try:
            self.worker.join(timeout=1.5)
        except Exception:
            pass

        for q in (self.out_q, self.in_q):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
