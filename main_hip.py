"""
main_hip.py — bilateral hip exoskeleton control loop (DOFC).

Hardware: TMotor AK80-9s over raw SocketCAN (ActuatorGroup / TMotorV3), encoders
only. That lives behind utils/exo_hardware.HipExo, so this file only sequences
the loop: read, control, command, log.

Usage:
    python main_hip.py cfg/dofc_hip.yaml

Startup order matters. HipExo zeroes the motor encoders when it is constructed,
so the wearer must be standing in the NEUTRAL reference pose at that prompt —
every angle for the rest of the trial is measured from that zero.

Before every subject run:
    Q1. trigger_type — "mocap" if the mocap PC starts the trial, else "typing".
    Q2. Is trial_name updated? The log is written to logged_data/<trial_name>.npz
        and an existing file of that name is overwritten without warning.
"""

import argparse
import atexit
import enum
import gc
import os
import signal
import sys
import time
import traceback
from typing import Optional

import numpy as np
import yaml
import Jetson.GPIO as GPIO

# Ensure this repo is importable no matter where the loop is launched from.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(REPO_ROOT, "logged_data")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.dont_write_bytecode = True   # no __pycache__ folders in the repo

from controllers import build_controller
from utils.utils import RateKeeper
from utils.Header_Mocap_trigger import Mocap_trigger
from utils.teleplot import Teleplot
from utils.exo_hardware import HipExo


# Keys shown for a check before each run, in this order.
SHOWN_CONFIG_KEYS = (
    "trial_name", "exo_on", "trigger_type", "scale", "torque_limit",
    "controller_name", "kappa", "delay", "exp_time_sec",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bilateral hip exoskeleton control loop.")
    parser.add_argument("config", help="Path to YAML config file.")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(cfg).__name__}")

    cfg["config_path"] = os.path.abspath(config_path)
    return cfg


def print_config(cfg: dict) -> None:
    print("=== CONFIG ===")
    for k in SHOWN_CONFIG_KEYS:
        if k not in cfg:
            continue
        print(f"{k}: {cfg[k]}")
    input("\n====Check Config====\nHit Enter to continue...")


def build_data_log(cfg: dict) -> dict:
    log_size = int(cfg["exp_time_sec"] * cfg["fs"])
    keys = [
        "time",
        # --- hip state, flexion positive, filtered, radians ---
        "hip_angle_r", "hip_angle_l",
        # --- raw encoder, degrees, each motor's own frame ---
        "enc_pos_r", "enc_pos_l", "enc_vel_r", "enc_vel_l",
        # --- DOFC internals ---
        "dofc_y", "dofc_y_delayed",
        # --- commands ---
        "cmd_R", "cmd_L",
        "GPIO",
    ]
    return {k: np.zeros(log_size) for k in keys}


class GPIOControl:
    """Single output pin pulsed at trial start and trial end so the mocap system
    can align its clock with this log."""

    def __init__(self, pin: int):
        self.pin = int(pin)
        self._state = 0
        try:
            GPIO.setwarnings(False)
            GPIO.cleanup()
        except Exception:
            pass
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

    def pulse_start(self):
        GPIO.output(self.pin, GPIO.HIGH)
        self._state = 1

    def pulse_end(self):
        GPIO.output(self.pin, GPIO.LOW)
        self._state = 0

    def state(self) -> int:
        return self._state

    def close(self):
        try:
            GPIO.output(self.pin, GPIO.LOW)
            GPIO.cleanup()
        except Exception:
            pass


class Side(enum.Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# ==============================
# Runner
# ==============================
class DualHipRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_log = build_data_log(cfg)
        self.exo: Optional[HipExo] = None
        self.controller = None
        self.current_idx = 0
        self.gpio = GPIOControl(self.cfg["GPIO_OUTPUT_PIN"])
        self.mocap_trigger = None
        self._shutdown_done = False

    def setup(self):
        if self.cfg["trigger_type"] == "mocap":
            try:
                self.mocap_trigger = Mocap_trigger(
                    server_ip=self.cfg.get("mocap_server_ip", "172.24.44.177"),
                    port_number=self.cfg.get("mocap_server_port", 10),
                )
                self.mocap_trigger.start_client()
            except Exception as e:
                print(f"[Mocap] start_client error: {e}")

        self.tp = Teleplot()   # address/port: utils/teleplot.py
        # Teleplot display signs, so every plot reads flexion-positive.
        c = self.cfg
        self._hip_sign = (float(c.get("hip_sign_r", 1.0)), float(c.get("hip_sign_l", 1.0)))
        self._hip_offset = (float(c.get("hip_offset_deg_r", 0.0)),
                            float(c.get("hip_offset_deg_l", 0.0)))
        self._torque_sign = (float(c.get("torque_sign_r", 1.0)), float(c.get("torque_sign_l", 1.0)))

        # Build the controller BEFORE enabling motors so config errors stop us early.
        self.controller = build_controller(self.cfg["controller_name"], config=self.cfg)
        self.controller.start()
        self.exo = HipExo(self.cfg)

        print("\n--- Bilateral Hip Exo Control Loop Started ---")
        print("Press Ctrl+C to stop.")

    def shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True

        # Motors first, always.
        try:
            if self.exo:
                self.exo.stop()
        except Exception as e:
            print(f"[Shutdown] exo.stop failed: {e}")

        try:
            if self.controller:
                self.controller.close()
        except Exception as e:
            print(f"[Shutdown] controller.close failed: {e}")

        try:
            if self.gpio is not None:
                self.gpio.pulse_end()
                self.gpio.close()
        except Exception as e:
            print(f"[Exit] GPIO cleanup error: {e}")

        try:
            gc.enable()
            gc.collect()
        except Exception:
            pass

        print("System Shutdown Complete.")

    def run(self):
        if self.cfg["trigger_type"] == "typing":
            _ = input("Press Enter to start...\n")
        elif self.cfg["trigger_type"] == "mocap":
            if self.mocap_trigger is not None:
                self.mocap_trigger.wait_for_trigger()
            else:
                print("[WARN] mocap_trigger is None. Starting immediately.")
        else:
            raise NotImplementedError(f"Unknown trigger_type: {self.cfg['trigger_type']}")

        side = Side(self.cfg["side"])
        use_left = side in (Side.LEFT, Side.BOTH)
        use_right = side in (Side.RIGHT, Side.BOTH)

        rk = RateKeeper(self.cfg["fs"])
        rk.start()
        t0 = time.perf_counter()
        LOG_DIVIDER = 1

        trial_start_sec = 0.0
        trial_dur_sec = self.cfg["exp_time_sec"]
        pulse_after_start = self.cfg["GPIO_START_DELAY_SEC"]

        first_pulse_sent = False
        first_pulse_end = None
        second_pulse_sent = False
        second_pulse_end = None

        while True:
            overrun, t_sched, k = rk.wait()
            step_start = time.perf_counter()

            # ---- 1. read hardware ----
            s = self.exo.read()

            # ---- 2. control ----
            r = self.controller.step(s)
            cmd_L = float(r.applied_L) if use_left else 0.0
            cmd_R = float(r.applied_R) if use_right else 0.0

            if not self.cfg["exo_on"]:
                cmd_L = 0.0
                cmd_R = 0.0

            cmd_L = cmd_L * self.cfg["scale"]
            cmd_R = cmd_R * self.cfg["scale"]

            cmd_L = clamp(cmd_L, -self.cfg["torque_limit"], self.cfg["torque_limit"])
            cmd_R = clamp(cmd_R, -self.cfg["torque_limit"], self.cfg["torque_limit"])

            # ---- 3. command motors ----
            # Sent verbatim. Mounting direction is handled once, by the
            # controller's torque_sign_r / torque_sign_l config keys.
            cmd_R, cmd_L = self.exo.set_torques(cmd_R if use_right else 0.0,
                                               cmd_L if use_left else 0.0)

            now = step_start - t0

            # ---- 4. GPIO sync pulses ----
            if (not first_pulse_sent) and (now >= trial_start_sec + pulse_after_start):
                try:
                    self.gpio.pulse_start()
                except Exception as e:
                    print(f"[GPIO] first pulse_start error: {e}")
                first_pulse_sent = True
                first_pulse_end = now + self.cfg["PULSE_WIDTH_SEC"]

            if first_pulse_sent and (first_pulse_end is not None) and (now >= first_pulse_end):
                try:
                    self.gpio.pulse_end()
                except Exception as e:
                    print(f"[GPIO] first pulse_end error: {e}")
                first_pulse_end = None

            if (not second_pulse_sent) and (now >= trial_start_sec + trial_dur_sec):
                try:
                    self.gpio.pulse_start()
                except Exception as e:
                    print(f"[GPIO] second pulse_start error: {e}")
                second_pulse_sent = True
                second_pulse_end = now + self.cfg["PULSE_WIDTH_SEC"]

            if second_pulse_sent and (second_pulse_end is not None) and (now >= second_pulse_end):
                try:
                    self.gpio.pulse_end()
                except Exception as e:
                    print(f"[GPIO] second pulse_end error: {e}")
                second_pulse_end = None

            step_end = time.perf_counter()
            actual_time = step_start - t0

            # ---- 5. telemetry ----
            # Flexion-positive on both sides, in degrees; the log keeps raw values.
            if k % LOG_DIVIDER == 0:
                try:
                    (hs_r, hs_l), (ts_r, ts_l) = self._hip_sign, self._torque_sign
                    self.tp.sendGroup("torque_cmd", (
                        ("cmd_R", ts_r * cmd_R),
                        ("cmd_L", ts_l * cmd_L),
                    ))
                    self.tp.sendGroup("encoder_position", (
                        ("enc_pos_r", hs_r * (s.pos_R - self._hip_offset[0])),
                        ("enc_pos_l", hs_l * (s.pos_L - self._hip_offset[1])),
                    ))
                    self.tp.sendGroup("dofc", (
                        ("y", r.extra.get("dofc_y", 0.0)),
                        ("y_delayed", r.extra.get("dofc_y_delayed", 0.0)),
                    ))
                    self.tp.sendValue("GPIO", self.gpio.state())
                    self.tp.sendValue("loop_time", step_end - step_start)
                except Exception:
                    pass

            # ---- 6. data logging ----
            if self.current_idx < len(self.data_log["time"]):
                i = self.current_idx
                dl = self.data_log

                dl["time"][i] = actual_time

                dl["hip_angle_r"][i] = r.extra.get("hip_angle_r", 0.0)
                dl["hip_angle_l"][i] = r.extra.get("hip_angle_l", 0.0)

                dl["enc_pos_r"][i] = s.pos_R
                dl["enc_pos_l"][i] = s.pos_L
                dl["enc_vel_r"][i] = s.vel_R
                dl["enc_vel_l"][i] = s.vel_L

                dl["dofc_y"][i] = r.extra.get("dofc_y", 0.0)
                dl["dofc_y_delayed"][i] = r.extra.get("dofc_y_delayed", 0.0)

                dl["cmd_R"][i] = cmd_R
                dl["cmd_L"][i] = cmd_L
                dl["GPIO"][i] = self.gpio.state()

            self.current_idx += 1


_RUNNER: Optional[DualHipRunner] = None


def _handle_signal(sig, frame):
    """Owns SIGINT/SIGTERM so the motors are disabled AND the trial is saved.
    HipExo passes exit_manually=True to ActuatorGroup precisely so this runs
    instead of ActuatorGroup's own handler, which would sys.exit(0) before the
    .npz is written."""
    global _RUNNER
    print(f"\nSignal {sig} received. Shutting down...")
    try:
        if _RUNNER:
            _RUNNER.shutdown()
            _save_log(_RUNNER)
    finally:
        sys.exit(0)


def _save_log(runner: "DualHipRunner"):
    if getattr(runner, "_log_saved", False):
        return
    runner._log_saved = True
    if runner.current_idx <= 0:
        print("No samples recorded — nothing to save.")
        return
    print("Preparing data for saving...")
    out = {k: v[:runner.current_idx] for k, v in runner.data_log.items()}
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{runner.cfg['trial_name']}.npz")
    np.savez(path, **out)
    print(f"=== Saved {runner.current_idx} samples to {path} ===")


def main():
    global _RUNNER
    args = parse_args()
    cfg = load_config(args.config)
    print_config(cfg)

    runner = DualHipRunner(cfg)
    _RUNNER = runner

    atexit.register(runner.shutdown)
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    gc.disable()
    try:
        runner.setup()
        runner.run()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received.")
    except Exception as e:
        print(f"[MAIN ERROR] {e}")
        traceback.print_exc()
    finally:
        runner.shutdown()
        gc.enable()
        gc.collect()
        _save_log(runner)


if __name__ == "__main__":
    main()
