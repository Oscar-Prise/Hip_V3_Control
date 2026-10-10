# controllers/base.py
#
# Hip-exo sensor and result types.
#
# Motor values are RAW, in each motor's own frame: degrees and degrees/second.
# Sign correction lives in the controllers, driven by config keys, so the trial
# log keeps the unmodified hardware values.
from dataclasses import dataclass


@dataclass(slots=True)
class Sensors:
    pos_L: float              # left hip encoder angle  (deg, motor frame)
    pos_R: float              # right hip encoder angle (deg, motor frame)
    vel_L: float              # left hip encoder velocity  (deg/s, motor frame)
    vel_R: float              # right hip encoder velocity (deg/s, motor frame)


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
