"""
actuators — TMotor CAN driver, vendored into this repo.

Previously this repo reached into the continual-online-personalization checkout
via sys.path. It is now a self-contained package: this repo can be
moved, copied to another Jetson, or run without that sibling checkout present.

The module filenames deliberately match upstream so a future sync is a plain
diff:

    diff --strip-trailing-cr -u \
        <path-to>/continual-online-personalization/Utils_actuator_group.py \
        actuators/Utils_actuator_group.py

(--strip-trailing-cr because the upstream files are CRLF and these are LF.)

(PROVENANCE.md, which recorded the source commit, was not included in this copy.)

Usage:
    from actuators import ActuatorGroup, TMotorV3

    motors = ActuatorGroup(
        [TMotorV3(1, "AK80-9"), TMotorV3(2, "AK80-9")],
        can_args={"bustype": "socketcan", "channel": "can0"},
        exit_manually=True,
    )
"""

from .Utils_tmotor_v3 import TMotorV3, MotorData
from .Utils_actuator_group import ActuatorGroup

__all__ = ["ActuatorGroup", "TMotorV3", "MotorData"]
