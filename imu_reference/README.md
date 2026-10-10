# IMU reference (Muse thigh IMUs)

Unmodified copies of the IMU-related files from the original branch
(`AMP-Hip-controller.zip`), kept so the IMU pipeline can be brought back
later. **Nothing in the repo imports this folder**; DOFC runs without it.

| File here | What it is |
|---|---|
| `utils/Header_MuseIMU.py` | The Muse BLE driver (self-contained; copy as is). Also a standalone IMU test. |
| `utils/exo_hardware.py` | Original `HipExo`: motors **plus** IMUs, calibration, dropout hold. |
| `controllers/base.py` | Original `Sensors` with the IMU fields (and `RollingWindow`). |
| `controllers/impedance_rl_hip.py` | The controller that consumes the IMU data (example consumer). |
| `cfg/impedance_rl_hip.yaml` | Config with the `muse_*` keys and `muse_devices`. |
| `main_hip.py` | Original loop, with IMU logging and Teleplot plots. |

---

## 1. How the data is acquired (`Header_MuseIMU.py`)

**Hardware.** Two 221e Muse IMUs, one per thigh, each over Bluetooth Low
Energy through its own USB dongle (`muse_devices` in the config maps each side
to an IMU address and a dongle MAC; both can share one dongle, less evenly).

**Background thread.** `MuseIMUs.start()` launches a daemon thread running an
asyncio/Bleak loop, so BLE never blocks the 100 Hz motor loop. Startup:

1. `hciconfig -a` maps each dongle MAC to `hciN`; clears any BLE link left
   over from a crashed run (`bluetoothctl disconnect`).
2. Scan + connect both sides in parallel, retrying (3 attempts by default).
3. Per IMU: send STOP and confirm IDLE, read the gyro/accel full-scale
   settings (to get the LSB → unit scale), print battery, subscribe to the
   data characteristic, send START in mode `0x33` (gyro + accel + orientation
   quaternion + timestamp) at 100 Hz, and confirm the device is streaming.
4. Rate check: the IMU's own timestamps must advance at 10 ms, or startup fails.
5. `start()` returns once both streams are fresh.

**Decoding each packet** (`decode_latest`). A BLE packet can batch several
records; only the **newest** is kept. Each record: 3 gyro, 3 accel, 3
quaternion components (int16) and a 48-bit device timestamp.
- Gyro → deg/s, accel → m/s² (using the full-scale read at startup).
- Quaternion `x,y,z` / 32767, `w` reconstructed as `sqrt(1 - x² - y² - z²)`.
- **Thigh angle** = one Euler angle from the quaternion about `muse_axis`
  (`x` roll, `y` pitch, `z` yaw), times `muse_axis_sign_r/_l`, minus
  `angle_offset_deg`, minus the neutral-calibration offset, wrapped to ±π, in
  **radians**. This is the thigh's orientation in the world, **not** the hip
  joint angle. With `y` (current rig), asin limits it to ±90°.
- Out-of-order or repeated timestamps are dropped.

Result per side is a `MuseSample`: `imu` = `(ax, ay, az, gx, gy, gz)` raw
body frame, `quaternion` = `(w,x,y,z)`, `thigh_angle` (rad), `device_time_ms`,
`received_at` (host time).

**Reading it.** `snapshot()` returns `(samples, ages, valid)` instantly under
a lock: the latest sample per side, its age in seconds, and whether **both**
sides are connected and fresher than `muse_stale_timeout_s` (30 ms).

**Neutral calibration** (`calibrate_neutral`, before motors are enabled). The
wearer stands still for 2 s; samples with gyro norm > 5 deg/s are rejected;
the circular mean of each side's thigh angle becomes that side's zero. Fails
startup if fewer than 40% of the expected samples were still, or if the std
exceeds 3°.

All tunables (rate, timeouts, retries, calibration limits, battery warnings)
live in `DEFAULTS` at the top of the file; any config key with the same name
overrides them.

## 2. How it is wired into the loop (`exo_hardware.py`, `base.py`)

Startup order in `HipExo.__init__`: connect IMUs → prompt "stand still in
NEUTRAL" → `calibrate_neutral()` → zero motor encoders and enable motors.

Every loop step `HipExo.read()` calls `imus.arm()` on the first step (from
then on a stale stream counts as a fault), then `snapshot()`, and packs a
`Sensors` with: `imu_thigh_L/R` (6-vectors), `thigh_angle_L/R`,
`quaternion_L/R`, `imu_age_L/R`, `imu_control_valid`, plus the encoders.

**Dropout safety** in `HipExo.set_torques()`: if the IMUs are invalid at the
moment of sending, it holds the last valid torque for up to
`muse_command_hold_timeout_s` (0.5 s), then sends zero. Torque resumes as soon
as both streams are fresh again (unless `muse_latch_stale_fault: true`).

## 3. How the controller uses it (`impedance_rl_hip.py`, `step()`)

1. **Fault gate:** if `not s.imu_control_valid` or any IMU value is NaN, return
   zero torque immediately (nothing stale goes into the model).
2. **Thigh velocity:** pick one gyro axis (`muse_gyro_axis`, currently `z` →
   index 5 of the 6-vector), multiply by `muse_gyro_sign_r/_l`, convert deg/s →
   **rad/s**. Unfiltered.
3. **Thigh angle:** `s.thigh_angle_R/L` used as is (already signed,
   calibrated, rad). Unfiltered.
4. Model input per step = `[thigh_angle_r, thigh_angle_l, thigh_vel_r,
   thigh_vel_l]`, z-scored with `sensor_mean/std` from the model's
   `.meta.json`, pushed into a 95-sample rolling window → TCN.

So in the end the policy uses only **2 numbers per thigh**: one Euler angle
and one gyro axis. Accel and quaternion are logged but not used for control.
The hip **encoder** angle is still used separately inside the impedance law.

Current rig signs (flexion positive on both sides): `muse_axis: y`,
`muse_axis_sign_r: 1`, `_l: -1`; `muse_gyro_axis: z`, `muse_gyro_sign_r: 1`,
`_l: -1`.

## 4. Bringing it back for a new IMU controller

1. Install: `sudo apt install bluez` and `python -m pip install --user bleak==0.22.2`.
2. Copy `utils/Header_MuseIMU.py` into `utils/`.
3. Replace `utils/exo_hardware.py` and `controllers/base.py` with the copies
   here (the current stripped versions are encoder-only; DOFC keeps working
   with the originals, since it only reads `pos_*`).
4. In your config, add `imu_type: "muse"` and copy the `muse_*` keys and the
   `muse_devices` block from `cfg/impedance_rl_hip.yaml`. Encoder-only configs
   need `imu_type: "none"`.
5. In your controller's `step()`, copy the fault gate and the gyro/angle
   conversion from section 3 (`impedance_rl_hip.py`, top of `step()`).
6. Optional: copy the IMU log keys and Teleplot plots from `main_hip.py`
   (`build_data_log`, the `muse_diag` block, the telemetry section).
7. Test the IMUs alone first (pass a config that has `muse_devices`):
   `python utils/Header_MuseIMU.py cfg/<your_config>.yaml --duration 30`.
   Both sides should show ~100 Hz with zero dropped samples.
