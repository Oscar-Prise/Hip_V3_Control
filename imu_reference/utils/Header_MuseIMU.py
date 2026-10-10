"""Two 221e Muse IMUs, received outside the motor loop (Linux / Bleak 0.22.2).

Protocol: https://docs.221e.com/documentation/muse-protocols/muse-v3_-communication/
Uses the frequency tester's direct IMU + orientation + timestamp mode (0x33).
Only the newest record is decoded; no plotting, queues or disk I/O here.
Quaternion is (w,x,y,z); thigh_angle is the Euler angle about muse_axis (x roll,
y pitch or z yaw) times muse_axis_sign[_r/_l], minus the fixed angle_offset_deg
and the neutral offset from calibrate_neutral(), in radians. Gyro is body-frame
deg/s, acceleration is m/s². The thigh angle is NOT a hip joint angle.

BLE only: wired (USB serial) transport was dropped. Differs from the original
Header_MuseIMU.py in connection setup: each
side scans and connects in parallel on its own dongle (or in turn, when both
share one dongle), and a failed scan or
connect is retried (muse_connect_attempts, muse_connect_timeout_s,
muse_retry_delay_s). Stream setup is also verified: STOP, confirm IDLE with
0x82, START, check START's ACK (mode + frequency) when readable, confirm direct
TX with 0x82, then require device timestamps to advance at the requested
period (muse_verify_intervals, muse_verify_timeout_s, muse_verify_tolerance).
Decoding, axis selection, neutral calibration and stale detection are
unchanged.

Run standalone to stream both IMUs without the controller and get a
timing report (latency, late and dropped records) when it stops:
    python utils/Header_MuseIMU.py [config.yaml] [--duration S] [--print-hz HZ]
"""
from array import array
import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
import math
import re
import statistics
import struct
import subprocess
import threading
import time

CMD = "d5913036-2d8a-41ee-85b9-4e361aa5c8a7"
DATA = "09bf2c52-d1d9-c0b7-4145-475964544307"
RECORD = struct.Struct("<9h6s")
FREQUENCIES = {25: 1, 50: 2, 100: 4, 200: 8}
MODE = bytes([0x33, 0, 0])         # gyro | accel | orientation | timestamp
STATE_IDLE, STATE_TX_DIRECT = 0x02, 0x08
STOP = bytes([2, 1, STATE_IDLE])

# Default for every Muse setting; a config key of the same name overrides it.
# Only the rig-specific keys (muse_devices, muse_axis*, muse_gyro_*) belong in
# the YAML. utils/exo_hardware.py and controllers/impedance_rl_hip.py read the
# stale-handling defaults from here too, so each value is defined once.
DEFAULTS = {
    "muse_transport": "ble",               # only BLE is supported
    "muse_frequency": 100,                 # 25 | 50 | 100 | 200 Hz
    # Stale input: a sample older than this (host reception age, NOT measured
    # acquisition-to-host latency) makes the input invalid. utils/exo_hardware.py
    # then holds the last valid commands for up to muse_command_hold_timeout_s
    # and forces zero after that. With muse_latch_stale_fault false, a recovered
    # stream immediately resumes new commands; true requires a restart.
    "muse_stale_timeout_s": 0.030,         # three 100 Hz periods
    "muse_latch_stale_fault": False,
    "muse_command_hold_timeout_s": 0.5,
    # Connection: each attempt scans, then connects; failures are retried.
    "muse_scan_timeout_s": 10,
    "muse_connect_attempts": 3,
    "muse_connect_timeout_s": 15,
    "muse_retry_delay_s": 1,
    # All retries (3 x (10 s scan + 15 s connect + 1 s)) plus stream setup and
    # the rate check fit in this; 60 s cut off the last retry.
    "muse_startup_timeout_s": 120,
    # Startup rate check: device timestamps must advance at the requested period.
    "muse_verify_intervals": 20,
    "muse_verify_timeout_s": 3,
    "muse_verify_tolerance": 0.2,
    # Startup prints each IMU's battery and warns below either limit.
    "muse_battery_warn_percent": 20,
    "muse_battery_warn_mv": 3600,
    # Neutral calibration: after both streams connect, the wearer stands still in
    # the neutral pose and each side's selected-axis offset is estimated before
    # the motors are enabled.
    "muse_auto_calibrate": True,
    "muse_calibration_duration_s": 2.0,
    "muse_calibration_gyro_max_dps": 5.0,  # samples above this 3-axis gyro norm are rejected
    "muse_calibration_max_std_deg": 3.0,   # fail startup if the accepted angle is unstable
    "muse_calibration_min_fraction": 0.4,  # required stationary samples vs ideal count
}


def _wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _circular_mean_std(angles):
    """Circular mean/std in radians, robust to the -pi/+pi boundary."""
    if not angles:
        raise ValueError("No angles")
    s = sum(math.sin(a) for a in angles) / len(angles)
    c = sum(math.cos(a) for a in angles) / len(angles)
    resultant = min(1.0, max(math.hypot(s, c), 1e-12))
    return math.atan2(s, c), math.sqrt(max(0.0, -2.0 * math.log(resultant)))


@dataclass(frozen=True)
class MuseSample:
    imu: tuple                     # ax,ay,az (m/s²), gx,gy,gz (deg/s)
    quaternion: tuple              # w,x,y,z
    thigh_angle: float             # signed selected axis minus fixed + auto neutral offsets, rad
    device_time_ms: int
    received_at: float             # host monotonic seconds, NOT acquisition time


def decode_latest(packet, gyro_scale, accel_scale, received_at, offset_deg=0.0,
                  axis="x", axis_sign=1.0):
    """Decode one newest sample; sensitivities are deg/s/LSB and mg/LSB."""
    if len(packet) <= 8 or (len(packet) - 8) % RECORD.size:
        raise ValueError("Malformed Muse IMU/orientation/timestamp packet")
    gx, gy, gz, ax, ay, az, qx, qy, qz, stamp = RECORD.unpack_from(
        packet, len(packet) - RECORD.size)
    x, y, z = (q / 32767.0 for q in (qx, qy, qz))
    n = x*x + y*y + z*z
    if n > 1.001:
        raise ValueError("Invalid Muse quaternion")
    if n > 1.0:                    # allow fixed-point quantization at w ~ 0
        scale = math.sqrt(n)
        x, y, z = x/scale, y/scale, z/scale
    w = math.sqrt(max(0.0, 1.0 - x*x - y*y - z*z))
    if axis == "x":
        euler = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    elif axis == "y":
        euler = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
    else:
        euler = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    angle = _wrap_angle(float(axis_sign) * euler - math.radians(offset_deg))
    imu = tuple(v * accel_scale * 0.00980665 for v in (ax, ay, az)) + tuple(
        v * gyro_scale for v in (gx, gy, gz))
    return MuseSample(imu, (w, x, y, z), angle,
                      int.from_bytes(stamp, "little"), received_at)


class MuseIMUs:
    """Latest-value receiver. start() blocks at setup; snapshot() never waits for BLE.

    Call calibrate_neutral() after start() and before arm(). After arm(), an
    excessive reception gap or disconnect marks the input invalid, until restart
    or until both streams recover (muse_latch_stale_fault). Host reception age
    cannot establish end-to-end sensor latency.
    """
    def __init__(self, cfg):
        def setting(key):
            return cfg.get(key, DEFAULTS[key])

        if str(setting("muse_transport")).lower() != "ble":
            raise ValueError("Header_MuseIMU supports muse_transport: ble only")
        self.axis = str(cfg.get("muse_axis", "x")).lower()
        if self.axis not in ("x", "y", "z"):
            raise ValueError("muse_axis must be x, y or z")
        fallback_sign = float(cfg.get("muse_axis_sign", 1.0))
        self.axis_sign = {
            "right": float(cfg.get("muse_axis_sign_r", fallback_sign)),
            "left": float(cfg.get("muse_axis_sign_l", fallback_sign)),
        }
        if any(sign not in (-1.0, 1.0) for sign in self.axis_sign.values()):
            raise ValueError("Muse axis signs must be 1.0 or -1.0")
        self.frequency = int(setting("muse_frequency"))
        if self.frequency not in FREQUENCIES:
            raise ValueError("muse_frequency must be 25, 50, 100 or 200 Hz")
        self.stale_s = float(setting("muse_stale_timeout_s"))
        self.scan_s = float(setting("muse_scan_timeout_s"))
        self.connect_attempts = int(setting("muse_connect_attempts"))
        self.connect_timeout_s = float(setting("muse_connect_timeout_s"))
        self.retry_delay_s = float(setting("muse_retry_delay_s"))
        if self.connect_attempts < 1:
            raise ValueError("muse_connect_attempts must be at least 1")
        self.startup_s = float(setting("muse_startup_timeout_s"))
        self.latch_fault = bool(setting("muse_latch_stale_fault"))
        self.battery_warn_percent = float(setting("muse_battery_warn_percent"))
        self.battery_warn_mv = float(setting("muse_battery_warn_mv"))
        if not all(math.isfinite(v) and v >= 0 for v in
                   (self.battery_warn_percent, self.battery_warn_mv)):
            raise ValueError("Muse battery warning limits must be finite and non-negative")
        self.battery = {"left": None, "right": None}   # (charge %, mV) read at startup
        self.auto_calibrate = bool(setting("muse_auto_calibrate"))
        self.calibration_duration_s = float(setting("muse_calibration_duration_s"))
        self.calibration_gyro_max_dps = float(setting("muse_calibration_gyro_max_dps"))
        self.calibration_max_std_deg = float(setting("muse_calibration_max_std_deg"))
        self.calibration_min_fraction = float(setting("muse_calibration_min_fraction"))
        if (not math.isfinite(self.calibration_duration_s) or self.calibration_duration_s <= 0 or
                not math.isfinite(self.calibration_gyro_max_dps) or
                self.calibration_gyro_max_dps < 0 or
                not math.isfinite(self.calibration_max_std_deg) or
                self.calibration_max_std_deg <= 0 or
                not 0 < self.calibration_min_fraction <= 1):
            raise ValueError("Invalid Muse neutral-calibration settings")
        # Startup rate check: device timestamps must advance at the requested
        # period, which also proves the IMU accepted this START and is not still
        # streaming an older configuration (0x82 reports state, not mode/Hz).
        self.verify_intervals = int(setting("muse_verify_intervals"))
        self.verify_timeout_s = float(setting("muse_verify_timeout_s"))
        self.verify_tolerance = float(setting("muse_verify_tolerance"))
        if self.verify_intervals < 1:
            raise ValueError("muse_verify_intervals must be at least 1")
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.stale_s, self.scan_s, self.startup_s,
                    self.connect_timeout_s, self.retry_delay_s,
                    self.verify_timeout_s, self.verify_tolerance)):
            raise ValueError("Muse timing limits must be finite and positive")
        self.devices = cfg["muse_devices"]
        if set(self.devices) != {"left", "right"}:
            raise ValueError("muse_devices must specify left and right")
        addresses = [str(v["address"]).upper() for v in self.devices.values()]
        if len(set(addresses)) != 2:
            raise ValueError("Left and right Muse addresses must be distinct")
        for spec in self.devices.values():
            if not math.isfinite(float(spec.get("angle_offset_deg", 0))):
                raise ValueError("Muse angle offsets must be finite")
        self._lock = threading.Lock()
        self._latest = {"left": None, "right": None}
        self._connected = {"left": False, "right": False}
        self._auto_offset_rad = {"left": 0.0, "right": 0.0}
        self.calibration_offsets_deg = {
            side: float(spec.get("angle_offset_deg", 0.0))
            for side, spec in self.devices.items()
        }
        self.calibrated = False
        self.valid_packets = {"left": 0, "right": 0}
        self.invalid_packets = {"left": 0, "right": 0}
        self._armed = False
        self._probe = None                 # side -> device timestamps, only during startup
        self.fault = None
        self.error = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._thread_main, name="MuseBLE", daemon=True)
        self._thread.start()
        try:
            if not self._ready.wait(self.startup_s):
                raise RuntimeError("Timed out waiting for both Muse streams")
            if self.error:
                raise RuntimeError(f"Muse startup failed: {self.error}") from self.error
        except BaseException:
            self.close()
            raise

    def _check_locked(self, now):
        fresh = all(self._connected[side] and sample is not None and
                    0 <= now - sample.received_at <= self.stale_s
                    for side, sample in self._latest.items())
        if self._armed and not fresh and self.fault is None:
            suffix = "until restart" if self.latch_fault else "until both streams recover"
            self.fault = f"Muse stale/disconnected: sensor input invalid {suffix}"
            # print(f"[Muse] {self.fault}")
        elif self._armed and fresh and self.fault is not None and not self.latch_fault:
            self.fault = None
            # print("[Muse] Both streams fresh: bilateral torque enabled")
        return fresh and self.fault is None

    def arm(self):
        with self._lock:
            self._armed = True
            self._check_locked(time.monotonic())

    def snapshot(self):
        with self._lock:
            now = time.monotonic()
            valid = self._check_locked(now)
            samples = self._latest.copy()
            ages = {side: now - sample.received_at if sample else math.inf
                    for side, sample in samples.items()}
        return samples, ages, valid

    def calibrate_neutral(self):
        """Zero each selected Muse angle while both thighs are stationary.

        Call after start() and before arm()/motor enable. Only unique device
        timestamps count, so a repeated BLE value cannot bias calibration.
        """
        if not self.auto_calibrate:
            print("[Muse] Automatic neutral calibration disabled; using configured offsets")
            return self.calibration_offsets_deg.copy()

        observations = {"left": [], "right": []}
        seen_stamp = {"left": None, "right": None}
        rejected = {"left": 0, "right": 0}
        deadline = time.monotonic() + self.calibration_duration_s
        axis_name = {"x": "X-roll", "y": "Y-pitch", "z": "Z-yaw"}[self.axis]
        print(f"[Muse] Calibrating neutral {axis_name} for "
              f"{self.calibration_duration_s:.1f} s; remain still...")
        while time.monotonic() < deadline:
            samples, _, _ = self.snapshot()
            for side, sample in samples.items():
                if sample is None or sample.device_time_ms == seen_stamp[side]:
                    continue
                seen_stamp[side] = sample.device_time_ms
                gyro_norm = math.sqrt(sum(g*g for g in sample.imu[3:6]))
                if gyro_norm <= self.calibration_gyro_max_dps:
                    observations[side].append(sample.thigh_angle)
                else:
                    rejected[side] += 1
            time.sleep(min(0.005, 0.25 / self.frequency))

        required = max(5, math.ceil(self.frequency * self.calibration_duration_s *
                                    self.calibration_min_fraction))
        corrections = {}
        standard_deviations = {}
        for side in ("left", "right"):
            count = len(observations[side])
            if count < required:
                raise RuntimeError(
                    f"Muse {side} neutral calibration got {count} stationary samples; "
                    f"need {required}. Remain still and check BLE reception.")
            mean, std = _circular_mean_std(observations[side])
            std_deg = math.degrees(std)
            if std_deg > self.calibration_max_std_deg:
                raise RuntimeError(
                    f"Muse {side} neutral calibration was unstable: std={std_deg:.2f} deg "
                    f"> {self.calibration_max_std_deg:.2f} deg")
            corrections[side] = mean
            standard_deviations[side] = std_deg

        with self._lock:
            for side in ("left", "right"):
                self._auto_offset_rad[side] += corrections[side]
                self.calibration_offsets_deg[side] = (
                    float(self.devices[side].get("angle_offset_deg", 0.0)) +
                    math.degrees(self._auto_offset_rad[side]))
                sample = self._latest[side]
                if sample is not None:
                    self._latest[side] = replace(sample, thigh_angle=_wrap_angle(
                        sample.thigh_angle - corrections[side]))
            self.calibrated = True

        print("[Muse] Neutral calibration complete: " + ", ".join(
            f"{side} offset={self.calibration_offsets_deg[side]:.2f} deg "
            f"(std={standard_deviations[side]:.2f} deg, "
            f"n={len(observations[side])}, moving_rejected={rejected[side]})"
            for side in ("left", "right")))
        return self.calibration_offsets_deg.copy()

    def _receive(self, side, packet, scales):
        now = time.monotonic()
        probe = self._probe
        if (probe is not None and len(packet) > 8 and not (len(packet) - 8) % RECORD.size
                and len(probe[side]) <= self.verify_intervals):
            # Every record, not just the newest: a packet may batch several.
            for offset in range(8 + 18, len(packet), RECORD.size):
                probe[side].append(int.from_bytes(packet[offset:offset + 6], "little"))
        try:
            sample = decode_latest(packet, *scales, now,
                                   float(self.devices[side].get("angle_offset_deg", 0)),
                                   self.axis, self.axis_sign[side])
        except (ValueError, struct.error):
            self.invalid_packets[side] += 1
            return
        with self._lock:
            # Check BEFORE replacing the sample: a late burst must not erase a gap.
            self._check_locked(now)
            previous = self._latest[side]
            if previous is not None and sample.device_time_ms <= previous.device_time_ms:
                self.invalid_packets[side] += 1
                return
            # Under the lock, so calibrate_neutral() cannot miss this sample.
            self._latest[side] = replace(sample, thigh_angle=_wrap_angle(
                sample.thigh_angle - self._auto_offset_rad[side]))
            self.valid_packets[side] += 1

    def _disconnected(self, side):
        with self._lock:
            self._connected[side] = False
            if not self._stop.is_set():
                self._check_locked(time.monotonic())

    @staticmethod
    async def _read_reply(client, code, length=None, timeout=3.0):
        """Poll the CMD value for a response to `code`; None if none appears.

        Response layout: 00 LEN CODE ERR PAYLOAD. `length` rejects a stale
        value left by a different command with the same code (STOP and START
        are both 0x02; only START's ACK has LEN 9).
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            reply = await client.read_gatt_char(CMD)
            # This firmware exposes the response in a fixed 20-byte GATT value,
            # zero-padded beyond the protocol's declared length.
            if (len(reply) >= 4 and reply[0] == 0 and reply[1] >= 2 and
                    len(reply) >= reply[1] + 2 and reply[2] == code and
                    (length is None or reply[1] == length)):
                return bytes(reply)
            await asyncio.sleep(0.02)
        return None

    @classmethod
    async def _query(cls, client, command):
        """Write a read-command and validate its application-level response."""
        await client.write_gatt_char(CMD, command, response=True)
        reply = await cls._read_reply(client, command[0])
        if reply is None:
            raise RuntimeError(f"No matching Muse response for {command.hex()}")
        if reply[3] != 0:
            raise RuntimeError(f"Muse rejected command {command.hex()}: {reply.hex()}")
        return reply[4:reply[1] + 2]

    @classmethod
    async def _wait_state(cls, client, expected, timeout=1.0):
        """Poll GET STATE (0x82) until it reports `expected`; return the last state."""
        deadline = time.monotonic() + timeout
        while True:
            payload = await cls._query(client, bytes([0x82, 0]))
            if len(payload) != 1:
                raise RuntimeError(f"Invalid Muse state response: {payload.hex()}")
            if payload[0] == expected or time.monotonic() >= deadline:
                return payload[0]
            await asyncio.sleep(0.05)

    async def _stop_stream(self, side, client):
        """Force IDLE and prove it with 0x82, so START is not sent mid-stream.

        STOP's own ACK is not read: it is unreliable when already idle.
        """
        for _ in range(3):
            await client.write_gatt_char(CMD, STOP, response=True)
            state = await self._wait_state(client, STATE_IDLE)
            if state == STATE_IDLE:
                return
            print(f"[Muse] {side} still in state 0x{state:02x} after STOP; resending")
        raise RuntimeError(f"Muse {side} did not enter IDLE (state 0x{state:02x})")

    async def _start_stream(self, side, client):
        """Send START, check its ACK if readable, and confirm direct TX via 0x82."""
        freq = FREQUENCIES[self.frequency]
        await client.write_gatt_char(CMD, bytes([2, 5, STATE_TX_DIRECT]) + MODE +
                                     bytes([freq]), response=True)
        # START's ACK echoes 00 09 02 ERR FS[3] MODE[3] FREQ: the only place the
        # device reports mode and frequency. Readability of write ACKs is not
        # guaranteed on this firmware, so a missing ACK is not fatal; the
        # startup rate check still covers frequency.
        ack = await self._read_reply(client, 0x02, length=9, timeout=0.5)
        if ack is None:
            print(f"[Muse] {side} START ACK not readable; relying on rate check")
        elif ack[3] != 0:
            raise RuntimeError(f"Muse {side} rejected START: {ack.hex()}")
        elif ack[7:10] != MODE or ack[10] != freq:
            raise RuntimeError(
                f"Muse {side} START ACK mode {ack[7:10].hex()} freq 0x{ack[10]:02x}, "
                f"expected {MODE.hex()} 0x{freq:02x}")
        state = await self._wait_state(client, STATE_TX_DIRECT)
        if state != STATE_TX_DIRECT:
            raise RuntimeError(f"Muse {side} state 0x{state:02x} after START, "
                               f"expected 0x{STATE_TX_DIRECT:02x} (direct TX)")

    async def _check_battery(self, side, client):
        """Print the battery level and warn when it is low; never fails startup."""
        try:
            charge = (await self._query(client, bytes([0x87, 0])))[0]
            mv = int.from_bytes((await self._query(client, bytes([0x88, 0])))[:2], "little")
        except Exception as exc:
            print(f"[Muse] {side} battery level unavailable: {exc!r}")
            return
        self.battery[side] = (charge, mv)
        if charge < self.battery_warn_percent or mv < self.battery_warn_mv:
            print(f"[Muse] WARNING: {side} battery low: {charge}% ({mv / 1000:.2f} V). "
                  "It may shut off or drop samples mid-trial; charge it first.")
        else:
            print(f"[Muse] {side} battery {charge}% ({mv / 1000:.2f} V)")

    async def _verify_rate(self):
        """Fail startup unless each side's device clock advances at the requested period."""
        period_ms = 1000.0 / self.frequency
        deadline = time.monotonic() + self.verify_timeout_s
        while any(len(v) <= self.verify_intervals for v in self._probe.values()):
            if time.monotonic() >= deadline:
                counts = {side: len(v) for side, v in self._probe.items()}
                raise RuntimeError(f"Muse rate check: too few records in "
                                   f"{self.verify_timeout_s:g} s: {counts}")
            await asyncio.sleep(0.02)
        for side, stamps in self._probe.items():
            stamps = stamps[:self.verify_intervals + 1]
            diffs = sorted(b - a for a, b in zip(stamps, stamps[1:]))
            median = diffs[len(diffs) // 2]
            hz = 1000.0 / median if median > 0 else math.inf
            # Integer-ms stamps jitter by 1 ms, hence the +1 slack; the median
            # also ignores an occasional lost record.
            if abs(median - period_ms) > self.verify_tolerance * period_ms + 1:
                raise RuntimeError(
                    f"Muse {side} streams at ~{hz:.0f} Hz (median {median} ms), "
                    f"requested {self.frequency} Hz; the IMU did not apply this START")
            print(f"[Muse] {side} rate OK: median {median} ms (~{hz:.0f} Hz)")

    @staticmethod
    def _release_stale_link(side, controller, address):
        """Disconnect a BlueZ link left behind by a run that never called close().

        BlueZ keeps the connection after the owning process dies, and a
        connected Muse stops advertising, so the scan could never find it.
        """
        def bluetoothctl(*commands):
            script = "".join(f"{c}\n" for c in (f"select {controller}",) + commands)
            return subprocess.run(["bluetoothctl"], input=script, capture_output=True,
                                  text=True, timeout=10).stdout
        try:
            if "Connected: yes" in bluetoothctl(f"info {address}"):
                print(f"[Muse] {side} {address} still connected from an earlier run; "
                      "disconnecting")
                bluetoothctl(f"disconnect {address}")
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"[Muse] {side}: could not check for a stale connection: {exc!r}")

    async def _run(self):
        from bleak import BleakClient, BleakScanner

        # This runs in the dedicated Muse thread, before acquisition starts, so
        # a bounded synchronous query is simpler and leaves no executor behind.
        result = subprocess.run(["hciconfig", "-a"], capture_output=True,
                                text=True, check=True, timeout=5)
        adapters = {mac.upper(): hci for hci, mac in re.findall(
            r"(hci\d+):.*?BD Address:\s*([\dA-Fa-f:]{17})", result.stdout, re.S)}
        # A missing dongle is a hardware/config error that retrying cannot fix.
        side_adapters = {}
        for side, spec in self.devices.items():
            controller = str(spec["adapter_mac"]).upper()
            if controller not in adapters:
                raise RuntimeError(f"Muse {side}: dongle {controller} not found")
            side_adapters[side] = adapters[controller]
            self._release_stale_link(side, controller, str(spec["address"]).upper())

        async with AsyncExitStack() as stack:
            clients = {}
            adapter_busy = {adapter: asyncio.Lock() for adapter in side_adapters.values()}

            async def connect(side):
                """Scan and connect one side, retrying transient BLE failures."""
                adapter = side_adapters[side]
                address = str(self.devices[side]["address"]).upper()
                for attempt in range(1, self.connect_attempts + 1):
                    try:
                        # BlueZ rejects overlapping connects on one adapter, so
                        # sides sharing a dongle take turns.
                        async with adapter_busy[adapter]:
                            print(f"[Muse] Scanning for {side} {address} via {adapter}... "
                                  f"(attempt {attempt}/{self.connect_attempts})")
                            found = await BleakScanner.discover(timeout=self.scan_s,
                                                                adapter=adapter)
                            device = next((d for d in found if d.address.upper() == address),
                                          None)
                            if device is None:
                                raise RuntimeError(f"Muse {side} not found via {adapter}")
                            # Only a fully connected client enters the stack, so a
                            # failed attempt leaves nothing behind to clean up.
                            client = await stack.enter_async_context(BleakClient(
                                device, adapter=adapter, timeout=self.connect_timeout_s,
                                disconnected_callback=lambda _, side=side:
                                    self._disconnected(side)))
                    except Exception as exc:
                        if attempt == self.connect_attempts:
                            raise RuntimeError(f"Muse {side}: failed after "
                                               f"{attempt} attempts: {exc!r}") from exc
                        print(f"[Muse] {side} attempt {attempt} failed: {exc!r}; retrying")
                        await asyncio.sleep(self.retry_delay_s)
                        continue
                    clients[side] = client
                    with self._lock:
                        self._connected[side] = True
                    print(f"[Muse] {side} connected via {adapter}")
                    return

            # Sides on separate dongles scan and connect in parallel; sides on one
            # shared dongle run one after the other. If one side fails, the other is cancelled and the stack disconnects
            # whatever already connected.
            tasks = [asyncio.create_task(connect(side)) for side in self.devices]
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            try:
                self._probe = {side: [] for side in clients}
                for side, client in clients.items():
                    # An IMU left streaming by an earlier session can ignore
                    # START, so force IDLE first and prove it with 0x82.
                    await self._stop_stream(side, client)
                    payload = await self._query(client, bytes([0xc0, 0]))
                    if len(payload) != 3:
                        raise RuntimeError("Invalid Muse full-scale response")
                    code = int.from_bytes(payload, "little")
                    await self._check_battery(side, client)
                    scales = ((.00875, .0175, .035, .07)[code & 3],
                              {0: .122, 8: .244, 12: .488, 4: .976}[code & 12])
                    await client.start_notify(DATA, lambda _, data, side=side, scales=scales:
                                              self._receive(side, data, scales))
                    await self._start_stream(side, client)
                await self._verify_rate()
                self._probe = None
                while not self._stop.is_set():
                    with self._lock:
                        if self._check_locked(time.monotonic()):
                            self._ready.set()
                    await asyncio.sleep(min(self.stale_s / 2, .01))
            finally:
                for client in clients.values():
                    if client.is_connected:
                        try:
                            await asyncio.wait_for(client.write_gatt_char(
                                CMD, STOP, response=True), timeout=2)
                        except Exception:
                            pass

    def _thread_main(self):
        async def run():
            task = asyncio.create_task(self._run())
            try:
                while not task.done() and not self._stop.is_set():
                    await asyncio.sleep(.02)
                if self._stop.is_set() and not task.done():
                    task.cancel()
                await task
            except asyncio.CancelledError:
                pass
        try:
            asyncio.run(run())
        except Exception as exc:
            self.error = exc
        finally:
            with self._lock:
                self._connected = dict.fromkeys(self._connected, False)
                if not self._stop.is_set():
                    self._check_locked(time.monotonic())
            self._ready.set()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=8)
            if self._thread.is_alive():
                print("[Muse] WARNING: BLE shutdown did not finish within 8 seconds")


class _RecordingMuseIMUs(MuseIMUs):
    """MuseIMUs that logs every record's arrival for main()'s timing report.

    Only the standalone script uses this, so the controller's receive path
    stays as it is.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.recording = False
        self.arrivals = {side: array("d") for side in self.devices}   # host monotonic s
        self.stamps = {side: array("d") for side in self.devices}     # device ms

    def _receive(self, side, packet, scales):
        if self.recording and len(packet) > 8 and not (len(packet) - 8) % RECORD.size:
            now = time.monotonic()
            # Every record, not just the newest: a packet may batch several.
            for offset in range(8 + 18, len(packet), RECORD.size):
                self.arrivals[side].append(now)
                self.stamps[side].append(int.from_bytes(packet[offset:offset + 6], "little"))
        super()._receive(side, packet, scales)


def _timing_report(side, arrivals, stamps, period_ms):
    """Print latency, late and dropped records for one side."""
    unique = sorted(set(stamps))
    if len(unique) < 2:
        print(f"[Muse] {side}: too few records for a timing report")
        return
    # Dropped: device timestamps missing between the first and last record.
    steps = [max(1, round((b - a) / period_ms)) for a, b in zip(unique, unique[1:])]
    expected = sum(steps) + 1
    dropped = expected - len(unique)
    # Latency: the IMU clock is not synchronized to the host, so fit out the
    # clock offset and drift, then measure each record's delay above the
    # fastest record's. Absolute radio latency cannot be measured this way.
    slope, intercept = statistics.linear_regression(stamps, arrivals)
    delays = [t - (intercept + slope * stamp) for stamp, t in zip(stamps, arrivals)]
    fastest = min(delays)
    delays = sorted((d - fastest) * 1000 for d in delays)
    late = sum(d > period_ms for d in delays)
    packets = sorted(set(arrivals))
    gap = max((b - a) * 1000 for a, b in zip(packets, packets[1:])) if len(packets) > 1 else 0
    print(f"  latency  avg {statistics.fmean(delays):.2f} ms  "
          f"p99 {delays[int(0.99 * (len(delays) - 1))]:.2f} ms  max {delays[-1]:.2f} ms  "
          f"(above the fastest record)")
    print(f"  late     {late} of {len(delays)} records > {period_ms:g} ms "
          f"({100 * late / len(delays):.2f} %)")
    print(f"  dropped  {dropped} of {expected} records never arrived "
          f"({100 * dropped / expected:.2f} %), longest run {max(steps) - 1}")
    print(f"  arrival  longest gap between packets {gap:.1f} ms")


def main():
    """Stream both IMUs with the controller's config, print live readings,
    then a timing report (latency, late and dropped records) on exit."""
    import argparse
    import collections
    import os

    import yaml

    default_cfg = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir,
        "cfg", "impedance_rl_hip.yaml"))
    parser = argparse.ArgumentParser(
        description="Stream both Muse IMUs without the controller and print live readings.")
    parser.add_argument("config", nargs="?", default=default_cfg,
                        help="YAML config with fs and muse_devices (default: %(default)s)")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="seconds to stream; 0 streams until Ctrl+C (default)")
    parser.add_argument("--print-hz", type=float, default=10.0,
                        help="status lines per second (default: %(default)s)")
    args = parser.parse_args()
    if not args.print_hz > 0:
        parser.error("--print-hz must be positive")
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    imus = _RecordingMuseIMUs(cfg)
    imus.start()                   # closes itself and raises if startup fails
    imus.arm()                     # stale/disconnect faults behave as in the controller
    imus.recording = True
    print(f"[Muse] Streaming at {imus.frequency} Hz; press Ctrl+C to stop")
    t0 = next_print = time.monotonic()
    start_counts = imus.valid_packets.copy()
    start_invalid = imus.invalid_packets.copy()
    # Rates span about one second, so BLE batching does not make them jump.
    history = collections.deque([(t0, start_counts)],
                                maxlen=max(2, round(args.print_hz) + 1))
    try:
        while not args.duration or time.monotonic() - t0 < args.duration:
            next_print += 1.0 / args.print_hz
            time.sleep(max(0.0, next_print - time.monotonic()))
            if imus.error:
                raise RuntimeError(f"Muse stream stopped: {imus.error!r}")
            samples, ages, valid = imus.snapshot()
            now = time.monotonic()
            counts = imus.valid_packets.copy()
            then, old = history[0]
            history.append((now, counts))
            parts = []
            for side, sample in samples.items():
                if sample is None:
                    parts.append(f"{side[0].upper()}: no data")
                    continue
                gx, gy, gz = sample.imu[3:]
                hz = (counts[side] - old[side]) / (now - then)
                parts.append(f"{side[0].upper()}: angle {math.degrees(sample.thigh_angle):+7.2f} deg"
                             f"  gyro {gx:+7.1f} {gy:+7.1f} {gz:+7.1f} deg/s"
                             f"  age {ages[side] * 1000:5.1f} ms  {hz:5.1f} Hz")
            print(f"{now - t0:7.2f} s  " + "  |  ".join(parts) +
                  f"  |  {'OK' if valid else 'STALE'}")
    except KeyboardInterrupt:
        print()
    finally:
        elapsed = time.monotonic() - t0
        imus.recording = False
        counts = {side: n - start_counts[side] for side, n in imus.valid_packets.items()}
        invalid = {side: n - start_invalid[side] for side, n in imus.invalid_packets.items()}
        imus.close()
    period_ms = 1000.0 / imus.frequency
    print(f"\n[Muse] Timing report; late = more than one {period_ms:g} ms period "
          f"behind the fastest record (host and IMU clocks are not synchronized)")
    for side, count in counts.items():
        print(f"[Muse] {side}: {count} valid packets in {elapsed:.1f} s "
              f"({count / elapsed:.1f} Hz), {invalid[side]} invalid")
        _timing_report(side, imus.arrivals[side], imus.stamps[side], period_ms)


if __name__ == "__main__":
    main()
