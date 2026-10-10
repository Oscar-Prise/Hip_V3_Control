# AMP-Hip: Bilateral Hip Exoskeleton, DOFC

This repository runs the bilateral hip exoskeleton from a Jetson with Delayed
Output Feedback Control (DOFC). A 100 Hz loop reads both hip motor encoders,
computes the DOFC torque, commands both hips over CAN, streams live plots, and
saves every trial to `logged_data/`. No IMUs and no trained model are used.

> **Safety first.** Always do a first run with nobody wearing the device, start
> new subjects at a low `kappa` / `scale`, and keep a hand on Ctrl+C, which
> disables both motors and saves the data.

## 1. Hardware

| Part | Details |
|---|---|
| Computer | NVIDIA Jetson running JetPack (provides `Jetson.GPIO`) |
| Motors | Two TMotor AK80-9 on SocketCAN `can0`, 1 Mbit/s. Left = CAN id 3, right = CAN id 2 |
| Sync (optional) | GPIO pin 7 (board numbering) pulses at trial start and end for motion capture |

## 2. Repository layout

```
main_hip.py                  Entry point: 100 Hz loop, telemetry, GPIO, logging
cfg/dofc_hip.yaml            Trial + DOFC config
controllers/
  dofc_hip.py                Delayed output feedback control
  base.py                    Shared data types (Sensors, CtrlResult)
actuators/                   TMotor CAN driver
utils/
  exo_hardware.py            Motors behind one read/command interface
  teleplot.py                Live plotting over UDP
  Header_Mocap_trigger.py    Motion-capture start trigger client
  utils.py                   Loop timing (RateKeeper)
logged_data/                 Trial logs (<trial_name>.npz)
```

## 3. One-time setup

```bash
python -m pip install --user -r requirements.txt
```

`Jetson.GPIO` comes with JetPack. At startup the motor driver resets and brings
up `can0` with `sudo modprobe` / `sudo ip link`, so run from a terminal where
you can enter your sudo password.

## 4. Running a trial

Edit `cfg/dofc_hip.yaml`, then:

```bash
python main_hip.py cfg/dofc_hip.yaml
```

1. **Config check.** Key values are printed; press Enter.
2. **Neutral pose.** The wearer stands upright and still; press Enter. Both
   encoders are zeroed and the motors are enabled. Every hip angle is measured
   from this pose.
3. **Start.** `trigger_type: typing` waits for Enter; `mocap` waits for the
   mocap PC to send `exo on`.
4. **Stop** with **Ctrl+C**. Both motors go to zero torque and are disabled,
   and the log is saved to `logged_data/<trial_name>.npz`. The loop does not
   stop at `exp_time_sec`; only logging stops.

Open Teleplot (VS Code extension, UDP port 47269, set in `utils/teleplot.py`)
to watch `torque_cmd`, `encoder_position` (flexion-positive, deg), `dofc`
(`y` and `y_delayed`), `GPIO` and `loop_time`.

## 5. Control law

```
theta_r, theta_l = flexion-positive hip angles (low-pass, lpf_tau)
y(t)   = sin(theta_r) - sin(theta_l)
tau(t) = clip(kappa * y(t - delay), ±hip_max_torque)
right  = -tau,  left = +tau          (then torque_sign_* per motor)
cmd    = clip(scale * tau, ±torque_limit)
```

`kappa` sets the magnitude and `delay` sets the timing. Start with a small
`kappa` and raise it while watching the wearer's cadence.

## 6. Config reference

| Key | Meaning |
|---|---|
| `trial_name` | Log file name. **Change it every trial**: an existing file is overwritten without warning. |
| `exo_on` | `false` runs everything but sends zero torque. |
| `trigger_type` | `typing` or `mocap`. |
| `side` | `both`, `left` or `right`. |
| `scale` | Final gain on the controller output. |
| `torque_limit` | Hard clamp in Nm, applied last. |
| `exp_time_sec` | Seconds of data logged; the end-of-trial GPIO pulse fires here. |
| `fs` | Loop rate in Hz. |
| `kappa` | Nm per unit of `sin(hip_r) - sin(hip_l)`. |
| `delay` | Feedback delay in seconds. |
| `lpf_tau` | Hip-angle low-pass time constant (s). |
| `hip_max_torque` | Internal DOFC clamp (Nm), before `scale` and `torque_limit`. |
| `hip_sign_r/l`, `torque_sign_r/l` | Map each motor's frame to flexion-positive and back. Left motor is mounted reversed (`-1`). |
| `hip_offset_deg_r/l` | Encoder zero offset (deg). |
| `can_channel`, `can_id_L`, `can_id_R`, `motor_type` | CAN interface, motor IDs, motor model. |
| `GPIO_OUTPUT_PIN`, `PULSE_WIDTH_SEC`, `GPIO_START_DELAY_SEC` | Sync pulse pin and timing. |
| `mocap_server_ip`, `mocap_server_port` | Used with `trigger_type: mocap`. |

## 7. Signs: check on a new setup

With nobody wearing the device and a low `scale`, move one hip at a time by
hand: `encoder_position` in Teleplot should go positive for flexion on the same
side. Confirm positive torque drives each hip toward flexion. If a side is
reversed, flip that side's `hip_sign_*` / `torque_sign_*`.

## 8. Logged data

`logged_data/<trial_name>.npz`, one row per loop step:

| Signals | Meaning |
|---|---|
| `time` | Seconds since start |
| `hip_angle_r`, `hip_angle_l` | Filtered, flexion-positive hip angles (rad) |
| `enc_pos_*`, `enc_vel_*` | Raw motor encoders (deg, deg/s), each motor's own frame |
| `dofc_y`, `dofc_y_delayed` | DOFC signal now and after the delay |
| `cmd_R`, `cmd_L` | Torque sent to the motors (Nm) |
| `GPIO` | Sync pin state |
