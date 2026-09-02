import math
import time

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class GamepadController:
    """Lightweight pygame gamepad -> velocity-command reader.

    Maps the left stick to forward/lateral velocity and the right stick X to yaw.
    The returned cmd is in physical units [vx m/s, vy m/s, yaw rad/s], matching the
    units of `cmd_init`, so it drops straight into `obs[6:9] = cmd * cmd_scale`.

    Direction is controlled per-axis by `sign_ly/lx/rx` (+1 or -1), NOT by negating
    the axis index (a negative index is invalid to pygame and would crash get_axis).
    Standard gamepads read up & left as negative, so the defaults (-1) make
    "up = forward", "left = strafe/turn left".

    Falls back gracefully (active=False) when disabled in config, when pygame is not
    installed, or when no joystick is connected -- the caller then keeps using its
    own static cmd (e.g. cmd_init).
    """

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        self.deadzone = float(cfg.get("deadzone", 0.15))
        # max physical command magnitudes [vx m/s, vy m/s, yaw rad/s]
        self.max_cmd = np.array(cfg.get("max_cmd", [0.8, 0.5, 1.57]), dtype=np.float32)
        # pygame/SDL axis indices (typical Xbox-style / Logitech F710 layout:
        # left stick X=0, Y=1; right stick X=3). Must be non-negative ints < numaxes;
        # abs() guards against a stray negative sign crashing get_axis().
        self.ax_ly = abs(int(cfg.get("axis_ly", 1)))
        self.ax_lx = abs(int(cfg.get("axis_lx", 0)))
        self.ax_rx = abs(int(cfg.get("axis_rx", 3)))
        # per-axis sign (+1 / -1). Standard gamepads read up & left as negative, so
        # -1 makes "up = forward", "left = strafe/turn left". Flip to +1 to invert.
        self.sign_ly = int(cfg.get("sign_ly", -1))
        self.sign_lx = int(cfg.get("sign_lx", -1))
        self.sign_rx = int(cfg.get("sign_rx", -1))
        self.cmd = np.zeros(3, dtype=np.float32)
        self.active = False
        self._js = None
        self._pg = None
        self._numaxes = 0

        if not self.enabled:
            return
        try:
            import os
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() <= 0:
                print("[gamepad] enabled but no joystick found -> using static cmd")
                return
            self._js = pygame.joystick.Joystick(0)
            self._js.init()
            self._pg = pygame
            self._numaxes = self._js.get_numaxes()
            self.active = True
            print(f"[gamepad] connected: {self._js.get_name()} "
                  f"({self._numaxes} axes, {self._js.get_numbuttons()} buttons)")
            print(f"[gamepad] axes -> cmd: ly={self.ax_ly}(vx), lx={self.ax_lx}(vy), "
                  f"rx={self.ax_rx}(yaw); signs=({self.sign_ly},{self.sign_lx},{self.sign_rx}); "
                  f"max_cmd={self.max_cmd.tolist()}")
            for nm, idx in (("ly", self.ax_ly), ("lx", self.ax_lx), ("rx", self.ax_rx)):
                if not 0 <= idx < self._numaxes:
                    print(f"[gamepad] WARNING: axis_{nm}={idx} out of range "
                          f"[0,{self._numaxes}); that axis will read 0")
        except Exception as e:
            print(f"[gamepad] init failed ({e}) -> using static cmd")

    def _deadzone(self, v: float) -> float:
        if abs(v) < self.deadzone:
            return 0.0
        # rescale so the output rises from 0 just past the deadzone (no step jump)
        return (v - math.copysign(self.deadzone, v)) / (1.0 - self.deadzone)

    def _axis(self, idx: int) -> float:
        """Safely read an axis value; returns 0.0 if idx is out of range
        (prevents `pygame.error: Invalid joystick axis` on a misconfigured index)."""
        if 0 <= idx < self._numaxes:
            return self._js.get_axis(idx)
        return 0.0

    def update(self) -> np.ndarray:
        """Poll the gamepad and return the current cmd [vx, vy, yaw] in m/s, rad/s."""
        if not self.active or self._js is None:
            return self.cmd
        self._pg.event.pump()
        ly = self._deadzone(self._axis(self.ax_ly)) * self.sign_ly
        lx = self._deadzone(self._axis(self.ax_lx)) * self.sign_lx
        rx = self._deadzone(self._axis(self.ax_rx)) * self.sign_rx
        stick = np.array([ly, lx, rx], dtype=np.float32)
        self.cmd = stick * self.max_cmd
        return self.cmd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        # URDF effort 限幅（=训练 torque clip）；缺省 None 则不限幅（旧行为）
        effort_limits = np.array(config["effort_limits"], dtype=np.float32) \
            if "effort_limits" in config else None

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        # Optional gamepad: overrides cmd every control step when active.
        # When not active (disabled / no joystick), cmd stays at cmd_init.
        gamepad = GamepadController(config.get("gamepad", {}))

        # g1_23dof 扩展键：高度观测通道 / 步态周期 / 初始高度。老配置(h1 等)缺省则保持旧行为
        base_height_target = config.get("base_height_target", None)
        height_obs_scale = config.get("height_obs_scale", 5.0)
        gait_period = config.get("gait_period", 0.8)
        init_height = config.get("init_height", None)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    # Match G1Robot23dof's phase-hold behavior: once a stop command arrives,
    # finish the current gait cycle and hold phase at zero.
    phase_hold = 0.0

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # 初始姿态对齐训练 default_joint_angles（XML 零位=手臂前平举，开局偏差过大）
    d.qpos[7:] = default_angles
    if init_height is not None:
        d.qpos[2] = init_height
    mujoco.mj_forward(m, d)

    # load policy
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            if effort_limits is not None:
                tau = np.clip(tau, -effort_limits, effort_limits)
            d.ctrl[:] = tau
            # Apply mouse perturbation to the robot: in the passive viewer, double-clicking
            # a body + dragging only updates the visual `perturb` object unless we copy its
            # force into d.xfrc_applied here. This is what makes push-recovery / robustness
            mujoco.mjv_applyPerturbForce(m, d, viewer.perturb)
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # Apply control signal here.

                # poll the gamepad (if active) to refresh the velocity command
                if gamepad.active:
                    cmd = gamepad.update()

                # create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                period = gait_period
                policy_dt = simulation_dt * control_decimation
                next_phase = phase_hold + policy_dt / period
                moving = (
                    np.linalg.norm(cmd[:2]) >= 0.2 or
                    abs(float(cmd[2])) >= 0.15
                )
                if moving:
                    phase_hold = next_phase % 1.0
                elif phase_hold > 0.0:
                    phase_hold = 0.0 if next_phase >= 1.0 else next_phase
                phase = phase_hold
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                # 高度观测(若有)：训练布局为 actions 后接 (z-target)*scale，再接 sin/cos 相位
                idx = 9 + 3 * num_actions
                if base_height_target is not None:
                    obs[idx] = (d.qpos[2] - base_height_target) * height_obs_scale
                    idx += 1
                obs[idx : idx + 2] = np.array([sin_phase, cos_phase])
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
