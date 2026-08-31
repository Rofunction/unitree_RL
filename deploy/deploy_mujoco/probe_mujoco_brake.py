# MuJoCo 刹车探针：cmd 满速→瞬切 0，量化减速曲线/高度下沉/碎步程度，与 gym 对照
# 用法: python probe_mujoco_brake.py [policy_path]   (缺省用 g1.yaml 的 policy_path)
import sys

import mujoco
import numpy as np
import torch
import yaml

from legged_gym import LEGGED_GYM_ROOT_DIR

# gym 实测(model_4450, cmd 1.0→0): 供对照
GYM_VX = [0.97, 0.93, 0.64, 0.49, 0.27, 0.11, 0.10, 0.01, 0.01, 0.02]
GYM_HMIN, GYM_STAND_RMS = 0.771, (0.75, 0.98)  # 高度min / 站定时腿q̇RMS区间


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def run_phase(m, cfg, policy, cmd_run, axis, label):
    dt, decim = cfg["simulation_dt"], cfg["control_decimation"]
    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    lim = np.array(cfg["effort_limits"], dtype=np.float32)
    default = np.array(cfg["default_angles"], dtype=np.float32)
    n_a, n_o = cfg["num_actions"], cfg["num_obs"]
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    h_target, h_scale = cfg["base_height_target"], cfg["height_obs_scale"]
    period = cfg["gait_period"]
    T_RUN, T_BRAKE = 3.0, 2.5

    d = mujoco.MjData(m)
    d.qpos[7:] = default
    d.qpos[2] = cfg["init_height"]
    mujoco.mj_forward(m, d)

    obs = np.zeros(n_o, dtype=np.float32)
    action = np.zeros(n_a, dtype=np.float32)
    target = default.copy()
    cmd = np.zeros(3, dtype=np.float32)
    rec, flips, pre_v = [], [0, 0], []
    fall_t, counter = None, 0
    prev_hip_v = None
    n_steps = int((T_RUN + T_BRAKE) / dt)
    for _ in range(n_steps):
        t = counter * dt
        cmd[:] = cmd_run if t < T_RUN else 0.0
        tau = (target - d.qpos[7:]) * kps - d.qvel[6:] * kds
        tau = np.clip(tau, -lim, lim)
        d.ctrl[:] = tau
        if not np.isfinite(d.qacc).all():
            print(f"  !!! 数值爆炸 @ t={t:.2f}s")
            break
        mujoco.mj_step(m, d)
        counter += 1
        if counter % decim:
            continue
        t = counter * dt
        if fall_t is None and d.qpos[2] < 0.55:
            fall_t = t
        if fall_t:
            break
        if T_RUN - 0.5 <= t < T_RUN:
            pre_v.append(d.qvel[axis])  # 切换前0.5s速度sanity
        hip_v = d.qvel[6]  # 左hip_pitch q̇(双腿反相,取单腿)
        if prev_hip_v is not None and np.sign(hip_v) != np.sign(prev_hip_v):
            flips[0 if t < T_RUN else 1] += 1
        prev_hip_v = hip_v
        if t >= T_RUN:
            legv = d.qvel[6:18]
            rec.append((t - T_RUN, d.qvel[axis], d.qpos[2],
                        float(np.sqrt((legv ** 2).mean()))))
        # --- 策略步 ---
        qj = (d.qpos[7:] - default).astype(np.float32)
        obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]
        obs[3:6] = get_gravity_orientation(d.qpos[3:7])
        obs[6:9] = cmd * cmd_scale
        obs[9:9 + n_a] = qj
        obs[9 + n_a:9 + 2 * n_a] = d.qvel[6:] * cfg["dof_vel_scale"]
        obs[9 + 2 * n_a:9 + 3 * n_a] = action
        obs[9 + 3 * n_a] = (d.qpos[2] - h_target) * h_scale
        phase = (t % period) / period
        obs[10 + 3 * n_a:] = np.array([np.sin(2 * np.pi * phase),
                                       np.cos(2 * np.pi * phase)], dtype=np.float32)
        action = policy(torch.from_numpy(obs).unsqueeze(0)).detach().numpy().squeeze()
        target = action * cfg["action_scale"] + default

    arr = np.array(rec) if rec else np.zeros((1, 4))
    print(f"\n===== {label}: cmd={cmd_run} → 0 @ t=3s =====")
    print(f"  切换前0.5s均速: {np.mean(pre_v):+.2f} m/s" if len(pre_v) > 2 else "  (未达稳态?)")
    print("  t(s)  : " + " ".join("%5.2f" % arr[i, 0] for i in range(0, len(arr), 10)))
    print("  v     : " + " ".join("%+5.2f" % arr[i, 1] for i in range(0, len(arr), 10)))
    print("  gym对照: " + " ".join("%+5.2f" % v for v in GYM_VX))
    print("  高度min: %.3f (gym %.3f)   摔倒: %s" % (
        arr[:, 2].min(), GYM_HMIN, f"t={fall_t:.2f}s" if fall_t else "无"))
    print("  hip_pitch q̇符号翻转率: 行走 %.1f 次/s | 刹车 %.1f 次/s (碎步=高频小步)" % (
        flips[0] / 3.0, flips[1] / max(arr[-1, 0], 0.1)))
    tail = arr[arr[:, 0] > arr[-1, 0] - 1.0]
    print("  停稳后1s: 速度 %+.3f m/s, 腿q̇RMS %.2f (gym站定 %.2f~%.2f)" % (
        tail[:, 1].mean(), tail[:, 3].mean(), *GYM_STAND_RMS))


def main():
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/g1.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    p = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    if len(sys.argv) > 1:
        p = sys.argv[1]
    print(f"policy: {p}")
    xml = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    m = mujoco.MjModel.from_xml_path(xml)
    m.opt.timestep = cfg["simulation_dt"]
    policy = torch.jit.load(p)
    run_phase(m, cfg, policy, np.array([1.0, 0, 0]), 0, "前进刹车 vx")
    run_phase(m, cfg, policy, np.array([0, 0.5, 0]), 1, "侧移刹车 vy")


if __name__ == "__main__":
    main()
