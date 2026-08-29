# MuJoCo 臂部探针：量化不同 dt 下臂关节行为（锁死/跟随/自激抖动），与 gym 数据对照
# 用法: python deploy/deploy_mujoco/probe_mujoco_arms.py 0.002 10
import sys

import mujoco
import numpy as np
import torch
import yaml

from legged_gym import LEGGED_GYM_ROOT_DIR


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    g = np.zeros(3)
    g[0] = 2 * (-qz * qx + qw * qy)
    g[1] = -2 * (qz * qy + qw * qx)
    g[2] = 1 - 2 * (qw * qw + qz * qz)
    return g


def main():
    sim_dt, decim = float(sys.argv[1]), int(sys.argv[2])
    with open(f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/g1.yaml") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    policy_path = cfg["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
    xml_path = cfg["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    lim = np.array(cfg["effort_limits"], dtype=np.float32)
    default = np.array(cfg["default_angles"], dtype=np.float32)
    n_a, n_o = cfg["num_actions"], cfg["num_obs"]
    cmd = np.array(cfg["cmd_init"], dtype=np.float32)
    period = cfg["gait_period"]
    h_target, h_scale = cfg["base_height_target"], cfg["height_obs_scale"]

    m = mujoco.MjModel.from_xml_path(xml_path)
    m.opt.timestep = sim_dt
    d = mujoco.MjData(m)
    d.qpos[7:] = default
    d.qpos[2] = cfg["init_height"]
    mujoco.mj_forward(m, d)
    jname = lambda i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1)

    # β=Kd·dt/I_eff（diag(M)，其余关节锁定）；β>=2 时显式阻尼自激
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    eff_I = np.diag(M)[6:]
    mode = sys.argv[3] if len(sys.argv) > 3 else 'explicit'
    noarm = mode == 'noarm'
    implicit = mode == 'implicit'
    if implicit:
        # 阻尼交给求解器隐式积分（MuJoCo Euler 默认隐式处理 dof_damping）
        m.dof_damping[6:] = kds
    print(f"\n===== dt={sim_dt}  decim={decim}  (物理{1/sim_dt:.0f}Hz, 策略50Hz)  模式={mode} =====")
    print(f"{'关节':<30}{'I_eff':>9}{'β':>7}  状态")
    for i in range(23):
        beta = kds[i] * sim_dt / eff_I[i]
        print(f"{jname(i):<30}{eff_I[i]:9.2e}{beta:7.2f}  {'自激(不稳定)' if beta >= 2 else '稳定'}")

    watch = [i for i in range(23) if any(s in jname(i) for s in
             ('wrist', 'shoulder_yaw', 'elbow'))] + [3]
    policy = torch.jit.load(policy_path)
    action = np.zeros(n_a, dtype=np.float32)
    target = default.copy()
    obs = np.zeros(n_o, dtype=np.float32)
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)

    T = 4.0
    rec = {i: {'q': [], 'v': [], 't': [], 'vmax': []} for i in watch}
    fall_t, counter = None, 0
    wmax = np.zeros(23)
    n_steps = int(T / sim_dt)
    for _ in range(n_steps):
        tau = (target - d.qpos[7:]) * kps
        if not implicit:
            tau -= d.qvel[6:] * kds
        tau = np.clip(tau, -lim, lim)
        if noarm:
            tau[13:] = 0.0
        d.ctrl[:] = tau
        if not np.isfinite(d.qacc).all():
            print(f"\n!!! 数值爆炸(非有限 qacc) @ t={counter * sim_dt:.3f}s")
            break
        mujoco.mj_step(m, d)
        wmax = np.maximum(wmax, np.abs(d.qvel[6:]))
        counter += 1
        if counter % decim == 0:
            t = counter * sim_dt
            if fall_t is None and d.qpos[2] < 0.55:
                fall_t = t
            if fall_t is None:
                for i in watch:
                    rec[i]['q'].append(d.qpos[7 + i])
                    rec[i]['v'].append(d.qvel[6 + i])
                    rec[i]['t'].append(target[i])
                    rec[i]['vmax'].append(wmax[i])
                wmax = np.zeros(23)
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

    print(f"\n摔倒时刻: {fall_t if fall_t else f'>{T}s 未倒'}   "
          f"(gym 对照: 腕 q̇ 读数≈+41.7 恒定, 关节位移 0.02~0.19 rad, 目标摆动 0.4~1.1 rad)")
    print(f"{'关节':<30}{'q范围':>16}{'目标范围':>16}{'读|q̇|均值':>10}{'子步|q̇|max':>11}")
    for i in watch:
        r = rec[i]
        q, tg, v, vm = map(np.array, (r['q'], r['t'], r['v'], r['vmax']))
        print(f"{jname(i):<30}[{q.min():+.2f},{q.max():+.2f}]"
              f"[{tg.min():+.2f},{tg.max():+.2f}]"
              f"{np.abs(v).mean():10.1f}{vm.max():11.1f}")


if __name__ == '__main__':
    main()
