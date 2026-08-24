# 测量脚本：固定 vx 指令，统计 pelvis 高度与膝角（支撑/摆动腿分开）。
# 用途：区分"支撑腿蹲"（pelvis 明显低于 0.78 且支撑膝深）与"摆动腿抬"（仅摆动膝深）。
# 用法：
#   python legged_gym/scripts/measure_height.py --task=g1_23dof --headless \
#       --load_run=Aug21_23-23-36_ --checkpoint=13550
#   方向用环境变量 VEL（+1 前走 / -1 后走）：VEL=-1 python ...

import isaacgym  # noqa: F401  必须最先 import
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch


def measure(args, vel):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 50
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.commands.ranges.lin_vel_x = [vel, vel]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)

    idx = {n: i for i, n in enumerate(env.dof_names)}
    knee = [idx['left_knee_joint'], idx['right_knee_joint']]
    hip = [idx['left_hip_pitch_joint'], idx['right_hip_pitch_joint']]

    z_rec, k_rec, hip_rec, ct_rec, vx_rec, mask_rec = [], [], [], [], [], []
    total = 1500                        # 30s 仿真
    for step in range(total):
        actions = ppo_runner.alg.actor_critic.act(obs.detach())   # σ 采样（训练同款）
        obs, _, rews, dones, infos = env.step(actions.detach())
        ppo_runner.alg.actor_critic.memory_a.reset(dones)

        ct_rec.append(torch.norm(env.contact_forces[:, env.feet_indices, :3],
                                 dim=2).cpu() > 1.)
        z_rec.append(env.root_states[:, 2].cpu())
        k_rec.append(env.dof_pos[:, knee].cpu())
        hip_rec.append(env.dof_pos[:, hip].cpu())
        vx_rec.append(env.base_lin_vel[:, 0].cpu())
        mask_rec.append((env.episode_length_buf > 50).cpu())       # 丢弃复位后 1s 瞬态

        if step % 500 == 0:
            print(f"step {step}/{total}", flush=True)

    mask = torch.stack(mask_rec)
    Z = torch.stack(z_rec)[mask]                      # [S]
    K = torch.stack(k_rec)[mask]                      # [S, 2]
    H = torch.stack(hip_rec)[mask]
    C = torch.stack(ct_rec)[mask]
    vx = torch.cat(vx_rec)[torch.cat(mask_rec)]

    q = lambda t, p: torch.quantile(t.float(), p).item()

    target = env_cfg.rewards.base_height_target
    print("\n================= 测量结果 =================")
    print(f"指令 vx = {vel:+.2f} m/s   实际平均 vx = {vx.mean():+.3f} m/s")
    print(f"pelvis 高度: 均值={Z.mean():.3f}  std={Z.std():.3f}  "
          f"5%={q(Z,.05):.3f}  中位={q(Z,.5):.3f}  [m]")
    print(f"  → 相对目标 {target}: 平均低 {target - Z.mean():+.3f} m")
    print(f"支撑膝(触地腿): 均值={K[C].mean():.2f} rad ({np.degrees(K[C].mean()):.0f}°)  "
          f"95%={q(K[C],.95):.2f}   ← 蹲的程度（默认 0.3）")
    print(f"摆动膝(离地腿): 均值={K[~C].mean():.2f} rad ({np.degrees(K[~C].mean()):.0f}°)  "
          f"95%={q(K[~C],.95):.2f}   ← 抬腿的程度")
    print(f"hip_pitch:     均值={H.mean():+.2f} rad ({np.degrees(H.mean()):+.0f}°)")


if __name__ == '__main__':
    import os
    VEL = float(os.environ.get('VEL', 1.0))   # +1 前走 / -1 后走
    measure(get_args(), VEL)
