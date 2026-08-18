
# 测量脚本：加载训练好的策略，固定 vy 指令，统计两脚间距的真实分布。
# 用途：区分"罚项太弱"（2D 距离也小）和"被 x 错开绕过"（2D 达标但横向 |Δy| 很小）。
# 用法：
#   python legged_gym/scripts/measure_feet.py --task=g1 --headless \
#       --load_run=Aug17_17-12-59_ --checkpoint=14000
#   可加 --vel=-0.5 / --vel=0.5 换方向

import isaacgym  # noqa: F401  必须最先 import
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

import numpy as np
import sys
import torch


def measure(args):
    vel = getattr(args, 'vel', -0.5)

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 50
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [vel, vel]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.env.test = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    N = env.num_envs
    warmup = 100                       # 丢掉起步瞬态
    total = 2 * int(env.max_episode_length)

    dy_all, dx_all, d2_all = [], [], []          # 双脚间距样本
    stance_y_all = []                            # 支撑脚横向偏置(相对躯干中线)
    hip_roll_all = []
    vy_track = []

    for step in range(total):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        foot_pos = env.feet_pos - env.root_states[:, 0:3].unsqueeze(1)
        quat = env.base_quat.unsqueeze(1).repeat(1, 2, 1)
        foot_rel = quat_rotate_inverse(quat.reshape(-1, 4),
                                       foot_pos.reshape(-1, 3)).reshape(N, 2, 3)
        dy_all.append((foot_rel[:, 0, 1] - foot_rel[:, 1, 1]).abs().cpu())
        dx_all.append((foot_rel[:, 0, 0] - foot_rel[:, 1, 0]).abs().cpu())
        d2_all.append(torch.norm(foot_rel[:, 0, :2] - foot_rel[:, 1, :2],
                                 dim=-1).cpu())

        contact = torch.norm(env.contact_forces[:, env.feet_indices, :3],
                             dim=2) > 1.
        stance_y = foot_rel[:, :, 1].abs() * contact          # 只统计支撑脚
        stance_y_all.append(stance_y[contact].cpu())
        hip_roll_all.append(env.dof_pos[:, [1, 7]].abs().cpu())
        vy_track.append(env.base_lin_vel[:, 1].detach().cpu())

        if step % 500 == 0:
            print(f"step {step}/{total}", flush=True)

    dy = torch.cat(dy_all)[warmup:]
    dx = torch.cat(dx_all)[warmup:]
    d2 = torch.cat(d2_all)[warmup:]
    stance_y = torch.cat(stance_y_all)
    roll = torch.cat(hip_roll_all)[warmup:]
    vyy = torch.cat(vy_track)[warmup:]

    q = lambda t, p: torch.quantile(t.float(), p).item() if t.numel() else float('nan')

    print("\n================= 测量结果 =================")
    print(f"指令 vy = {vel:+.2f} m/s   实际平均 vy = {vyy.mean():+.3f} m/s")
    print(f"横向间距 |Δy|:  min={dy.min():.3f}  1%={q(dy,.01):.3f}  "
          f"5%={q(dy,.05):.3f}  中位={q(dy,.5):.3f}   [m]")
    print(f"2D 距离(含x):   min={d2.min():.3f}  1%={q(d2,.01):.3f}  "
          f"5%={q(d2,.05):.3f}  中位={q(d2,.5):.3f}   [m]")
    print(f"前后错开 |Δx|:  均值={dx.mean():.3f}  5%={q(dx,.05):.3f}   [m]")
    print(f"支撑脚横向偏置: 均值={stance_y.mean():.3f} → 双脚站姿宽度≈"
          f"{2*stance_y.mean():.3f} m")
    print(f"hip_roll |角度|: 均值={roll.mean():.3f} rad ({np.degrees(roll.mean()):.1f}°)")
    print("===========================================")
    if q(d2, .01) > 0.15 and q(dy, .01) < 0.10:
        print("→ 诊断: 2D 达标但横向很窄 = 策略用前后错开绕过了间距要求")
    elif q(dy, .01) < 0.15:
        print("→ 诊断: 2D 距离本身也不够 = 罚项压力太弱，策略在硬吃罚分")
    else:
        print("→ 诊断: 间距基本达标，问题在别处")


if __name__ == '__main__':
    args = get_args()
    # 给 get_args 补一个自定义参数（不破坏原有解析）
    if '--vel' in sys.argv:
        args.vel = float(sys.argv[sys.argv.index('--vel') + 1])
    measure(args)
