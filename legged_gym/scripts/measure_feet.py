
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
    stance = getattr(args, 'stance', 0.55)   # 支撑相位窗终点：g1=0.55, g1_23dof=0.60

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
    z_rec, ph_rec, ct_rec = [], [], []           # 摆动剖面：脚高/相位/接触

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
        z_rec.append(env.feet_pos[:, :, 2].cpu())
        ph_rec.append(env.leg_phase.cpu().clone())
        ct_rec.append(contact.cpu())

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

    Z = torch.stack(z_rec).numpy()
    P = torch.stack(ph_rec).numpy()
    C = torch.stack(ct_rec).numpy()
    rises, peaks, peak_fr, falls, lags, durs, plats = [], [], [], [], [], [], []
    T, N, _ = Z.shape
    for j in range(N):
        for f in range(2):
            air = ~C[:, j, f]
            k = 0
            while k < T:
                if air[k] and k >= warmup:
                    a = k
                    while k < T and air[k]:
                        k += 1
                    L = k - a
                    if L >= 8:                      # 只统计完整摆动事件(≥0.16s)
                        z = Z[a:k, j, f]
                        kp = int(z.argmax())
                        peaks.append(z[kp]); peak_fr.append(kp / L)
                        plats.append(int((z >= z[kp] - 0.01).sum()) * env.dt)
                        falls.append((L - 1 - kp) * env.dt); durs.append(L * env.dt)
                        above = np.nonzero(z >= 0.05)[0]
                        rises.append(above[0] * env.dt if len(above) else np.nan)
                        lags.append((P[a, j, f] - stance + 0.5) % 1 - 0.5)
                else:
                    k += 1
    med = lambda x: np.nanmedian(x) if len(x) else float('nan')
    print(f"支撑期脚z: 中位={np.median(Z[C]):.3f}m  空中脚z: 25%={np.percentile(Z[~C],25):.3f} "
          f"中位={np.median(Z[~C]):.3f}  → 实际抬脚量≈{np.median(Z[~C])-np.median(Z[C]):.3f}m")
    print(f"摆动事件数={len(peaks)}  摆动时长中位={med(durs):.3f}s (相位窗={(1-stance)*0.60:.3f}s)")
    print(f"离地滞后(相对相位{stance}): 中位={med(lags):+.3f} 相位 ≈ {med(lags)*0.60*1000:.0f}ms")
    print(f"升至0.05m耗时: 中位={med(rises):.3f}s  25%分位={np.nanpercentile(rises,25):.3f}s")
    print(f"峰值高度: 中位={med(peaks):.3f}m  5%分位={np.percentile(peaks,5):.3f}m")
    print(f"峰值时刻占摆动比: 中位={med(peak_fr):.2f}  下降耗时中位={med(falls):.3f}s")
    print(f"顶点平台时长(峰值-1cm内): 中位={med(plats):.3f}s  75%分位={np.percentile(plats,75):.3f}s")
    print("===========================================")
    if q(d2, .01) > 0.15 and q(dy, .01) < 0.10:
        print("→ 诊断: 2D 达标但横向很窄 = 策略用前后错开绕过了间距要求")
    elif q(dy, .01) < 0.15:
        print("→ 诊断: 2D 距离本身也不够 = 罚项压力太弱，策略在硬吃罚分")
    else:
        print("→ 诊断: 间距基本达标，问题在别处")


if __name__ == '__main__':
    vel, stance = -0.5, 0.55                      # 先剥离自定义参数再交给 get_args
    for i, a in enumerate(sys.argv):
        if a == '--vel' and i + 1 < len(sys.argv):
            vel = float(sys.argv[i + 1]); del sys.argv[i:i + 2]; break
        if a.startswith('--vel='):
            vel = float(a.split('=')[1]); del sys.argv[i]; break
    for i, a in enumerate(sys.argv):
        if a == '--stance' and i + 1 < len(sys.argv):
            stance = float(sys.argv[i + 1]); del sys.argv[i:i + 2]; break
        if a.startswith('--stance='):
            stance = float(a.split('=')[1]); del sys.argv[i]; break
    args = get_args()
    args.vel = vel
    args.stance = stance
    measure(args)
