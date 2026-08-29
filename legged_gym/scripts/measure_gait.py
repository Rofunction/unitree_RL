
# 守门测量：固定 vx 指令，同口径输出 前后脚错开/躯干倾角/支撑膝分布/身高/跟踪。
# 用法:
#   python legged_gym/scripts/measure_gait.py --task=g1_23dof --headless \
#       --load_run=<run> --checkpoint=<ckpt> [--vel=0.5]
import sys

import isaacgym  # noqa: F401 必须最先 import
from isaacgym.torch_utils import quat_rotate_inverse
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch


def measure(args):
    vel = getattr(args, 'vel', 0.5)
    stance = getattr(args, 'stance', 0.60)   # g1=0.55, g1_23dof=0.60

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
    env_cfg.env.test = True

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    N = env.num_envs
    knee_i = [env.dof_names.index('left_knee_joint'),
              env.dof_names.index('right_knee_joint')]
    warmup, total = 150, 1300
    dx_s, gx_s, bz_s, vx_s = [], [], [], []
    stance_kn, swing_kn, resets = [], [], 0

    for step in range(total):
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        if hasattr(ppo_runner.alg.actor_critic, 'memory_a'):
            ppo_runner.alg.actor_critic.memory_a.reset(dones)
        resets += int((dones & ~env.time_out_buf).sum())   # 只数真实终止，不含超时

        foot_pos = env.feet_pos - env.root_states[:, 0:3].unsqueeze(1)
        quat = env.base_quat.unsqueeze(1).repeat(1, 2, 1)
        foot_rel = quat_rotate_inverse(quat.reshape(-1, 4),
                                       foot_pos.reshape(-1, 3)).reshape(N, 2, 3)
        dx_s.append((foot_rel[:, 1, 0] - foot_rel[:, 0, 0]).cpu())   # 右-左
        gx_s.append(env.projected_gravity[:, 0].cpu())               # 正=前倾
        bz_s.append(env.root_states[:, 2].cpu())
        vx_s.append(env.base_lin_vel[:, 0].cpu())

        contact = torch.norm(env.contact_forces[:, env.feet_indices, :3], dim=2) > 1.
        ph = env.leg_phase < stance
        kn = env.dof_pos[:, knee_i]
        stance_kn.append(kn[contact & ph].cpu())
        swing_kn.append(kn[~contact].cpu())
        if step % 300 == 0:
            print(f"step {step}/{total}", flush=True)

    dx = torch.stack([t for t in dx_s[warmup:]])                     # [T, N]
    signed_env = dx.numpy().mean(axis=0)                             # 每env时间均值
    gxx = torch.cat(gx_s[warmup:]).numpy()
    bz = torch.cat(bz_s[warmup:]).numpy()
    vxx = torch.cat(vx_s[warmup:]).numpy()
    sk = torch.cat(stance_kn).numpy()
    wk = torch.cat(swing_kn).numpy()
    p = np.percentile

    pitch_mean = np.degrees(np.arcsin(np.clip(gxx.mean(), -1, 1)))
    pitch_rms = np.degrees(np.arcsin(np.clip(np.sqrt((gxx ** 2).mean()), 0, 1)))
    print("\n================= 步态姿态守门 =================")
    print(f"指令 vx={vel:+.2f}  实际 vx 均值={vxx.mean():+.3f} ({vxx.mean()/vel*100:.0f}%)  "
          f"真实摔倒次数={resets}")
    print(f"带符号Δx: 总均值={dx.numpy().mean():+.3f}  每env均值中位={np.median(signed_env):+.3f}  "
          f"|·|>0.05 env占比={(np.abs(signed_env) > 0.05).mean()*100:.0f}%")
    print(f"躯干倾角: 均值={pitch_mean:+.1f}°  RMS={pitch_rms:.1f}°   [正=前倾,负=后仰]")
    print(f"支撑膝: p5={p(sk,5):.2f} 中位={p(sk,50):.2f} p90={p(sk,90):.2f}   "
          f"摆动膝: 中位={p(wk,50):.2f}  [rad]")
    print(f"base z 均值={bz.mean():.3f}")
    print(f"守门线: 支撑膝中位≤0.55 | 倾角均值∈[-5,+6]° | z≥0.765 | 前后脚只记录")
    print("==============================================")


if __name__ == '__main__':
    vel, stance = 0.5, 0.60
    for i, a in enumerate(sys.argv):                                 # 先剥自定义参数
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
