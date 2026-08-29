# 时间序列探针：展示臂远端关节的高频自激(对照：膝关节干净)。
# 打印连续 20 个策略步(每步20ms)的 q̇ 与 q，右列为差分符号翻转计数。

import isaacgym  # noqa: F401
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry

import torch


def probe(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 5
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.commands.ranges.lin_vel_x = [1.0, 1.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg)

    idx = {n: i for i, n in enumerate(env.dof_names)}
    cols = [('左腕roll', 'left_wrist_roll_joint'), ('左肩yaw', 'left_shoulder_yaw_joint'),
            ('左肘', 'left_elbow_joint'), ('左踝roll', 'left_ankle_roll_joint'),
            ('左踝pitch', 'left_ankle_pitch_joint'), ('左膝(对照)', 'left_knee_joint')]
    ids = [idx[n] for _, n in cols]
    env0 = 0

    obs = env.get_observations()
    q_rec, v_rec = [], []
    for _ in range(150):
        actions = ppo_runner.alg.actor_critic.act_inference(obs.detach())
        obs, _, _, dones, _ = env.step(actions.detach())
        ppo_runner.alg.actor_critic.memory_a.reset(dones)
        q_rec.append(env.dof_pos[env0, ids].cpu())
        v_rec.append(env.dof_vel[env0, ids].cpu())

    Q = torch.stack(q_rec); V = torch.stack(v_rec)
    print("\n连续 20 个策略步(每步 20ms,共 0.4s)——env0 行走中:\n")
    for k, (label, _) in enumerate(cols):
        v = V[100:120, k]; q = Q[100:120, k]
        flips = int((torch.sign(v[1:]) != torch.sign(v[:-1])).sum())
        print(f"--- {label}  (符号翻转 {flips}/19) ---")
        print("  q̇:", ' '.join(f"{x:+7.1f}" for x in v.tolist()))
        print("  q :", ' '.join(f"{x:+7.3f}" for x in q.tolist()))
        print()


if __name__ == '__main__':
    probe(get_args())
