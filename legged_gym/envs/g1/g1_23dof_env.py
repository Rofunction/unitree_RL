
from legged_gym.envs.g1.g1_env import G1Robot
from isaacgym.torch_utils import quat_rotate_inverse

import numpy as np
import torch

# 23dof：腿 12 + 腰 1 + 双臂 10。臂/腰索引按名查表不写死；
# 步态相位/接触/feet 系列奖励全部继承 12dof 版本。
# 符号均经 URDF FK 验证：肩 pitch 负=前摆，elbow 正=伸直(1.57=自然下垂)，
# 左 shoulder_roll 正=外展，waist_yaw 正=左肩向后。
class G1Robot23dof(G1Robot):
    stance_frac = 0.60     # 双支撑 20% 周期（人类水平；12dof 是 0.55）
    ARM_SWING_AMP = 0.45   # |vx|=1 时肩摆幅[rad]≈26°(人类快走上臂 ±25°)
    ELBOW_COUPLE = 0.4     # 肘联动比：肘摆幅/肩摆幅(人类 30-50%)
    ARM_SPREAD = 0.06      # 侧移臂横张幅[rad]
    WAIST_SWING = 0.10     # 腰反旋幅[rad]≈6°(人类肩-髋反扭 ~10°)
    WAIST_TURN = 0.20      # 转弯肩预旋[rad]（ωz=1 时）
    LEAN_COEF = 0.09       # |vx|=1 躯干前倾[rad]≈5°(人类快走)
    KNEE_DEADBAND = 0.50   # 支撑膝罚起征点[rad]：36000后中位稳0.58两轮不过线，起征点下移逼站直(z也受益)
    HIP_YAW_STEER = 0.35   # ωz=1 时双脚同向劈尖参考[rad]：转向需要髋yaw，实测策略仅用2.3°

    def _reward_alive(self):
        # 底薪只发给"用脚活着"：跪/坐姿脚不承重(实测触地仅18%) → alive≈0，堵死跪着领薪
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        return torch.any(contact, dim=1).float()

    def _init_buffers(self):
        super()._init_buffers()
        idx = {n: i for i, n in enumerate(self.dof_names)}
        self.knee_idx = [idx['left_knee_joint'], idx['right_knee_joint']]
        self.hip_yaw_idx = [idx['left_hip_yaw_joint'], idx['right_hip_yaw_joint']]
        # 步态姿态诊断状态：EMA(dx) 与 episode 累计量（reset_idx 上报清零）
        self.stagger_ema = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_knee_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_knee_w = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_pitch_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_dx_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_stagger_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.arm_pitch_idx = [idx['left_shoulder_pitch_joint'],
                              idx['right_shoulder_pitch_joint']]
        self.arm_pitch_default = self.default_dof_pos[:, self.arm_pitch_idx]  # [1,2]
        self.elbow_idx = [idx['left_elbow_joint'], idx['right_elbow_joint']]
        self.elbow_default = self.default_dof_pos[:, self.elbow_idx]
        self.arm_roll_idx = [idx['left_shoulder_roll_joint'],
                             idx['right_shoulder_roll_joint']]
        self.arm_roll_default = self.default_dof_pos[:, self.arm_roll_idx]
        self.arm_yaw_idx = [idx['left_shoulder_yaw_joint'],
                            idx['right_shoulder_yaw_joint']]
        self.wrist_idx = [idx['left_wrist_roll_joint'],
                          idx['right_wrist_roll_joint']]
        self.waist_idx = idx['waist_yaw_joint']
        # dof_vel 缓冲对臂roll/yaw/肘/腕8关节读冻结伪值(实测41.7rad/s vs 刚体真值1.4)
        self.leg_waist_idx = [i for n, i in idx.items()
                              if 'shoulder' not in n and 'elbow' not in n and 'wrist' not in n]

    def _feet_dx_torso(self):
        # 躯干系两脚前后差：右-左，正=右脚在前（feet_indices 左,右 顺序已实测核对）
        foot_pos = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        quat = self.base_quat.unsqueeze(1).repeat(1, 2, 1)
        foot_rel = quat_rotate_inverse(quat.reshape(-1, 4),
                                       foot_pos.reshape(-1, 3)).reshape(self.num_envs, 2, 3)
        return foot_rel[:, 1, 0] - foot_rel[:, 0, 0]

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        # EMA(τ=一个步态周期 0.6s)：交替步态→0，常驻前后脚→定值
        dx = self._feet_dx_torso()
        alpha = np.exp(-self.dt / 0.6)
        self.stagger_ema = alpha * self.stagger_ema + (1 - alpha) * dx
        # episode 诊断累计：支撑膝(带权)、躯干倾角、带符号 dx、EMA
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        w = (contact & (self.leg_phase < self.stance_frac)).float()
        self.episode_knee_sum += (self.dof_pos[:, self.knee_idx] * w).sum(dim=1) * self.dt
        self.episode_knee_w += w.sum(dim=1) * self.dt
        pitch = torch.asin(torch.clamp(self.projected_gravity[:, 0], -1., 1.))
        self.episode_pitch_sum += pitch * self.dt
        self.episode_dx_sum += dx * self.dt
        self.episode_stagger_sum += self.stagger_ema * self.dt

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # 样本时长要在 super 里被清零前先取（用于 pitch/dx 的均值分母）
        dur = torch.clamp(self.episode_height_sample_count[env_ids] * self.dt, min=self.dt)
        super().reset_idx(env_ids)
        self.stagger_ema[env_ids] = 0.
        knee = self.episode_knee_sum[env_ids] / torch.clamp(self.episode_knee_w[env_ids], min=self.dt)
        self.extras["episode"]["knee_stance_mean"] = torch.mean(knee)
        self.extras["episode"]["torso_pitch_deg"] = torch.mean(
            torch.rad2deg(self.episode_pitch_sum[env_ids] / dur))
        self.extras["episode"]["feet_dx_mean"] = torch.mean(self.episode_dx_sum[env_ids] / dur)
        self.extras["episode"]["stagger_ema_mean"] = torch.mean(
            self.episode_stagger_sum[env_ids] / dur)
        self.episode_knee_sum[env_ids] = 0.
        self.episode_knee_w[env_ids] = 0.
        self.episode_pitch_sum[env_ids] = 0.
        self.episode_dx_sum[env_ids] = 0.
        self.episode_stagger_sum[env_ids] = 0.

    # 支撑期膝过弯罚：深蹲是支撑现象；摆动期抬腿弯膝、摆动窗内偶然触地不罚
    def _reward_stance_knee(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        mask = contact & (self.leg_phase < self.stance_frac)
        excess = torch.square(torch.clamp(self.dof_pos[:, self.knee_idx] - self.KNEE_DEADBAND, min=0.0))
        return torch.sum(excess * mask, dim=1)

    def _reward_hip_pos(self):
        # 基类罚 hip_yaw 归零(防内八)顶死了转向原语：目标角改为随 ωz 指令同向劈开(两轴均+z)
        ref = self.HIP_YAW_STEER * self.commands[:, 2].unsqueeze(1)
        return torch.sum(torch.square(self.dof_pos[:, self.hip_yaw_idx] - ref), dim=1)

    def _resample_commands(self, env_ids):
        super()._resample_commands(env_ids)
        # yaw 死区清除：一半重采样压到 |ωz|∈[0.2,0.5]，持续转向经验不被均匀采样稀释
        mask = torch.rand(len(env_ids), device=self.device) < 0.5
        sign = torch.where(torch.rand(len(env_ids), device=self.device) < 0.5, -1.0, 1.0)[mask]
        self.commands[env_ids[mask], 2] = sign * (0.2 + 0.3 * torch.rand(int(mask.sum()), device=self.device))

    # 步态对称罚（阶段二启用，当前 config 无 scale）：EMA 常驻值，有运动指令才生效
    def _reward_gait_symmetry(self):
        motion = torch.norm(self.commands[:, :2], dim=1) > 0.1
        return torch.square(self.stagger_ema) * motion

    # 全局速度税只收腿+腰：8个臂dof的dof_vel是伪值(见_init_buffers)，收了=罚不存在的抖振
    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel[:, self.leg_waist_idx]), dim=1)

    def _arm_swing_cmd(self):
        # 肩摆指令 = ±A·v̂x·cos(2πφ)，与同侧腿反相(φ=0 左腿落地在前→左臂在后)
        vxn = torch.clamp(self.commands[:, 0], -1.0, 1.0).unsqueeze(1)
        cos2 = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        return self.ARM_SWING_AMP * vxn * cos2   # [N,1]，左臂取 +，右臂取 −

    def _reward_arm_swing(self):
        # vx=0 时参考退化为默认角：站立/侧移手自然贴身，无需门控
        sgn = torch.tensor([1.0, -1.0], device=self.device)
        ref = self.arm_pitch_default + self._arm_swing_cmd() * sgn
        return torch.sum(torch.square(self.dof_pos[:, self.arm_pitch_idx] - ref), dim=1)

    def _reward_arm_elbow(self):
        # 肘联动：elbow_ref = 默认 + C·肩摆指令(臂后摆→肘伸直些)
        sgn = torch.tensor([1.0, -1.0], device=self.device)
        ref = self.elbow_default + self.ELBOW_COUPLE * self._arm_swing_cmd() * sgn
        return torch.sum(torch.square(self.dof_pos[:, self.elbow_idx] - ref), dim=1)

    def _reward_arm_spread(self):
        # 侧移臂微张：roll_ref = 默认 ± S·|v̂y|，外侧张开保平衡
        ayn = self.commands[:, 1].abs().clamp(max=1.0).unsqueeze(1)
        sgn = torch.tensor([1.0, -1.0], device=self.device)
        ref = self.arm_roll_default + self.ARM_SPREAD * ayn * sgn
        return torch.sum(torch.square(self.dof_pos[:, self.arm_roll_idx] - ref), dim=1)

    def _reward_shoulder_yaw_pos(self):
        # 肩yaw死区±8°(0.14rad)外罚平方：防小臂外翻，死区内留给平衡微调
        yaw = self.dof_pos[:, self.arm_yaw_idx]
        return torch.sum(torch.square(torch.clamp(yaw.abs() - 0.14, min=0.0)), dim=1)

    def _reward_wrist_pos(self):
        # 腕roll无任务引用，罚漂移防常驻限位
        return torch.sum(torch.square(self.dof_pos[:, self.wrist_idx]), dim=1)

    def _reward_waist_swing(self):
        # 腰反旋(与左臂同相：臂/肩在后=腰正转) + 转弯肩预旋(ωz 同号)
        vxn = torch.clamp(self.commands[:, 0], -1.0, 1.0)
        wzn = torch.clamp(self.commands[:, 2], -1.0, 1.0)
        cos2 = torch.cos(2 * np.pi * self.phase)
        ref = self.WAIST_SWING * vxn * cos2 + self.WAIST_TURN * wzn
        return torch.square(self.dof_pos[:, self.waist_idx] - ref)

    def _reward_orientation(self):
        # 躯干随 vx 前倾：重力参考 x 分量 = sin(lean)。前倾=绕+y 正转(机头向下)
        lean = self.LEAN_COEF * torch.clamp(self.commands[:, 0], -1.0, 1.0)
        ref = torch.stack([torch.sin(lean), torch.zeros_like(lean)], dim=1)
        return torch.sum(torch.square(self.projected_gravity[:, :2] - ref), dim=1)
