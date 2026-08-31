
from legged_gym.envs.g1.g1_env import G1Robot
from isaacgym.torch_utils import quat_rotate_inverse

import numpy as np
import torch

class G1Robot23dof(G1Robot):
    stance_frac = 0.60     # 双支撑 20% 周期（人类水平；12dof 是 0.55）
    ARM_SWING_AMP = 0.45   # |vx|=1 时肩摆幅[rad]≈26°(人类快走上臂 ±25°)
    ELBOW_COUPLE = 0.4     # 肘联动比：肘摆幅/肩摆幅(人类 30-50%)
    ARM_SPREAD = 0.06      # 侧移臂横张幅[rad]
    WAIST_SWING = 0.10     # 腰反旋幅[rad]≈6°(人类肩-髋反扭 ~10°)
    WAIST_TURN = 0.20      # 转弯肩预旋[rad]（ωz=1 时）
    LEAN_MIN = 0.05        # 躯干前倾窗[2.9°,6.9°](人体稳态): 全速域统一, 倒走不给后仰参考
    LEAN_MAX = 0.12
    KNEE_DEADBAND = 0.50   # 支撑膝罚起征点[rad]：36000后中位稳0.58两轮不过线，起征点下移逼站直(z也受益)
    HIP_YAW_STEER = 0.35   # ωz=1 时双脚同向劈尖参考[rad]：转向需要髋yaw，实测策略仅用2.3°
    KNEE_SWING_AMP = 0.55  # 摆动膝参考幅: 默认0.3+0.55·sin(πs), 峰0.88rad≈50°(人快走~60°)

    def _reward_alive(self):
        # 底薪只发给"用脚活着"：跪/坐姿脚不承重(实测触地仅18%) → alive≈0，堵死跪着领薪
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        return torch.any(contact, dim=1).float()

    def _init_buffers(self):
        super()._init_buffers()
        idx = {n: i for i, n in enumerate(self.dof_names)}
        self.knee_idx = [idx['left_knee_joint'], idx['right_knee_joint']]
        self.hip_yaw_idx = [idx['left_hip_yaw_joint'], idx['right_hip_yaw_joint']]
        self.hip_roll_idx = [idx['left_hip_roll_joint'], idx['right_hip_roll_joint']]
        # 步态姿态诊断状态：EMA(dx) 与 episode 累计量（reset_idx 上报清零）
        self.stagger_ema = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 站桩相位冻结缓存(见 _post_physics_step_callback)
        self.phase_hold = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 直行 yaw 偏置 EMA(τ=1s>步态周期0.6s): 滤掉步态振荡剩恒偏
        self.yaw_ema = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_knee_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_knee_w = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_pitch_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_dx_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_stagger_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.episode_leg_asym_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 左右腿对称：腿对索引(每腿序=pitch,roll,yaw,knee,ankle_pitch,ankle_roll)与 EMA 姿态
        leg_jn = ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']
        self.leg_l_idx = [idx['left_' + j + '_joint'] for j in leg_jn]
        self.leg_r_idx = [idx['right_' + j + '_joint'] for j in leg_jn]
        self.leg_ids = self.leg_l_idx + self.leg_r_idx
        self.pose_ema = self.default_dof_pos.repeat(self.num_envs, 1)
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
        # 站桩相位冻结: 基类按墙钟推进相位, 零指令时 sin/cos 相位输入仍 1.67Hz 旋转,
        # 泄漏进动作=自持晃动。站定后走完当前周期(半空的脚自然落地)停在 0(两腿同入支撑窗)
        moving = (torch.norm(self.commands[:, :2], dim=1) >= 0.2) | \
                 (self.commands[:, 2].abs() >= 0.15)
        nxt = self.phase_hold + self.dt / 0.6
        self.phase_hold = torch.where(
            moving, nxt % 1.0,
            torch.where(self.phase_hold > 0,
                        torch.where(nxt >= 1.0, torch.zeros_like(nxt), nxt),
                        self.phase_hold))
        self.phase = self.phase_hold
        self.phase_left = self.phase
        self.phase_right = (self.phase + 0.5) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1),
                                    self.phase_right.unsqueeze(1)], dim=-1)
        # EMA(τ=一个步态周期 0.6s)：交替步态→0，常驻前后脚→定值
        dx = self._feet_dx_torso()
        alpha = np.exp(-self.dt / 0.6)
        beta = np.exp(-self.dt / 1.0)
        self.yaw_ema = beta * self.yaw_ema + (1 - beta) * self.base_ang_vel[:, 2]
        self.stagger_ema = alpha * self.stagger_ema + (1 - alpha) * dx
        # EMA 姿态(同 τ)：滤掉交替摆动剩常驻差；hip_yaw 归 hip_pos(含转向参考)管，不入
        self.pose_ema = alpha * self.pose_ema + (1 - alpha) * self.dof_pos
        L, R = self.pose_ema[:, self.leg_l_idx], self.pose_ema[:, self.leg_r_idx]
        self.leg_asym = (torch.square(L[:, [0, 3, 4]] - R[:, [0, 3, 4]]).sum(dim=1)
                         + torch.square(L[:, [1, 5]] + R[:, [1, 5]]).sum(dim=1))
        # episode 诊断累计：支撑膝(带权)、躯干倾角、带符号 dx、EMA
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        w = (contact & (self.leg_phase < self.stance_frac)).float()
        self.episode_knee_sum += (self.dof_pos[:, self.knee_idx] * w).sum(dim=1) * self.dt
        self.episode_knee_w += w.sum(dim=1) * self.dt
        pitch = torch.asin(torch.clamp(self.projected_gravity[:, 0], -1., 1.))
        self.episode_pitch_sum += pitch * self.dt
        self.episode_dx_sum += dx * self.dt
        self.episode_stagger_sum += self.stagger_ema * self.dt
        self.episode_leg_asym_sum += self.leg_asym * self.dt

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # 样本时长要在 super 里被清零前先取（用于 pitch/dx 的均值分母）
        dur = torch.clamp(self.episode_height_sample_count[env_ids] * self.dt, min=self.dt)
        super().reset_idx(env_ids)
        self.stagger_ema[env_ids] = 0.
        self.phase_hold[env_ids] = 0.
        self.yaw_ema[env_ids] = 0.
        self.pose_ema[env_ids] = self.default_dof_pos
        knee = self.episode_knee_sum[env_ids] / torch.clamp(self.episode_knee_w[env_ids], min=self.dt)
        self.extras["episode"]["knee_stance_mean"] = torch.mean(knee)
        self.extras["episode"]["torso_pitch_deg"] = torch.mean(
            torch.rad2deg(self.episode_pitch_sum[env_ids] / dur))
        self.extras["episode"]["feet_dx_mean"] = torch.mean(self.episode_dx_sum[env_ids] / dur)
        self.extras["episode"]["stagger_ema_mean"] = torch.mean(
            self.episode_stagger_sum[env_ids] / dur)
        self.extras["episode"]["leg_asym_mean"] = torch.mean(
            self.episode_leg_asym_sum[env_ids] / dur)
        self.episode_knee_sum[env_ids] = 0.
        self.episode_knee_w[env_ids] = 0.
        self.episode_pitch_sum[env_ids] = 0.
        self.episode_dx_sum[env_ids] = 0.
        self.episode_stagger_sum[env_ids] = 0.
        self.episode_leg_asym_sum[env_ids] = 0.

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
        # (纯转样本钉 xy=0 曾试 25% 占比: 转没学会(±0.35→~0)还把 xy=0 区域改写成"要动",
        #  站桩塌成 -3.5rad/s 搅拌, 已撤回; 纯转要么另设计要么不做)
        mask = torch.rand(len(env_ids), device=self.device) < 0.5
        sign = torch.where(torch.rand(len(env_ids), device=self.device) < 0.5, -1.0, 1.0)[mask]
        self.commands[env_ids[mask], 2] = sign * (0.2 + 0.3 * torch.rand(int(mask.sum()), device=self.device))

    # 步态对称罚（阶段二启用，当前 config 无 scale）：EMA 常驻值，有运动指令才生效
    def _reward_gait_symmetry(self):
        motion = torch.norm(self.commands[:, :2], dim=1) > 0.1
        return torch.square(self.stagger_ema) * motion

    # 左右镜像罚(EMA 均值)：pitch/knee/ankle_pitch 同号差 + roll/ankle_roll 反号和
    def _reward_leg_symmetry(self):
        return self.leg_asym

    # 站桩罚：‖cmd_xy‖<0.2 时腿偏离默认角与腿速收税，权重随 |ωz| 高斯衰减——
    # 硬豁免(≥0.2 全免)曾把 ωz=0 拖进站/转断层抖成搅拌；全税又把转向压死(ωz=0.35 不转)
    def _reward_stand_still(self):
        xy = (torch.norm(self.commands[:, :2], dim=1) < 0.2).float()
        w = xy * torch.exp(-torch.square(self.commands[:, 2] / 0.15))
        dev = torch.square(self.dof_pos[:, self.leg_ids]
                           - self.default_dof_pos[:, self.leg_ids]).sum(dim=1)
        vel = torch.square(self.dof_vel[:, self.leg_ids]).sum(dim=1)
        return w * (dev + 0.1 * vel)

    # 摆动膝下限(单侧)：ref = 默认 + AMP·sin(πs)，直腿钟摆摆动在旧 reward 下免费
    # (feet_swing_height 只管脚高)。sin 两端归零不扰触地；站桩不泵膝(运动门控)
    def _reward_swing_knee(self):
        motion = (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        swing = (self.leg_phase >= self.stance_frac).float()
        s = torch.clamp((self.leg_phase - self.stance_frac) / (1.0 - self.stance_frac), 0.0, 1.0)
        ref = self.default_dof_pos[:, self.knee_idx] + self.KNEE_SWING_AMP * torch.sin(np.pi * s)
        err = torch.square(torch.clamp(ref - self.dof_pos[:, self.knee_idx], min=0.0))
        return torch.sum(err * swing, dim=1) * motion

    # 双髋外张税：两腿一起外摆时镜像差≈0(leg_symmetry 失明)，hip_roll 此前全库无罚
    def _reward_hip_roll_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_roll_idx]), dim=1)

    # 直行偏置罚：步态固有 yaw 振荡(±0.1-0.3, 周期0.6s)合法，罚慢均值(τ=1s EMA)——
    # 实测恒偏 0.9-1.6°/s→13.9cm/m；tracking 二次型近零区太平(err0.03 仅损~0.01/s)压不动
    def _reward_straight_yaw(self):
        moving = (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()
        straight = (self.commands[:, 2].abs() < 0.05).float()
        return moving * straight * self.yaw_ema.abs()

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
        # 躯干前倾窗[LEAN_MIN,LEAN_MAX]外罚平方(参考不随 vx 变: 原参考 vx=-1→后仰5°是倒走翻车主因)
        lean = torch.asin(torch.clamp(self.projected_gravity[:, 0], -1., 1.))
        err = torch.clamp(self.LEAN_MIN - lean, min=0.) + torch.clamp(lean - self.LEAN_MAX, min=0.)
        return torch.square(err) + torch.square(self.projected_gravity[:, 1])
