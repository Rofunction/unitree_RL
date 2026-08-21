
from legged_gym.envs.g1.g1_env import G1Robot

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

    def _reward_alive(self):
        # 底薪只发给"用脚活着"：跪/坐姿脚不承重(实测触地仅18%) → alive≈0，堵死跪着领薪
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        return torch.any(contact, dim=1).float()

    def _init_buffers(self):
        super()._init_buffers()
        idx = {n: i for i, n in enumerate(self.dof_names)}
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
