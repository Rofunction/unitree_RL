
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch

class G1Robot(LeggedRobot):
    stance_frac = 0.55  # 支撑相位窗终点(=摆动起点)。双支撑占比 = 2*stance_frac-1 → 0.55=10%周期

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        height_idx = 9 + 3 * self.num_actions
        phase_idx = height_idx
        if getattr(self.cfg.env, 'observe_base_height', False):
            noise_vec[height_idx] = noise_scales.height_measurements * noise_level # ±0.1 obs 单位 ≈ ±2cm，对齐真机腿运动学估计误差
            phase_idx += 1
        noise_vec[phase_idx:phase_idx + 2] = 0. # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        # feet_state: [num_envs, feet_num, 13]
        # 13: pos(3), quat(4), vel(3), ang_vel(3)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.60
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        height_obs = []
        if getattr(self.cfg.env, 'observe_base_height', False):
            # Match the reward's physical target and keep the input numerically scaled.
            height_obs.append(
                (self.root_states[:, 2] - self.cfg.rewards.base_height_target).unsqueeze(1)
                * self.obs_scales.height_measurements
            )
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    *height_obs,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    *height_obs,
                                    sin_phase,
                                    cos_phase
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        # Reward for correct contact timing
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < self.stance_frac
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    # 摆动脚高 v5：目标=0.13·sin(πs)，s=摆动窗内进度(stance_frac→1.0)。前段逼快抬、后段压落，
    # 消除恒定目标导致的顶点悬停(实测平台0.12s)。0.13=支撑高0.035+离地间隙0.095
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        s = torch.clamp((self.leg_phase - self.stance_frac) / (1.0 - self.stance_frac), min=0.0, max=1.0)
        z_ref = 0.13 * torch.sin(np.pi * s)
        pos_error = torch.square(self.feet_pos[:, :, 2] - z_ref) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1,2))
    
    def _reward_hip_pos(self):
        # 只罚 hip_yaw([2,8]，URDF 顺序 pitch,roll,yaw)：防内八；[0,6] 是 pitch，罚了会压制抬腿。
        return torch.sum(torch.square(self.dof_pos[:,[2,8]]), dim=1)

    # ② 摆动腿朝指令侧(vy)移动，抑制错腿先抬/交叉迈步。
    def _reward_feet_lateral_align(self):
        contact  = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        swing    = ~contact                                  # [N, 2] True=空中
        feet_vy  = self.feet_vel[:, :, 1]                    # [N, 2] 两脚横向速度
        cmd_vy   = self.commands[:, 1].unsqueeze(1)          # [N, 1] 横向指令
        return torch.sum(feet_vy * cmd_vy * swing, dim=1)

    # ④ 摆动窗口内脚上出现接触力 = 刮蹭另一条腿/没抬起（脚-脚接触原本零成本）。
    def _reward_feet_collision(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        swing_phase = self.leg_phase >= self.stance_frac          # 相位规定的摆动窗口
        return torch.sum(contact.float() * swing_phase, dim=1)

    # ⑤ 摆动期两脚水平 2D 间距，<0.12 平方尖峰（物理碰撞临界≈0.07）；
    #    含 x 允许剪刀式错开绕行，门控摆动脚。阈值几何依据见 config 注释。
    def _reward_feet_clearance(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        swing = ~contact                               # [N, 2] 摆动脚在场才算
        foot_pos = (self.feet_pos - self.root_states[:, 0:3].unsqueeze(1))
        quat = self.base_quat.unsqueeze(1).repeat(1, 2, 1)
        foot_rel = quat_rotate_inverse(quat.reshape(-1, 4),
                                       foot_pos.reshape(-1, 3)).reshape(self.num_envs, 2, 3)
        dist = torch.norm(foot_rel[:, 0, :2] - foot_rel[:, 1, :2], dim=-1)  # 水平面距离
        too_close = torch.square(torch.clamp(0.12 - dist, min=0.0) / 0.12)
        return torch.sum(too_close.unsqueeze(1) * swing, dim=1)

    # ③ turn_arc（默认关闭）：转弯时外脚多走、内脚当轴。启用：取消下述注释 + config 里 turn_arc = 0.2
    # def _reward_turn_arc(self):
    #     feet_vx = self.feet_vel[:, :, 0]                     # [N, 2] 两脚前向速度
    #     foot_y  = self.feet_pos[:, :, 1]                     # [N, 2] 两脚横向位置
    #     yaw     = self.commands[:, 2].unsqueeze(1)           # [N, 1] 偏航指令
    #     return torch.sum(feet_vx * (yaw * foot_y), dim=1)
