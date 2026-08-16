
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1Robot(LeggedRobot):
    
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
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # sin/cos phase
        
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

        period = 0.65  # 步态周期 0.8→0.65s：提高步频，降低每步横移量，抑制垫步
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase ).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase ).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
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
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        # Penalize feet being too low during swing phase，desired height: 0.10m
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.10) * ~contact
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
        # 只惩罚 hip_yaw，防止两腿交叉；不惩罚 hip_roll（侧向迈步主关节）
        return torch.sum(torch.square(self.dof_pos[:,[0,6]]), dim=1)

    def _reward_feet_distance(self):
        # 只在单脚支撑时罚横向间距<0.15m（摆动脚经过支撑脚），防刮蹭且不干扰正常步态
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        single_support = (contact[:, 0] ^ contact[:, 1]).float()
        foot_pos = (self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)).reshape(self.num_envs * 2, 3)
        quat = self.base_quat.unsqueeze(1).repeat(1, 2, 1).reshape(self.num_envs * 2, 4)
        foot_y = quat_rotate_inverse(quat, foot_pos)[:, 1].reshape(self.num_envs, 2)
        dist = torch.abs(foot_y[:, 0] - foot_y[:, 1])
        return torch.clamp(0.15 - dist, min=0.0) * single_support

    # ======================================================================
    # ② feet_lateral_align —— 横向步态奖励（已启用）
    #    目的：摆动腿朝指令侧(vy)移动，抑制“错腿先抬 / 交叉迈步”。
    #    配套：g1_config.py rewards.scales 里 feet_lateral_align = 0.5
    # ------------------------------------------------------------------
    def _reward_feet_lateral_align(self):
        contact  = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        swing    = ~contact                                  # [N, 2] True=空中
        feet_vy  = self.feet_vel[:, :, 1]                    # [N, 2] 两脚横向速度
        cmd_vy   = self.commands[:, 1].unsqueeze(1)          # [N, 1] 横向指令
        return torch.sum(feet_vy * cmd_vy * swing, dim=1)

    # ======================================================================
    # ③ turn_arc —— 转弯步态奖励（默认关闭）
    #    启用方式：① 取消下面 _reward_turn_arc 的注释；
    #              ② 在 g1_config.py 的 rewards.scales 里取消注释
    #                 turn_arc = 0.2
    #    目的：转弯时外脚多走、内脚当轴（步幅随 yaw 成弧线）。
    # ------------------------------------------------------------------
    # def _reward_turn_arc(self):
    #     feet_vx = self.feet_vel[:, :, 0]                     # [N, 2] 两脚前向速度
    #     foot_y  = self.feet_pos[:, :, 1]                     # [N, 2] 两脚横向位置
    #     yaw     = self.commands[:, 2].unsqueeze(1)           # [N, 1] 偏航指令
    #     return torch.sum(feet_vx * (yaw * foot_y), dim=1)
    