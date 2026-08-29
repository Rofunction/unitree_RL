from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO

# 23dof = 12 腿 + 1 腰 + 2×5 臂。URDF 前 12 关节与 12dof 完全同序，
class G1Rough23dofCfg( G1RoughCfg ):

    class init_state( G1RoughCfg.init_state ):
        default_joint_angles = {
           'left_hip_yaw_joint' : 0. ,
           'left_hip_roll_joint' : 0,
           'left_hip_pitch_joint' : -0.1,
           'left_knee_joint' : 0.3,
           'left_ankle_pitch_joint' : -0.2,
           'left_ankle_roll_joint' : 0,
           'right_hip_yaw_joint' : 0.,
           'right_hip_roll_joint' : 0,
           'right_hip_pitch_joint' : -0.1,
           'right_knee_joint' : 0.3,
           'right_ankle_pitch_joint': -0.2,
           'right_ankle_roll_joint' : 0,
           # 臂零位=小臂前平举(elbow 1.57 才是自然下垂)：1.2=微屈肘近似人站立
           'waist_yaw_joint' : 0.,
           'left_shoulder_pitch_joint' : 0.,
           'left_shoulder_roll_joint' : 0.08,
           'left_shoulder_yaw_joint' : 0.,
           'left_elbow_joint' : 1.2,
           'left_wrist_roll_joint' : 0.,
           'right_shoulder_pitch_joint' : 0.,
           'right_shoulder_roll_joint' : -0.08,
           'right_shoulder_yaw_joint' : 0.,
           'right_elbow_joint' : 1.2,
           'right_wrist_roll_joint' : 0.,
        }

    class env(G1RoughCfg.env):
        num_actions = 23
        observe_base_height = True
        # 9 + 3*23 + base-height error + sin/cos phase
        num_observations = 81
        # privileged observation additionally contains base linear velocity
        num_privileged_obs = 84

    class asset(G1RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_23dof.urdf'
        penalize_contacts_on = ["hip", "knee", "shoulder", "elbow"]

    class control(G1RoughCfg.control):
        stiffness = {**G1RoughCfg.control.stiffness,
                     'waist': 100,
                     'shoulder': 40,
                     'shoulder_yaw': 40,
                     'elbow': 40,
                     'wrist': 5,
                     }  # [N*m/rad]
        # kd 按各关节 I_eff 配 ζ（参照 12dof 腿: 髋≈0.11/膝≈0.48/踝≈1.4）且满足 β=kd·dt/I<2
        damping = {**G1RoughCfg.control.damping,
                    'waist': 2,
                    'shoulder': 2,
                    'shoulder_yaw': 1.4,   # I≈4.3e-3, kd>1.63 会自激; 1.4→β=1.62, ζ=1.7
                    'elbow': 2,
                    'wrist': 0.05,   # I≈2e-4, kd≤0.067 才稳; 0.05→β=1.26, ζ=0.8
                    }  # [N*m*s/rad]

    class domain_rand(G1RoughCfg.domain_rand):
        added_mass_range = [-2., 4.]

    class commands(G1RoughCfg.commands):
        class ranges(G1RoughCfg.commands.ranges):
            ang_vel_yaw = [-0.5, 0.5]

    class rewards( G1RoughCfg.rewards ):
        base_height_target = 0.78
        only_positive_rewards = False  # 站立净正前拆 clip 无信号；现靠 alive+罚项削减保证净 ≥ −0.022
        class scales( G1RoughCfg.rewards.scales ):
            alive = 1.0    # 姿态线已关+净+0.07/步稳：从2降1，收入重心从"活着"转"听指挥"(yaw 38000实测-2%被无视)
            tracking_lin_vel = 2.0   # 站 0.005 vs 好步态 0.034/步：走路比站多赚 60%
            tracking_ang_vel = 3.0   # 1.5 下 yaw 仍被弃(原地/边走边转均-2%)：翻倍+alive降1把转向拉出弃学区，学出后回1.5
            dof_vel = -1e-3   # 只收腿+腰(env重写)：臂8dof的dof_vel读冻结伪值41.7，旧"抖振大头"是缓冲区伪影
            dof_acc = -1e-7
            base_height = -30.0     # -10 时 5cm 蹲幅仅罚 0.025(收入 4.3 的 0.6%)；-30 兜底，主约束靠 stance_knee
            stance_knee = -3.0      # 支撑膝>0.55rad 罚平方：基线支撑膝中位 0.79→罚~0.06/腿，站直归零
            gait_symmetry = -20.0   # 阶段二b：-5 在Δx=0.067处仅罚0.022(收入0.4%)形同虚设→27000回吐；-20 逼回≤0.05
            orientation = -4.0      # -1 时 13° 后仰仅罚 0.05；先 -4 看 torso_pitch_deg，再决定是否 -6。gravity_x 负=后仰已实测核对
            feet_swing_height = -10.0
            collision = -0.5
            dof_pos_limits = -2.5
            arm_swing = -10.0  # 摆臂跟踪(见 g1_23dof_env)：vx=0 时参考退化为贴默认角；-2 实测罚金太小策略交罚不摆
            arm_elbow = -5.0   # 肘联动：臂后摆时肘伸直些
            arm_spread = -5.0  # 侧移时臂横向微张保平衡
            waist_swing = -5.0 # 腰反旋 + 转弯肩预旋
            shoulder_yaw_pos = -5.0  # 肩yaw死区±8°外罚平方：防小臂外翻(model_8550 外翻常驻)
            wrist_pos = -1.0   # 腕roll无任务引用，罚漂移防常驻限位
            feet_collision = -0.5
            lin_vel_z = -2.0   # 抑制周期性上下速度，减少一蹲一蹲
            termination = -250.0  # 生效=scale×dt=−5/次摔倒：防"速死止损"套利


class G1Rough23dofCfgPPO( G1RoughCfgPPO ):
    class algorithm(G1RoughCfgPPO.algorithm):
        entropy_coef = 0.002

    class policy(G1RoughCfgPPO.policy):
        actor_hidden_dims = [64]
        critic_hidden_dims = [64]

    class runner(G1RoughCfgPPO.runner):
        max_iterations = 10000
        experiment_name = 'g1_23dof'
