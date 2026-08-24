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
        num_observations = 80
        num_privileged_obs = 83

    class asset(G1RoughCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_23dof.urdf'
        penalize_contacts_on = ["hip", "knee", "shoulder", "elbow"]

    class control(G1RoughCfg.control):
        stiffness = {**G1RoughCfg.control.stiffness,
                     'waist': 100,
                     'shoulder': 40,
                     'elbow': 40,
                     'wrist': 5,
                     }  # [N*m/rad]
        damping = {**G1RoughCfg.control.damping,
                    'waist': 2,
                    'shoulder': 2,
                    'elbow': 2,
                    'wrist': 0.5,
                    }  # [N*m*s/rad]

    class domain_rand(G1RoughCfg.domain_rand):
        added_mass_range = [-2., 4.]

    class commands(G1RoughCfg.commands):
        class ranges(G1RoughCfg.commands.ranges):
            ang_vel_yaw = [-0.5, 0.5]

    class rewards(G1RoughCfg.rewards):
        only_positive_rewards = False  # 站立净正前拆 clip 无信号；现靠 alive+罚项削减保证净 ≥ −0.022
        class scales(G1RoughCfg.rewards.scales):
            alive = 2.0
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.5 
            dof_vel = -2e-4
            dof_acc = -1e-7
            base_height = -15.0
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
            lin_vel_z = -1.0   # 12dof 为 -2.0；放软让 CoM 每周期两次自然起伏(人类 2-3cm)
            termination = -250.0


class G1Rough23dofCfgPPO( G1RoughCfgPPO ):
    class algorithm(G1RoughCfgPPO.algorithm):
        entropy_coef = 0.002

    class policy(G1RoughCfgPPO.policy):
        actor_hidden_dims = [64]
        critic_hidden_dims = [64]

    class runner(G1RoughCfgPPO.runner):
        max_iterations = 10000
        experiment_name = 'g1_23dof'
