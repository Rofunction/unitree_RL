from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1RoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.8] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
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
           'torso_joint' : 0.
        }
    
    class env(LeggedRobotCfg.env):
        # --- Observation / action dimensions ---
        # G1 overrides compute_observations() in g1_env.py, so the obs layout here
        # differs from the base LeggedRobot. obs_buf size = 9 + 3*N + 2, N = num_actions.
        #   [0:3]          base_ang_vel * scale
        #   [3:6]          projected_gravity
        #   [6:9]          commands[:3] (vx, vy, yaw_rate) * scale
        #   [9 : 9+N]      (dof_pos - default_dof_pos) * scale
        #   [9+N : 9+2N]   dof_vel * scale
        #   [9+2N : 9+3N]  actions (previous step)
        #   [9+3N : 9+3N+2] sin(phase), cos(phase)   # periodic gait prior (see _post_physics_step_callback)
        # base_lin_vel is NOT in obs_buf (hard to measure on hardware) -> only the critic
        # sees it via privileged_obs. With N=12: 9 + 36 + 2 = 47.
        num_observations = 47
        # privileged obs = num_observations + base_lin_vel(3); sim-to-real: the actor/
        # policy never depends on linear velocity, only the value network does.
        num_privileged_obs = 50
        # num_actions = dimension of the policy action vector = number of actuated DOFs
        # (revolute/prismatic joints in the URDF). For g1_12dof this is the 12 leg joints
        # (6 per leg: hip pitch/roll/yaw, knee, ankle pitch/roll); arms/torso are fixed.
        # To switch to 23-dof: set num_actions=23 -> num_observations=80, num_privileged_obs=83,
        # and update the URDF, default_joint_angles, and PD gains accordingly.
        num_actions = 12


    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        # push robots in sim
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.5
      

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
          # PD Drive parameters:
        stiffness = {'hip_yaw': 100,
                     'hip_roll': 100,
                     'hip_pitch': 100,
                     'knee': 150,
                     'ankle': 40,
                     }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                     'hip_roll': 2,
                     'hip_pitch': 2,
                     'knee': 4,
                     'ankle': 2,
                     }  # [N*m/rad]  # [N*m*s/rad]
        # action scale: q_target = action_scale * action + default_joint_angle,
        # then PD torque tau = Kp*(q_target - q) + Kd*(0 - qd)  (qd_target defaults to 0)
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_12dof.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        # terminate_after_contacts_on 指定接触后终止仿真的部位（根据指定的部位名称的接触力来判断是否终止仿真）
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        
        class scales( LeggedRobotCfg.rewards.scales ):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.0
            base_height = -10.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            feet_air_time = 0.0
            collision = 0.0
            action_rate = -0.01
            dof_pos_limits = -5.0
            alive = 0.15
            hip_pos = -1.0
            contact_no_vel = -0.2
            feet_swing_height = -20.0
            contact = 0.18

class G1RoughCfgPPO( LeggedRobotCfgPPO ):
    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        # activation functiona for active non-linearities in the network, avoiding the output layer is always linear
        # for learning complex control policies, 'elu' usually works best
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
        
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        # note: The larger the value, the more random the strategy, 
        # which is beneficial for exploration; the smaller the value, 
        # the more deterministic the strategy. Balances exploration and exploitation.
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 10000
        run_name = ''
        experiment_name = 'g1'

  
