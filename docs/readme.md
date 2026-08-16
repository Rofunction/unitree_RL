# RL强化学习的基础
## IsaacGym部分
![alt text](picture/image-2.png)
### 1 create_sim
create_sim主要涉及的是legged_robot.py和它继承的baskTask.py中的init函数。它主要做了一下操作，如下图所示。如果需要改地形等，可以在g1_config.py中修改对应继承的变量的方式来修改。

![alt text](picture/image.png)![alt text](picture/image-1.png)

### 2 get/set State
![alt text](picture/image-4.png)
对用户来说主要关注的是Pytorch Tensor层级的get/set State。
#### 2.1 get State
get State函数开始在Legged_robot.py中的init_buffers初始化，更新函数为step(self, action)函数
![alt text](picture/image-5.png)
![alt text](picture/image-7.png)
#### 2.2 set State
主要在reset_idx和step函数
![alt text](picture/image-8.png)
#### 2.3 set Command
采用PD控制：
$\tau = K_p*(q_d-q)+K_d(\dot{q}_d-\dot{q})$
由于poliy得到的$q_d$,因此默认设置$\dot{q}_d=0$。参考_compute_torques函数
![alt text](picture/image-9.png)

## RSL_RL
rsl_rl是RL算法的核心实现部分，比如PPO等算法均是在这里实现的。
![alt text](picture/image-3.png)
![alt text](picture/image-25.png)
![alt text](picture/image-6.png)
![alt text](picture/image-26.png)
![alt text](picture/image-29.png)
![alt text](picture/image-28.png)
![alt text](picture/image-27.png)
iteration的作用就是收集训练数据 && 更新神经网络
### RNN和MLP网络
RNN和MLP的区别在于：RNN会利用历史+当前信息推理和预测未来输出结果，而MLP只会利用当前信息推理未来输出结果。RNN的优势在机器人训练的优势：尽早发现异常ation，能及时纠正；MLP无法利用历史信息纠偏。
![alt text](picture/image-10.png)
![alt text](picture/image-11.png)
![alt text](picture/image-12.png)
![alt text](picture/image-14.png)
![alt text](picture/image-13.png)
在g1_config.py里面定义了由RNN输出结果转化为action|values的MLP隐藏层数，以及由MLP输出的维度转化为1维的方法：
数据流如下：（以 critic 为例，配置 critic_hidden_dims=32、rnn_hidden_size=64）：

```mermaid
flowchart TD
    A["原始观测<br/>num_critic_obs"] --> B["Memory (LSTM)<br/>input_size = 原始观测维<br/>hidden_size = rnn_hidden_size = 64"]
    B --> C["循环输出 64维<br/>= rnn_hidden_size<br/>RNN 隐藏态 / 记忆容量"]
    C --> D1
    subgraph MLP["Critic MLP — 输入维=64, 隐藏层=critic_hidden_dims=[32]"]
      D1["Linear(64→32)"] --> D2["ELU"] --> D3["Linear(32→1)"]
    end
    D3 --> E["价值 V(s)"]

    classDef recurrent fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#0d47a1;
    classDef mlp fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#bf360c;
    class B,C recurrent;
    class D1,D2,D3 mlp;
```

![alt text](picture/image-15.png)
### 1 train
train部分主要负责的就是RL算法训练的调用。主要的步骤就是以下三个函数：
![alt text](picture/train.png)
### 2 play
play.py是用来播放训练好的policy的代码，核心的部分也是以下3部分：
![alt text](picture/play.png)
### 3 task_registry
![alt text](picture/task_registry.png)
task_registry它相当于打开rsl_rl的钥匙，建立起env中定义的机器人环境与rsl_rl的桥梁。我们在输入命令时使用task=g1起作用的原因是因为在env/__init__.py函数中预先定义好相应的cfg。因此当需要训练一个新的机器人型号时，必要步骤：
1、在envs中定义好cfg配置文件；
2、在__init__.py中初始化相应的task_registry代码。
![alt text](picture/task_registry_1.png)
![alt text](picture/image-22.png)
在创建新的机器人训练模型时，在robot_config.py中的rewards中增加了LeggedRobotCfg.reward中没有的参数时，需要在对应的robot_env.cfg中增加对应的reward计算函数
![alt text](picture/config.png)
![alt text](picture/env.png)
### 4 Observation
![alt text](picture/image-23.png)
![alt text](picture/image-24.png)

### 5 LeggedRobotCfgPPO

















## Reward Functions（奖励函数详解）

强化学习中，机器人每一步 `step` 结束后会计算一个__总奖励__ `r`，它由一组 `_reward_xxx` 函数加权求和得到：

$$r = \sum_i \text{scale}_i \cdot \text{reward}_i$$

- 每个 `_reward_xxx` 函数对应配置里 `cfg.rewards.scales.xxx` 的一个权重项 `scale_xxx`。
- 函数定义在 [legged_robot.py](../legged_gym/envs/base/legged_robot.py)（base 类，所有机器人通用），G1/H1/H1_2 又在各自的 `xxx_env.py` 中__新增__了若干项。
- __权重为正__表示"鼓励"（奖励项），__权重为负__表示"惩罚"（约束项）；__权重为 0__ 则该项不生效（相当于关闭）。
- 每个函数返回 shape 为 `[num_envs]` 的 tensor，框架会自动乘以 `scale` 并累加。

下面按功能分组说明。公式记号：$q$ 关节位置、$\dot q$ 关节速度、$\tau$ 关节力矩、$v$ 基座线速度、$\omega$ 基座角速度、`commands` 指令（`[vx, vy, wz]`）。

---

### 1. 任务目标奖励（正向引导，鼓励完成任务）

| 函数 | 公式 | 含义 |
|---|---|---|
| `_reward_tracking_lin_vel` | $\exp\!\left(-\frac{\lVert \text{cmd}_{xy} - v_{xy}\rVert^2}{\sigma}\right)$ | __跟踪线速度指令__（xy）。实际速度越接近指令速度奖励越大，最大为 1。是行走的核心目标项。 |
| `_reward_tracking_ang_vel` | $\exp\!\left(-\frac{(\text{cmd}_\omega - \omega_z)^2}{\sigma}\right)$ | __跟踪偏航角速度指令__（绕 z 轴转向）。让机器人按指令转向。 |
| `_reward_feet_air_time` | $\sum_{\text{first contact}} (t_{\text{air}} - 0.5)\cdot\mathbf{1}\!\left[\lVert \text{cmd}_{xy}\rVert > 0.1\right]$ | __鼓励迈步__。只在脚__首次着地__那一刻结算空中时间，超过 0.5s 才给正奖励；指令为 0 时不奖励（避免原地空踏）。促使机器人迈出稳定步伐。 |
| `_reward_contact` *(G1/H1/H1_2)* | $\sum_i \neg(\text{contact}_i \oplus \text{stance}_i)$ | __步态相位一致性__。`leg_phase < 0.55` 为支撑相（应着地）。当"处于支撑相"与"实际接触地面"两者一致时给 +1，不一致（该着地没着地 / 该抬脚没抬脚）则不给。强制双脚按预设相位交替落脚。 |
| `_reward_feet_swing_height` *(G1/H1/H1_2)* | $\sum_i (z_{\text{foot}_i} - 0.08)^2 \cdot \neg\text{contact}_i$ | __摆动相抬脚高度__。脚__离地（摆动相）__时惩罚其高度偏离目标 0.08m，促使抬脚到合适高度、避免拖地。（权重为负，故实际是惩罚） |
| `_reward_alive` *(G1/H1/H1_2)* | 常数 `1.0` | __存活奖励__。只要没摔倒每步都给，鼓励保持站立。 |
| `_reward_stand_still` | $\sum_i \lvert q_i - q_i^{\text{default}}\rvert \cdot \mathbf{1}\!\left[\lVert \text{cmd}_{xy}\rVert < 0.1\right]$ | __零指令下保持静止__。当没有移动指令时，惩罚关节偏离默认位姿，让机器人原地站好不乱动。 |

### 2. 姿态与稳定性惩罚（让机器人走得稳、不晃）

| 函数 | 公式 | 含义 |
|---|---|---|
| `_reward_lin_vel_z` | $v_z^2$ | __惩罚竖直方向线速度__。行走应水平，避免上下颠簸（弹跳/跌落感）。 |
| `_reward_ang_vel_xy` | $\omega_x^2 + \omega_y^2$ | __惩罚横滚/俯仰角速度__。避免机器人在 x/y 平面翻滚或前后栽倒。 |
| `_reward_orientation` | $\lVert g_{xy}\rVert^2$（`projected_gravity`） | __惩罚躯干倾斜__。投影重力在 xy 分量越小说明躯干越水平（直立）。防止左右晃倒或前倾后仰。 |
| `_reward_base_height` | $(z_{\text{base}} - h_{\text{target}})^2$ | __惩罚基座高度偏离目标__。`base_height_target` 按机器人设定（G1≈0.78m、H1≈1.05m、H1_2≈1.0m），让机器人保持目标身高、不蹲不跳。 |

### 3. 运动平滑性惩罚（让动作连续、不抖动）

| 函数 | 公式 | 含义 |
|---|---|---|
| `_reward_action_rate` | $\sum_i (a_i - a_i^{\text{last}})^2$ | __惩罚策略输出（动作）变化过快__。抑制相邻步动作跳变，得到平滑控制信号。 |
| `_reward_dof_acc` | $\sum_i \left(\frac{\dot q_i^{\text{last}} - \dot q_i}{\Delta t}\right)^2$ | __惩罚关节加速度__。关节速度的差分近似加速度，抑制剧烈加减速，保护电机、减少抖动。 |
| `_reward_dof_vel` | $\sum_i \dot q_i^2$ | __惩罚关节速度__。鼓励低能耗的缓和运动。 |

### 4. 能耗与硬件安全惩罚（保护电机、防止超限）

| 函数 | 公式 | 含义 |
|---|---|---|
| `_reward_torques` | $\sum_i \tau_i^2$ | __惩罚力矩（能耗）__。力矩平方正比于电机功率，鼓励省力行走。 |
| `_reward_torque_limits` | $\sum_i \max(0,\, \lvert\tau_i\rvert - \tau_i^{\text{limit}} \cdot c_{\text{soft}})$ | __惩罚接近力矩上限__。超过软限制（`soft_torque_limit`）才罚，防止电机过载。 |
| `_reward_dof_pos_limits` | $\sum_i \big[\max(0, q_i^{\text{lo}}-q_i) + \max(0, q_i-q_i^{\text{hi}})\big]$ | __惩罚关节接近行程极限__。超出 `[dof_pos_limits]` 范围才罚，防止关节撞限位损坏。 |
| `_reward_dof_vel_limits` | $\sum_i \mathrm{clip}(\lvert\dot q_i\rvert - \dot q_i^{\text{limit}} \cdot c_{\text{soft}},\, 0,\, 1)$ | __惩罚接近速度上限__。超出软限制（`soft_dof_vel_limit`）才罚，并截断到 1 rad/s 避免罚值爆炸。 |
| `_reward_feet_contact_forces` | $\sum_i \max(0,\, \|F_i\| - F_{\max})$ | __惩罚过大的足部接触力__。超过 `max_contact_force`(默认 100N) 才罚，落脚轻柔、避免砸地。 |

### 5. 碰撞与异常惩罚（避免危险/非法状态）

| 函数 | 公式 | 含义 |
|---|---|---|
| `_reward_collision` | $\sum_{\text{penalised}} \mathbf{1}[\|F\| > 0.1]$ | __惩罚指定部位碰撞__。统计"惩罚接触体"列表（如膝盖、躯干）是否发生接触，鼓励除了脚以外不碰到地面/自身。 |
| `_reward_termination` | `reset_buf & ¬time_out_buf` | __非超时终止惩罚__。机器人因摔倒等异常被 reset（而非正常计时结束）时给负奖励，明确 discourage 危险行为。 |
| `_reward_stumble` | $\mathbf{1}\!\left[\exists\, i:\ \lVert F_{i,xy}\rVert > 5\,\lvert F_{i,z}\rvert\right]$ | __惩罚绊倒__。当脚受到的水平方向冲击远大于竖直方向时判定为撞到竖直障碍（绊脚），予以惩罚。 |
| `_reward_contact_no_vel` *(G1/H1/H1_2)* | $\sum_i \|v_{\text{foot}_i}\|^2 \cdot \text{contact}_i$ | __惩罚着地打滑__。脚__处于接触（支撑相）时__其速度应近似为 0，若有速度说明在滑动/拖拽，予以惩罚。 |
| `_reward_hip_pos` *(G1/H1/H1_2)* | $q_1^2 + q_2^2 + q_7^2 + q_8^2$ | __惩罚髋关节偏移__。索引 `[1,2,7,8]` 为双腿髋部的 roll/pitch，鼓励髋部接近 0 位姿，防止双腿劈叉/内八、保持自然站姿。 |

---
![alt text](picture/image-17.png)
![alt text](picture/image-16.png)
![alt text](picture/image-18.png)
![alt text](picture/image-19.png)
![alt text](picture/image-20.png)
![alt text](picture/image-21.png)
### 6. 相关配置参数

这些参数在 `cfg.rewards` 中定义，控制上面函数的行为：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `tracking_sigma` | 0.25 | 跟踪奖励 `exp(-error²/σ)` 中的 σ。σ 越小对误差越敏感、奖励越尖锐。 |
| `base_height_target` | 1.0（base） | 目标身高，各机器人覆盖（G1=0.78, H1=1.05, H1_2=1.0）。 |
| `soft_dof_vel_limit` | 1.0 | 关节速度软限制系数（<1 时提前开始惩罚）。 |
| `soft_torque_limit` | 1.0 | 力矩软限制系数。 |
| `soft_dof_pos_limit` | 0.9 *(G1/H1/H1_2)* | 关节位置软限制系数，在到达硬限位前 90% 处开始罚。 |
| `max_contact_force` | 100. | 足部接触力阈值(N)，超过则惩罚。 |

### 7. 各机器人实际启用的奖励权重（以 G1 为例）

G1 的 `scales`（见 [g1_config.py](../legged_gym/envs/g1/g1_config.py)），数值为 0 表示该项关闭：

| 项 | 权重 | 类别 |
|---|---|---|
| tracking_lin_vel | __1.0__ | 任务目标 |
| tracking_ang_vel | __0.5__ | 任务目标 |
| lin_vel_z | −2.0 | 姿态稳定 |
| ang_vel_xy | −0.05 | 姿态稳定 |
| orientation | −1.0 | 姿态稳定 |
| base_height | −10.0 | 姿态稳定 |
| dof_acc | −2.5e-7 | 运动平滑 |
| dof_vel | −1e-3 | 运动平滑 |
| action_rate | −0.01 | 运动平滑 |
| feet_swing_height | −20.0 | 任务目标 |
| contact | 0.18 | 任务目标 |
| alive | 0.15 | 任务目标 |
| contact_no_vel | −0.2 | 碰撞异常 |
| dof_pos_limits | −5.0 | 硬件安全 |
| hip_pos | −1.0 | 碰撞异常 |
| collision | 0.0（关闭） | — |

> __提示__：H1、H1_2 的权重与 G1 基本一致，主要差异是 H1 启用了 `collision = −1.0`、并把 `torques` 设为 0；H1_2 与 G1 权重几乎完全相同。新增奖励函数（`contact`、`feet_swing_height`、`alive`、`contact_no_vel`、`hip_pos`）必须在对应机器人 env 中实现，详见 [g1_env.py](../legged_gym/envs/g1/g1_env.py)。

---

出现训练因为5070ti显卡报错问题，请重新编译torch-2.3.0a0+git63d5e92-cp38-cp38-linux_x86_64.whl即可。
### 后续重装isaacgym，避免torch被冲的方式
pip install -e . --no-deps --no-build-isolation
