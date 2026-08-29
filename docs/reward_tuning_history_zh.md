# g1_23dof 奖励参数调参史（2026-08-21 ~ 08-29）

来源：`git log` 对 `legged_gym/envs/g1/g1_23dof_config.py` 的 8 个提交。
**可恢复的最早状态 = d2c41f9（08-21 首次提交）**；更早的调整只存活在注释里（见下表"提交前已发生"列）。

## 1. 演变总表（初版 → 当前）

| 奖励项 | 08-21 初版 | 当前 (08-29) | 轨迹 |
|---|---|---|---|
| alive | 2.0（更早 7.0，提交前已降） | **1.0** | 08-27 2.0→1.0（收入重心从"活着"转"听指挥"） |
| tracking_lin_vel | 2.0 | 2.0 | 未变 |
| tracking_ang_vel | 继承 0.5 | **3.0** | 08-22 显式 1.5；08-27 1.5→3.0（yaw 仍被弃，翻倍拉出弃学区） |
| dof_vel | −2e-4（更早 −1e-3，提交前已放软） | **−1e-3** | 08-25 加 `dof_vel_arms −8e-4`（臂合计≈旧税 1e-3）；08-27 删该项回 −1e-3，但 env 重写为**只收腿+腰** |
| dof_acc | −1e-7 | −1e-7 | 未变 |
| base_height | **−5.0**（比基类 −10 软，留 CoM 起伏） | **−30.0** | 08-24 −5→−10；08-25 −10→−30（5cm 蹲幅仅罚收入 0.6%，形同虚设） |
| stance_knee | — | **−3.0** | 08-25 新增（支撑膝>0.55rad 罚平方，主约束从 base_height 移交至此） |
| gait_symmetry | — | **−20.0** | 08-25 新增（曾试 −5 未提交：Δx=0.067 处仅罚 0.022 形同虚设） |
| orientation | 继承 −1.0 | **−4.0** | 08-25 显式（13° 后仰仅罚 0.05 不够） |
| feet_swing_height | −10.0 | −10.0 | 未变 |
| collision | −0.5 | −0.5 | 未变 |
| dof_pos_limits | −2.5 | −2.5 | 未变 |
| arm_swing | −2.0 | **−10.0** | 08-22（−2 罚金太小，策略交罚不摆） |
| arm_elbow | −1.0 | **−5.0** | 08-22 ×5 |
| arm_spread | −1.0 | **−5.0** | 08-22 ×5 |
| waist_swing | −1.0 | **−5.0** | 08-22 ×5 |
| shoulder_yaw_pos | — | **−5.0** | 08-22 新增（防小臂外翻，model_8550） |
| wrist_pos | — | **−1.0** | 08-22 新增（防漂移到限位） |
| feet_collision | −0.5 | −0.5 | 未变 |
| lin_vel_z | **−1.0**（比 12dof 基类 −2 软，留自然起伏） | **−2.0** | 08-24 收回基类值（抑一蹲一蹲） |
| termination | −250.0 | −250.0 | 未变 |
| only_positive_rewards | False | False | 自初版 |

从未被 23dof 覆盖、一直以 12dof 基类值生效：
`ang_vel_xy −0.05`、`action_rate −0.01`、`hip_pos −4.0`、`contact_no_vel −0.2`、`contact 0.18`、`feet_lateral_align 0.5`、`feet_clearance −8.0`。

## 2. 按提交时间线

| 日期 | 提交 | 内容 |
|---|---|---|
| 08-21 | d2c41f9 添加23自由度代码 | 初版：从 12dof 基类继承，按 23dof 重配（alive 2、lin_vel ×2、dof_vel 放软 5 倍、base_height 放软、碰撞/限位减半、四个臂部姿态罚起步 −1/−2） |
| 08-22 | b9307d6 | 臂部罚 ×5；新增 shoulder_yaw_pos / wrist_pos；tracking_ang_vel 0.5→1.5 |
| 08-24 | f42a716 | base_height −5→−10；lin_vel_z −1→−2 |
| 08-24 | 1547b86 / 351cede | 仅注释整理，无数值变化 |
| 08-25 | 2e62f21 | 新增 stance_knee / gait_symmetry / orientation；base_height −10→−30；dof_vel 拆出 dof_vel_arms |
| 08-27 | 9efd165 | alive 2→1；tracking_ang_vel 1.5→3；dof_vel 合并回 −1e-3（env 重写只收腿+腰）；wrist kd 0.5→2.0（**误诊**：想压"噪声激振"，实为缓冲区伪值，见 pd_damping_stability_zh.md） |
| 08-29 | d65dcc5 | 奖励值与 09-27 版相同（提交时未改 scales） |

## 3. 重训（新 kd 物理下）可复议项

以下调整的**动机建立在旧物理伪影上**，新 kd（肩yaw 1.4 / 腕 0.05）下前提已变：

1. **dof_vel 的 env 重写（只收腿+腰）**——理由是"臂 8 dof 的读数是冻结伪值 41.7"。新物理下臂真的会动、读数真实，是否收回统一税制可复议。
2. **shoulder_yaw_pos / wrist_pos 护栏罚**——当初是给"半焊死"关节设的防漂移护栏；新物理下两关节可控，罚保留合理，但力度（−5 / −1）可重新标定。
3. 9efd165 的 wrist kd 0.5→2.0 已被 β 分析取代（现 0.05），相关注释已更新。
