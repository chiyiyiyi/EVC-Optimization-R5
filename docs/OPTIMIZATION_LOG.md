# 优化实验总览

## 基线

- 仓库：Picasso9jiu/EVC `evsod-main`
- 方案：M10 低密度路由 + M26 + P41 + P0/P0c/P18-global + P6
- val Score：0.9638562171
- Pd 0.9785804284 / IoU 0.9423022866 / Acc 0.9763227701 /
  Fa 4.6632902944e-06

## 后处理方向

### P30 密度/运动组件过滤

- 思路：把队友 0.97 的监督组件分类器改造成无监督规则，删除
  “运动不规则 + 局部密度背景化”的低置信组件。
- 结果：默认与收紧两版均无可删除组件，Score 与基线逐位相同。
- 结论：关闭。当前 M26 输出组件在特征上与真实目标高度重叠，规则无法
  无监督分离。

### P31 运动外推弱轨迹恢复

- 思路：对 seed 支持的轨迹做常数速度外推，恢复高速低事件目标。
- 默认版：Pd +0.0027，但 Fa +0.6e-6、IoU 下降，Score 0.9627206931。
- 保守版：恢复量减半，仍 Score 0.9633163511，低于基线。
- 结论：关闭。每次恢复都在用 IoU/Fa 换 Pd，换不过 Score。

### P32 轨迹质量分数加成

- 思路：对轨迹一致的低置信事件做小幅分数加成，不新增外推事件。
- 参数平台：candidate_floor=0.60, bonus=0.010, min_seed_components=2,
  min_track_bins=4, max_motion_residual=2.0, max_score_cap=0.97。
- P32v7：Score 0.9639901960（+0.000134），三次复跑一致。
- 结论：采纳为后处理增益，与后续训练侧改进叠加。

## 推理侧 TTA

### P54 时间反转

- Score 与 P32v7 完全相同，无增益。
- 结论：关闭。

### P50 三相位

- Score 0.9626172665。
- 结论：关闭。

### P44 空间相位

- Score 0.9609068874。
- 结论：关闭。

## 训练侧：轨迹感知焦点损失（M28）

- 思路：用 target_id 聚轨迹，对轨迹低分真事件加权、惩罚轨迹附近高分背景。
- A（weight 0.05）最优 0.9638808979；
- B（weight 0.10）最优 0.9638450222；
- C（radius 8 / sigma 5）最优 0.9638450222。
- 结论：均低于 P32v7，关闭。

## 训练侧：数据增强/采样（M29）

### D cross-video copy-paste

- 最优 epoch 5，Score 0.9638941627。

### E trajectory augmentation

- 最优 epoch 2，Score 0.9638097017。

### F motion sampling

- 最优 epoch 5，Score 0.9638553020。

- 结论：均低于 P32v7，关闭。

## 训练侧：对象级 TrackQueryHead（M30）

### 机制

- 在 M26 bottleneck 上挂接可学习 query；
- query 通过交叉注意力输出 objectness、逐帧中心、速度和尺度；
- 零初始化残差保证挂接瞬间预测不变；
- 训练损失：objectness BCE + center SmoothL1 + velocity SmoothL1，
  使用最近锚点贪心匹配。

### 32-query

- seed54 e6：0.9640400468（两次复跑一致）；
- seed55 e6：0.9639460212；
- 结论：seed54 增益未获交叉确认，32-query 不作为最终方案。

### 48-query

- seed54 e6：0.9640377302；
- seed55 e6：0.9640935522（两次复跑一致）；
- seed55 e4/e5/e7/e8：0.96406 / 0.96397 / 0.96403 / 0.96398，
  e6 为平台峰点；
- 结论：两个 seed 均超过 0.96399，交叉确认成立，采纳
  **48-query e6 seed55**。

## 最终采纳配置

```text
M26 + P41
+ TrackQueryHead(num_queries=48, hidden=128, max_flow=2.0) epoch 6 seed55
+ P32v7 轨迹质量分数加成
```

Score 0.9640935522。
