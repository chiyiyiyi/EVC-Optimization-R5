# EVC Optimization R5：TrackQueryHead + P32v7

> 仓库：<https://github.com/chiyiyiyi/EVC-Optimization-R5>

## 核心三点

### 1. 最优得分

| 版本 | val Score | 说明 |
|---|---:|---|
| 官方基线（M26 + P41） | 0.9638562171 | Picasso9jiu/EVC `evsod-main` 分支 |
| 上一轮后处理最优（P32v7） | 0.9639901960 | 轨迹质量分数加成 |
| **本轮最优（TrackQueryHead 48-query e6 seed55 + P32v7）** | **0.9640935522** | 相对官方基线 **+0.000237** |

最优完整指标：

| Pd | IoU | Acc | Fa | Score |
|---:|---:|---:|---:|---:|
| 0.9785804284 | 0.9433465004 | 0.9778798819 | 4.7077307991e-06 | **0.9640935522** |

### 2. 核心改进思路

**对象级 TrackQueryHead + 轨迹质量分数加成（P32v7）**

1. 在 M26 的 bottleneck 序列上挂接一组可学习 query，通过交叉注意力输出
   每条轨迹的 objectness、逐帧中心、速度和尺度；
2. 输出以零初始化残差接入 bottleneck，挂接时 M26 预测逐位不变；
3. 训练阶段用 `track_query_loss` 对 query 做对象级监督，直接提升
   高速、小尺度、低事件量目标的轨迹建模；
4. 推理阶段保留 M26 + P41 相位集成，再叠加 P32v7 轨迹质量分数加成，
   只对轨迹一致的低置信事件做小幅加分。

### 3. 复现方式

详细步骤见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。

简要流程：

```bash
# 1. 准备基线与本仓库
git clone --branch evsod-main https://github.com/Picasso9jiu/EVC.git EVSOD-main
cd EVSOD-main
cp -r <本仓库>/model/object_track_head.py model/
cp -r <本仓库>/utils/object_track_loss.py utils/
cp -r <本仓库>/utils/adaptive_postprocess.py utils/
cp <本仓库>/apply_query_head_patch.py .
cp <本仓库>/apply_round4_patch.py .

# 2. 打补丁
python apply_round4_patch.py
python apply_query_head_patch.py

# 3. 放置 checkpoint 到 checkpoints/
#    m26_targetflow_m20e3_epoch_003_seed53.pt
#    m10_dense_views2_epoch_002_seed42.pt
#    query48_e6_seed55.pt（见 checkpoints/README.md）

# 4. 验证最优 Score
bash eval_checkpoint_p32v7.sh checkpoints/query48_e6_seed55.pt

# 5. 生成提交
bash run_submit_query48_seed55.sh
```

## 全部尝试方向

| 方向 | 结果 | 决策 |
|---|---|---|
| P30 密度/运动组件过滤 | 无可删除组件，Score 不变 | 关闭 |
| P31 运动外推弱轨迹恢复 | Pd 升但 Fa/IoU 恶化 | 关闭 |
| P32 轨迹质量分数加成 | P32v7 Score 0.9639901960 | 采纳 |
| P54 时间反转 TTA | 无增益 | 关闭 |
| P50 三相位 TTA | 负收益 | 关闭 |
| P44 空间相位 TTA | 负收益 | 关闭 |
| Track-Aware Focal Loss | 3 组超参均负收益 | 关闭 |
| 数据增强（cross-paste / traj-aug / motion-sampling） | 3 组均负收益 | 关闭 |
| TrackQueryHead 32-query | seed54 高但 seed55 不确认 | 关闭 |
| **TrackQueryHead 48-query e6 seed55** | **0.9640935522，双 seed 确认** | **采纳** |

详细实验记录见 [docs/OPTIMIZATION_LOG.md](docs/OPTIMIZATION_LOG.md)。

## 泛化性保障

- 所有参数为通用规则，不依赖视频名、目标 ID 或标签做推理分支；
- 训练随机种子固定（seed 54/55），关键结论用独立 seed 复训确认；
- 最优 checkpoint 的 e4-e8 平台验证，非单点；
- 未读取测试集，所有选择基于 val 24 视频。
