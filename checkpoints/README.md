# Checkpoint 说明

复现最优 Score 需要：

| 文件 | 用途 | 来源 |
|---|---|---|
| `m10_dense_views2_epoch_002_seed42.pt` | 低密度路由模型 | Picasso9jiu/EVC evsod-main |
| `m26_targetflow_m20e3_epoch_003_seed53.pt` | 高密度基线模型 | Picasso9jiu/EVC evsod-main |
| `query48_e6_seed55.pt` | 48-query e6 seed55 训练产物 | 本次训练 |

`query48_e6_seed55.pt` 训练配置：

- 初始化：M26 epoch 003 seed53
- seed：55
- num_queries：48
- hidden：128
- max_flow：2.0
- loss_weight：0.05
- warmup：3
- epochs：12，取 epoch 6
- 训练命令：`SEED=55 QUERIES=48 bash train_m30_query.sh 12 3`

上传 GitHub 前，请把该 checkpoint 从服务器
`log/m30_query_q48_w0.05_e12_seed55/runs/<run>/epoch_006_seed55.pt`
复制到本目录并重命名为 `query48_e6_seed55.pt`。
