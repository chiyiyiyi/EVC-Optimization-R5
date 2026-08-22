# 复现说明

## 环境

与 Picasso9jiu/EVC `evsod-main` 相同：

- Python 3.9 / PyTorch 1.9.1+cu111（服务器实测也可用
  Python 3.8 + torch 1.13.1 + cu116 环境跑通）
- spconv、HAIS_OP 编译通过
- `env_round4.sh` 设置 CUDA_HOME / PYTHONPATH / DATA_ROOT

## 1. 准备基线

```bash
git clone --branch evsod-main https://github.com/Picasso9jiu/EVC.git EVSOD-main
cd EVSOD-main
```

## 2. 拷贝本仓库代码

```bash
cp model/object_track_head.py EVSOD-main/model/
cp utils/object_track_loss.py EVSOD-main/utils/
cp utils/adaptive_postprocess.py EVSOD-main/utils/
cp apply_query_head_patch.py EVSOD-main/
cp apply_round4_patch.py EVSOD-main/
cp eval_checkpoint_p32v7.sh EVSOD-main/
cp run_submit_query48_seed55.sh EVSOD-main/
```

## 3. 打补丁

```bash
cd EVSOD-main
python apply_round4_patch.py
python apply_query_head_patch.py
```

> 若 `apply_query_head_patch.py` 因服务器代码版本差异报 marker 缺失，
> 参考仓库内对应的手工修复记录；本仓库代码基于服务器实际代码生成。

## 4. checkpoint

需要三个 checkpoint：

- `checkpoints/m10_dense_views2_epoch_002_seed42.pt`
- `checkpoints/m26_targetflow_m20e3_epoch_003_seed53.pt`
- `checkpoints/query48_e6_seed55.pt`（48-query e6 seed55 训练产物）

前两个来自 Picasso 基线仓库；第三个为本次训练产物，见
`checkpoints/README.md`。

## 5. 验证

```bash
bash eval_checkpoint_p32v7.sh checkpoints/query48_e6_seed55.pt
```

预期：

```text
Score: 0.9640935522
Pd:    0.9785804284
IoU:   0.9433465004
Acc:   0.9778798819
Fa:    4.7077307991e-06
```

## 6. 提交生成

```bash
bash run_submit_query48_seed55.sh
```

输出：`log/challenge2/query48_e6_seed55_final.zip`（24 个 val_*.txt）。
