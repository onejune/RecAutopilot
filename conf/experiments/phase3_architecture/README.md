# Phase 3 - 模型架构探索

## 目标
对比不同模型架构在 ivr_sample_v16 数据上的效果。

## Baseline
- **deeper_dnn**（WideDeep, use_wide=False）: AUC=0.8354, PCOC=1.0798

## 实验列表

| 实验 | 模型 | 假设 | 状态 |
|------|------|------|------|
| exp_deepfm | DeepFM | FM 二阶交叉更充分 | 待运行 |
| exp_dcn | DCN | 显式高阶特征交叉 | 待运行 |
| exp_masknet | MaskNet | 特征掩码减少噪音 | 待运行 |
| exp_widedeep_wide | WideDeep(wide=True) | Wide 部分参与训练 | 待运行 |

## 启动方式
```bash
bash scripts/run_batch.sh conf/plans/phase3_architecture.yaml
```

## 注意
- 所有实验使用相同特征集（baseline 175个特征）
- 训练数据：2026-03-08 ~ 2026-03-10，验证：2026-03-11
- 每组实验约 110~120 分钟
