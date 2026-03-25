# RecAutoPilot

> **AI Agent 驱动的推荐模型自动化研究框架**

类似 Karpathy 的 [autoresearch](https://github.com/karpathy/autoresearch)，但针对**推荐系统 CTR 预估**场景。

你睡觉，Agent 帮你跑实验、记录结果、积累研究洞察。

---

## 项目定位

传统推荐模型研究需要人工反复调整特征、架构、超参，效率低下。RecAutoPilot 的目标是：

- 让 AI Agent 自动探索**特征筛选、特征交叉、模型架构、超参设置**
- 每个实验固定时间预算（约 20-30 分钟），自动评估并记录结果
- 维护实验排行榜和研究洞察，持续积累知识
- 最终找到比 baseline 提升 1%+ AUC 的最优配置

---

## 技术栈

- **训练框架**：MetaSpore（PyTorch + PySpark，分布式 PS）
- **支持模型**：WideDeep、DeepFM、DCN、MaskNet、FFM、APG、FwFM、PPNet
- **评估指标**：AUC、PCOC、LogLoss（按 business_type 分组）
- **数据**：`/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v16/parquet/YYYY-MM-DD`

---

## 目录结构

```
RecAutoPilot/
├── src/                    # 核心代码
│   ├── autopilot_runner.py # 实验运行器
│   ├── base_trainFlow.py   # 训练流程基类
│   ├── dnn_trainFlow.py    # DNN 训练流程
│   ├── movas_logger.py     # 日志工具
│   └── feishu_notifier.py  # 通知工具
├── conf/
│   ├── base.yaml           # 基础配置（Spark、数据路径等）
│   ├── combine_schema      # 特征配置（Agent 可改）
│   ├── column_name         # 特征全集（只读参考）
│   └── experiments/        # 实验配置目录
│       ├── phase1_baseline/    # Phase 1：基础架构
│       ├── phase2_feature/     # Phase 2：特征筛选
│       ├── phase3_arch/        # Phase 3：架构探索
│       ├── phase4_hyperparam/  # Phase 4：超参优化
│       └── phase5_combined/    # Phase 5：组合超参
├── experiments/            # 实验记录（config + metrics）
├── devlog/                 # 每日开发日志
├── log/                    # 实验运行日志
├── output/                 # 模型输出（各实验独立目录）
├── leaderboard.json        # 实验排行榜
├── insights.md             # 研究洞察（Agent 维护）
├── run_experiments.sh      # Phase 1 运行脚本
├── run_fea_experiments.sh  # Phase 2 运行脚本
├── run_phase4_experiments.sh # Phase 4 运行脚本
├── run_phase5_experiments.sh # Phase 5 运行脚本
└── DESIGN.md               # 详细设计文档
```

---

## 快速开始

```bash
cd /mnt/workspace/open_research/rec-autopilot

# Phase 1：基础架构实验
bash run_experiments.sh

# Phase 2：特征筛选实验
bash run_fea_experiments.sh

# Phase 4：超参优化实验
bash run_phase4_experiments.sh

# Phase 5：组合超参实验
bash run_phase5_experiments.sh

# 查看排行榜
cat leaderboard.json
```

---

## 实验进展（截至 2026-03-25）

### 🏆 最优配置

```yaml
model_type: WideDeep
dnn_hidden_units: [1024, 512, 256, 128]
embedding_size: 8
adam_learning_rate: 5.0e-05
net_dropout: 0.3
batch_size: 256
```

**AUC：0.8374**（vs 起点 0.8212，总提升 **+0.0162**）

---

### Phase 1 — 基础架构

| 实验 | 模型 | AUC | PCOC | 备注 |
|------|------|-----|------|------|
| baseline_v2 | WideDeep [512,256,64] emb=8 | 0.8212 | 1.1724 | 起点基准 |
| deeper_dnn | WideDeep [1024,512,256,128] emb=8 | **0.8354** | 1.0798 | +0.0142 ↑ |
| wider_emb16 | WideDeep [1024,512,256] emb=16 | 0.8342 | 1.0659 | +0.0130 ↑ |

### Phase 2 — 特征筛选

| 实验 | 说明 | AUC | Δ |
|------|------|-----|---|
| fea_A | 去掉20个外部行为特征 | 0.8349 | -0.0005 |
| fea_B | 去掉15d时间窗口特征 | 0.8337 | -0.0017 |
| fea_C | 增加9个交叉特征 | 0.8352 | -0.0002 |
| fea_D | 去掉ruf2_*实时特征 | 0.8347 | -0.0007 |

> 结论：特征删减均有损失，现有特征体系保持不变。

### Phase 3 — 架构探索

| 实验 | 模型 | AUC | Δ |
|------|------|-----|---|
| arch_deepfm | DeepFM | 0.8251 | -0.0103 ❌ |
| arch_masknet | MaskNet | 0.8325 | -0.0029 ❌ |
| arch_dcn | DCN | 0.8344 | -0.0010 |
| arch_widedeep_wide | WideDeep(wide增强) | 0.8348 | -0.0006 |

> 结论：WideDeep deeper_dnn 仍是最优架构。

### Phase 4 — 超参优化

| 实验 | 变化 | AUC | Δ |
|------|------|-----|---|
| hp_lr_5e5 | lr=5e-5 | 0.8373 | **+0.0019 ✅** |
| hp_lr_1e4 | lr=1e-4 | 0.8371 | +0.0017 ✅ |
| hp_dropout03 | dropout=0.3 | 0.8360 | +0.0006 ✅ |
| hp_emb12/16/24 | emb扩大 | 0.8345~0.8348 | ≈0 |
| hp_dropout07 | dropout=0.7 | 0.8333 | -0.0021 ❌ |

> 结论：学习率提升有效，embedding扩大无收益，dropout降低略有提升。

### Phase 5 — 组合超参

| 实验 | 组合 | AUC | Δ |
|------|------|-----|---|
| comb_lr5e5_drop03 | lr=5e-5 + dropout=0.3 | **0.8374** | **+0.0020 🏆** |
| comb_lr5e5_emb16 | lr=5e-5 + emb=16 | 0.8367 | +0.0013 ✅ |

> 结论：两个正向因子叠加有效，lr=5e-5 + dropout=0.3 为当前最优。

---

## AUC 提升路径

```
0.8212 (baseline_v2)
  → 0.8354 (deeper_dnn, Phase 1, +0.0142)
    → 0.8373 (hp_lr_5e5, Phase 4, +0.0019)
      → 0.8374 (comb_lr5e5_drop03, Phase 5, +0.0001)
```

---

## 开发规范

- 代码开发和实验在 `/mnt/workspace/open_research/rec-autopilot/`
- 所有改动及时同步到本仓库并 push
- 每日开发进度记录在 `devlog/YYYY-MM-DD.md`
- 实验配置：yaml + 同名子目录/readme（飞书通知用）
- 并行实验必须使用独立 `--output_dir`，避免 meta 文件冲突
