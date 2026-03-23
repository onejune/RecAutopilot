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
├── experiments/            # 实验记录（config + metrics）
├── devlog/                 # 每日开发日志
├── scripts/                # 运行脚本
├── leaderboard.json        # 实验排行榜
├── insights.md             # 研究洞察（Agent 维护）
├── program.md              # Agent 研究计划（人类编写）
└── DESIGN.md               # 详细设计文档
```

---

## 快速开始

```bash
cd /mnt/workspace/open_research/rec-autopilot

# 运行批量实验
bash run_experiments.sh

# 查看结果
cat leaderboard.json
```

---

## 当前进展

| 实验 | 模型 | AUC | 备注 |
|------|------|-----|------|
| baseline_v2 | WideDeep [512,256,64] emb=8 | 0.8212 | 基准线 |
| deeper_dnn  | WideDeep [1024,512,256,128] emb=8 | 0.8354 | +0.0142 ↑ |
| wider_emb16 | WideDeep [1024,512,256] emb=16 | 进行中 | - |

详见 [devlog/](devlog/) 和 [DESIGN.md](DESIGN.md)。

---

## 开发规范

- 代码开发和实验在 `/mnt/workspace/open_research/rec-autopilot/`
- 所有改动及时同步到本仓库并 push
- 每日开发进度记录在 `devlog/YYYY-MM-DD.md`
