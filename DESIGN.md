# RecAutoPilot - 推荐系统自动化研究框架

> 让 AI Agent 自动探索推荐模型的特征、架构、超参，基于 MetaSpore 框架。

## 1. 项目概述

### 1.1 目标
- 自动化探索特征筛选、特征组合、模型架构、超参设置
- 每个实验固定时间预算（约 20-30 分钟）
- 自动记录实验结果，维护排行榜和研究洞察

### 1.2 技术栈
- **训练框架**: MetaSpore (PyTorch + PySpark)
- **数据**: ivr_sample_v7 parquet 数据集
- **计算资源**: CPU (6 workers, 2 servers)
- **评估指标**: AUC, PCOC, LogLoss

---

## 2. 数据说明

### 2.1 数据路径
```
/mnt/data/oss_dsp_algo/ivr/sample/ivr_sample_v7/parquet/part=YYYY-MM-DD/
```

### 2.2 业务过滤
只保留以下 `business_type`:
- `shein`
- `ae*` (aliexpress)
- `shopee*`
- `lazada*`

### 2.3 采样策略
- 正样本：全部保留
- 负样本：shein 采样 1%，其他采样 10%

### 2.4 可用特征
见 `conf/column_name`，共 633 个特征（含辅助列）

**特征分类**:
| 类别 | 前缀 | 数量 | 说明 |
|------|------|------|------|
| 基础特征 | - | ~30 | adid, bundle, country 等 |
| 内部设备行为 | duf_inner_dev_* | ~200 | 多时间窗口 |
| 外部设备行为 | duf_outer_dev_* | ~100 | 多时间窗口 |
| 实时特征 | huf_*, ruf2_* | ~100 | 小时级行为 |
| IPUA 特征 | duf_inner_ipua_* | ~100 | **可删除** |
| 设备统计 | duf_dev_* | ~100 | 设备级聚合 |

---

## 3. 项目结构

```
rec-autopilot/
├── DESIGN.md               # 本文档
├── program.md              # Agent 研究计划（人类编写）
├── conf/
│   ├── base.yaml           # 基础配置（固定，不改）
│   ├── combine_schema      # 默认特征配置
│   ├── column_name         # 特征全集（只读参考）
│   ├── experiments/        # 各阶段实验配置
│   │   ├── phase1_baseline/    # Phase 1: 基准实验
│   │   ├── phase2_feature/     # Phase 2: 特征筛选
│   │   ├── phase3_architecture/ # Phase 3: 架构探索
│   │   └── phase4_hyperparam/  # Phase 4: 超参优化
│   └── plans/              # 批量实验计划（YAML）
│       ├── phase2_feature.yaml
│       ├── phase3_architecture.yaml
│       └── phase4_hyperparam.yaml
├── src/
│   ├── base_trainFlow.py   # 训练基类（核心，不改）
│   ├── dnn_trainFlow.py    # DNN 训练流程（轻量，调注册表）
│   ├── model_registry.py   # 模型注册表（新增模型在此注册）
│   ├── models/             # 模型实现（每类一个文件）
│   │   ├── widedeep_models.py  # WideDeep, WideDeep2
│   │   ├── lr_models.py        # LRFtrl, LRFtrl2, LRFtrl3
│   │   ├── interaction_models.py # DeepFM, DCN, FFM, FwFM, MaskNet
│   │   └── advanced_models.py  # APGNet, PPNet, FourChannelGateModel
│   ├── autopilot_runner.py # 实验运行器（入口）
│   ├── metrics_eval.py     # 评估工具
│   ├── movas_logger.py     # 日志工具
│   └── feishu_notifier.py  # 通知工具
├── scripts/
│   ├── run_batch.sh        # 通用批量启动脚本（主入口）
│   ├── batch_runner.py     # Python 批量调度器
│   ├── summarize_results.py # 实验结果汇总
│   ├── run_experiment.sh   # 单次实验脚本（兼容保留）
│   └── init_env.sh         # 环境初始化
├── experiments/            # 实验记录目录（自动生成）
├── leaderboard.json        # 实验排行榜（自动更新）
├── log/                    # 实验日志目录
└── insights.md             # 研究洞察（Agent 维护）
```

---

## 4. Agent 可探索的维度

### 4.1 特征探索 (conf/combine_schema)

| 探索方向 | 具体操作 |
|----------|----------|
| 特征筛选 | 删除低效特征（如 IPUA 特征） |
| 特征组合 | 添加/删除交叉特征（用 # 分隔） |
| 时间窗口 | 选择不同时间窗口的特征版本 |

**示例**:
```
# 单特征
campaignid
country
bundle

# 交叉特征
bundle#country
campaignid#adx
```

### 4.2 模型架构探索 (conf/experiment.yaml)

| 参数 | 可选值 | 说明 |
|------|--------|------|
| model_type | WideDeep, DeepFM, DCN, MaskNet, FFM, APG, FwFM, PPNet | 模型类型 |
| embedding_size | 4, 8, 12, 16, 24, 32 | Embedding 维度 |
| dnn_hidden_units | [256,128], [512,256,64], [1024,512,256,64] | DNN 层配置 |
| use_wide | True, False | 是否使用 Wide 部分 |
| net_dropout | 0.0, 0.1, 0.3, 0.5 | Dropout 比率 |
| batch_norm | True, False | 是否使用 BatchNorm |

### 4.3 超参探索 (conf/experiment.yaml)

| 参数 | 范围 | 说明 |
|------|------|------|
| adam_learning_rate | 1e-5 ~ 1e-3 | Adam 学习率 |
| ftrl_l1 | 0.1 ~ 10.0 | FTRL L1 正则 |
| ftrl_l2 | 1.0 ~ 100.0 | FTRL L2 正则 |
| batch_size | 128, 256, 512, 1024 | 批大小 |

---

## 5. 实验流程

### 5.1 单次实验流程
```
1. Agent 修改 conf/experiment.yaml 和/或 conf/combine_schema
2. 运行 scripts/run_experiment.sh
3. 训练 3 天数据 + 验证 1 天数据
4. 记录结果到 experiments/exp_XXX/
5. 更新 leaderboard.json
6. Agent 分析结果，更新 insights.md
```

### 5.2 时间预算
- 训练数据：3 天
- 验证数据：1 天
- 单次实验：约 20-30 分钟
- 一晚上（8小时）：可跑 16-24 个实验

### 5.3 评估指标
- **主指标**: AUC（分 business_type 计算）
- **辅助指标**: PCOC, LogLoss
- **约束**: 正样本数 >= 50 才计入排行

---

## 6. 实验记录格式

### 6.1 experiments/exp_XXX/config.json
```json
{
  "exp_id": "exp_001",
  "timestamp": "2026-03-22T18:00:00",
  "hypothesis": "增大 embedding_size 可能提升效果",
  "changes": {
    "embedding_size": {"from": 8, "to": 16}
  },
  "config_snapshot": { ... }
}
```

### 6.2 experiments/exp_XXX/metrics.json
```json
{
  "overall": {"auc": 0.7823, "pcoc": 1.02, "logloss": 0.45},
  "by_business_type": {
    "shein": {"auc": 0.7912, "pcoc": 0.98, "pos": 12345, "neg": 234567},
    "ae_xxx": {"auc": 0.7654, "pcoc": 1.05, "pos": 5678, "neg": 98765}
  },
  "train_time_minutes": 25.3,
  "model_path": "./output/model_2026-03-22"
}
```

### 6.3 leaderboard.json
```json
{
  "best_overall_auc": {
    "exp_id": "exp_015",
    "auc": 0.7912,
    "config_summary": "DeepFM, emb=16, dnn=[512,256,64]"
  },
  "experiments": [
    {"exp_id": "exp_015", "auc": 0.7912, "pcoc": 0.99, "model_type": "DeepFM"},
    {"exp_id": "exp_008", "auc": 0.7856, "pcoc": 1.01, "model_type": "WideDeep"},
    ...
  ]
}
```

---

## 7. Agent 研究计划模板 (program.md)

```markdown
# 当前研究阶段：XXX

## 目标
...

## 约束
- 每个实验最多 30 分钟
- AUC 下降不能超过 baseline 的 1%

## 探索策略
1. ...
2. ...

## 完成标准
- ...
```

---

## 8. 环境配置

### 8.1 Python 环境
```bash
PYTHON_ENV="/root/anaconda3/envs/spore/bin/python"
```

### 8.2 MetaSpore 路径
```bash
METASPORE_DIR="/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python"
```

### 8.3 Spark 配置
```yaml
local_spark: local
worker_count: 6
server_count: 2
worker_memory: '8G'
server_memory: '8G'
coordinator_memory: '10G'
batch_size: 256
```

---

## 9. 快速开始

### 9.1 运行批量实验（推荐）
```bash
cd /mnt/workspace/open_research/rec-autopilot

# 运行 Phase 3 架构探索（4个实验，2组并行）
bash scripts/run_batch.sh conf/plans/phase3_architecture.yaml

# 运行 Phase 4 超参优化
bash scripts/run_batch.sh conf/plans/phase4_hyperparam.yaml
```

### 9.2 查看实验结果
```bash
# 汇总所有实验，按 AUC 排序
python scripts/summarize_results.py

# 只看 Phase 3 架构实验
python scripts/summarize_results.py --filter arch

# 对比 top 10，以 deeper_dnn 为 baseline
python scripts/summarize_results.py --top 10 --baseline deeper_dnn
```

### 9.3 新增模型
只需在 `src/models/` 下创建新文件并注册：
```python
# src/models/my_models.py
from model_registry import register
from metaspore.algos.my_net import MyModel

@register("MyModel", "mymodel")
def build_mymodel(params: dict):
    return MyModel(
        embedding_dim=params.get('embedding_size', 8),
        # ... 其他参数
    )
```
然后在 yaml 中指定 `model_type: MyModel` 即可，无需修改其他文件。

### 9.4 新增实验
1. 在 `conf/experiments/phaseX/` 下创建 yaml 文件
2. 在 `conf/plans/` 下的计划文件中添加实验条目
3. 运行 `bash scripts/run_batch.sh conf/plans/phaseX.yaml`

---

## 10. 注意事项

1. **不要修改** `conf/column_name`，这是特征全集参考
2. **不要修改** `src/base_trainFlow.py` 的核心逻辑
3. 每次实验前确保 `train_interrupt.flag` 文件存在
4. 实验失败时检查 `log/` 目录下的日志
