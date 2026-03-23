# RecAutoPilot 研究计划

> Agent 遵循此文件进行自动化研究。人类可随时修改研究方向。

---

## 当前阶段：Baseline 建立

### 目标
建立可靠的 baseline 模型，作为后续实验的对比基准。

### 当前配置
- **模型**: WideDeep (use_wide=False)
- **Embedding**: 8 维
- **DNN**: [512, 256, 64]
- **Dropout**: 0.5
- **数据**: 3 天训练 + 1 天验证

### 约束
- 每个实验最多运行 30 分钟
- 必须记录所有实验结果到 experiments/ 目录
- 更新 leaderboard.json 排行榜

### 下一步计划

#### Phase 1: Baseline 验证
1. 运行当前配置，确认流程正常
2. 记录 baseline AUC/PCOC 指标

#### Phase 2: 特征探索
1. 删除 IPUA 特征，观察效果变化
2. 尝试不同的特征交叉组合
3. 精简特征集，提升训练效率

#### Phase 3: 模型架构探索
1. 尝试 DeepFM 模型
2. 尝试 DCN 模型
3. 尝试 MaskNet 模型
4. 比较不同架构的效果

#### Phase 4: 超参优化
1. 调整 embedding_size: 8 → 12 → 16
2. 调整 DNN 层数和宽度
3. 调整学习率和正则化参数

### 完成标准
- 建立稳定的 baseline（AUC 方差 < 0.5%）
- 找到比 baseline 提升 1%+ AUC 的配置
- 生成至少 20 个有效实验记录

---

## 实验操作指南

### 修改配置
1. 编辑 `conf/experiment.yaml` 修改模型和超参
2. 编辑 `conf/combine_schema` 修改特征配置

### 运行实验
```bash
cd /mnt/workspace/open_research/rec-autopilot
bash scripts/run_experiment.sh [实验名] [评估键] [假设]
```

### 查看结果
```bash
# 查看排行榜
cat leaderboard.json

# 查看实验详情
ls experiments/
cat experiments/exp_XXX/metrics.json
```

### 停止实验
```bash
rm train_interrupt.flag
```

---

## 研究记录

### 实验日志
| 日期 | 实验ID | 假设 | 结果 | 结论 |
|------|--------|------|------|------|
| - | - | - | - | - |

### 关键发现
- _(待补充)_

### 待验证假设
- [ ] 删除 IPUA 特征是否影响效果
- [ ] DeepFM 是否优于 WideDeep
- [ ] 更大的 embedding 是否有帮助
- [ ] 更深的 DNN 是否有帮助
