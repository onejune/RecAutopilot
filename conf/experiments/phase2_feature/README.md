# Phase 2: 特征筛选实验

目标：通过增删特征，找到最优特征集合。基于 deeper_dnn 架构（AUC最优）做对照。

**baseline 参考：deeper_dnn AUC=0.8354**

| 实验 | 假设 | 特征变化 | AUC | PCOC | 结论 |
|------|------|----------|-----|------|------|
| fea_no_outer_dev | 外部行为特征贡献低 | 去掉20个 duf_outer_dev_* | - | - | 待跑 |
| fea_short_window | 近期行为更有价值 | 去掉19个 _15d 特征 | - | - | 待跑 |
| fea_more_cross   | 更多交叉能捕捉模式 | 新增9个基础交叉 | - | - | 待跑 |
| fea_no_ruf2      | 验证实时流特征贡献 | 去掉15个 ruf2_* | - | - | 待跑 |

## combine_schema 说明
- `exp_fea_A/combine_schema` — 155 行（去掉 duf_outer_dev_*）
- `exp_fea_B/combine_schema` — 156 行（去掉 _15d）
- `exp_fea_C/combine_schema` — 184 行（新增交叉）
- `exp_fea_D/combine_schema` — 160 行（去掉 ruf2_*）
