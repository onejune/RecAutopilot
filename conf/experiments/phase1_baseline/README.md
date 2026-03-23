# Phase 1: Baseline 建立

目标：建立可靠的 baseline，探索基础模型架构。

| 实验 | 模型 | DNN | Embedding | AUC | 备注 |
|------|------|-----|-----------|-----|------|
| baseline_v2 | WideDeep | [512,256,64] | 8 | 0.8212 | 基准线 |
| deeper_dnn  | WideDeep | [1024,512,256,128] | 8 | 0.8354 | +0.0142 ↑ |
| wider_emb16 | WideDeep | [1024,512,256] | 16 | 进行中 | - |

**结论（待补充）**
