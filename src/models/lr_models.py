#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lr_models.py - LR/FTRL 系列模型注册

包含:
- LRFtrl  : 标准 LR + FTRL 优化器
- LRFtrl2 : Adam 版本
- LRFtrl3 : 简化版本
"""
import sys
sys.path.insert(0, "/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python")

from metaspore.algos.lr_ftrl_net import LRFtrl, LRFtrl2, LRFtrl3
from model_registry import register


@register("LRFtrl", "lrftrl")
def build_lrftrl(params: dict):
    """标准 LR + FTRL 优化器"""
    return LRFtrl(
        batch_norm=params.get('batch_norm', False),
        net_dropout=params.get('net_dropout', 0),
        wide_embedding_dim=params.get('embedding_size', 8),
        wide_combine_schema_path=params.get('wide_combine_schema_path'),
        ftrl_l1=params.get('ftrl_l1', 1.0),
        ftrl_l2=params.get('ftrl_l2', 10.0),
        ftrl_alpha=params.get('ftrl_alpha', 0.005),
        ftrl_beta=params.get('ftrl_beta', 0.05),
    )


@register("LRFtrl2", "lrftrl2")
def build_lrftrl2(params: dict):
    """LRFtrl2：使用 Adam 优化器"""
    return LRFtrl2(
        wide_embedding_dim=params.get('embedding_size', 8),
        wide_combine_schema_path=params.get('wide_combine_schema_path'),
        adam_learning_rate=params.get('adam_learning_rate', 1e-5),
    )


@register("LRFtrl3", "lrftrl3")
def build_lrftrl3(params: dict):
    """LRFtrl3：简化版 LR"""
    return LRFtrl3(
        wide_embedding_dim=params.get('embedding_size', 8),
        wide_combine_schema_path=params.get('wide_combine_schema_path'),
        learning_rate=params.get('adam_learning_rate', 1e-5),
    )
