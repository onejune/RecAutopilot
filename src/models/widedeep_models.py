#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
widedeep_models.py - WideDeep 系列模型注册

包含:
- WideDeep  : 标准 Wide&Deep
- WideDeep2 : Adam 优化器版本
"""
import sys
sys.path.insert(0, "/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python")

from metaspore.algos.widedeep_net import WideDeep, WideDeep2
from model_registry import register


def _common_params(params: dict) -> dict:
    """提取 WideDeep 系列公共参数"""
    return dict(
        use_wide=params.get('use_wide', False),
        batch_norm=params.get('batch_norm', False),
        net_dropout=params.get('net_dropout', 0),
        wide_embedding_dim=params.get('embedding_size', 8),
        deep_embedding_dim=params.get('embedding_size', 8),
        wide_combine_schema_path=params.get('wide_combine_schema_path'),
        deep_combine_schema_path=params.get('combine_schema_path'),
        dnn_hidden_units=params.get('dnn_hidden_units', [512, 256, 64]),
        dnn_hidden_activations="relu",
        ftrl_l1=params.get('ftrl_l1', 1.0),
        ftrl_l2=params.get('ftrl_l2', 10.0),
        ftrl_alpha=params.get('ftrl_alpha', 0.005),
        ftrl_beta=params.get('ftrl_beta', 0.05),
    )


@register("WideDeep", "widedeep")
def build_widedeep(params: dict):
    """标准 Wide&Deep 模型"""
    return WideDeep(**_common_params(params))


@register("WideDeep2", "widedeep2")
def build_widedeep2(params: dict):
    """WideDeep2：使用 Adam 优化器的 Wide&Deep"""
    p = _common_params(params)
    p['adam_learning_rate'] = params.get('adam_learning_rate', 1e-5)
    return WideDeep2(**p)
