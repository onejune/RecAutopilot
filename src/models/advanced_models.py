#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
advanced_models.py - 高级/复杂模型注册

包含:
- APGNet           : Adaptive Parameter Generation Network（apg）
- PPNet            : Parameter Prediction Network
- FourChannelGateModel : 四通道门控模型
"""
import sys
sys.path.insert(0, "/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python")

from metaspore.algos.apg_net import APGNet
from metaspore.algos.ppnet import PPNet
from metaspore.algos.fg_net import FourChannelGateModel
from model_registry import register


def _ftrl_params(params: dict) -> dict:
    return dict(
        ftrl_l1=params.get('ftrl_l1', 1.0),
        ftrl_l2=params.get('ftrl_l2', 10.0),
        ftrl_alpha=params.get('ftrl_alpha', 0.005),
        ftrl_beta=params.get('ftrl_beta', 0.05),
    )


@register("apg", "APGNet")
def build_apg(params: dict):
    """APGNet：自适应参数生成网络"""
    return APGNet(
        use_wide=params.get('use_wide', False),
        batch_norm=params.get('batch_norm', False),
        net_dropout=params.get('net_dropout', 0),
        wide_embedding_dim=params.get('embedding_size', 8),
        deep_embedding_dim=params.get('embedding_size', 8),
        wide_combine_schema_path=params.get('wide_combine_schema_path'),
        deep_combine_schema_path=params.get('combine_schema_path'),
        apg_hidden_units=params.get('dnn_hidden_units', [512, 256, 64]),
        sparse_optimizer_type=params.get('sparse_optimizer_type', 'adam'),
        **_ftrl_params(params),
    )


@register("ppnet", "PPNet")
def build_ppnet(params: dict):
    """PPNet：参数预测网络，需要额外的 gate_combine_schema_path"""
    return PPNet(
        embedding_dim=params.get('embedding_size', 8),
        combine_schema_path=params.get('combine_schema_path'),
        gate_combine_schema_path=params.get('gate_combine_schema_path',
                                            params.get('combine_schema_path')),
        batch_norm=params.get('batch_norm', False),
        net_dropout=params.get('net_dropout', 0),
        dnn_hidden_units=params.get('dnn_hidden_units', [512, 256, 64]),
        **_ftrl_params(params),
    )


@register("FourChannelGateModel", "fourchannelgate")
def build_four_channel_gate(params: dict):
    """FourChannelGateModel：四通道门控模型"""
    return FourChannelGateModel(
        batch_norm=params.get('batch_norm', False),
        net_dropout=params.get('net_dropout', 0),
        use_wide=params.get('use_wide', False),
        embedding_dim=params.get('embedding_size', 8),
        combine_schema_path=params.get('combine_schema_path'),
        wide_combine_schema_path=params.get('wide_combine_schema_path'),
        dnn_hidden_units=params.get('dnn_hidden_units', [512, 256, 64]),
        dnn_hidden_activations="relu",
        **_ftrl_params(params),
    )
