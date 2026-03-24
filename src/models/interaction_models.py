#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
interaction_models.py - 特征交叉/交互类模型注册

包含:
- DeepFM    : FM + DNN
- DCN       : Deep & Cross Network
- FFM       : Field-aware Factorization Machine
- FwFM      : Field-weighted Factorization Machine
- MaskNet   : 特征掩码网络（masknet）
"""
import sys
sys.path.insert(0, "/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python")

from metaspore.algos.deepfm_net import DeepFM
from metaspore.algos.dcn_net import DCN
from metaspore.algos.ffm_net import FFM
from metaspore.algos.fwfm_net import FwFM
from metaspore.algos.maskNet import MaskNet
from model_registry import register


def _ftrl_params(params: dict) -> dict:
    """提取 FTRL 相关公共参数"""
    return dict(
        ftrl_l1=params.get('ftrl_l1', 1.0),
        ftrl_l2=params.get('ftrl_l2', 10.0),
        ftrl_alpha=params.get('ftrl_alpha', 0.005),
        ftrl_beta=params.get('ftrl_beta', 0.05),
    )


@register("DeepFM", "deepfm")
def build_deepfm(params: dict):
    """DeepFM：FM 交叉 + DNN"""
    return DeepFM(
        use_wide=params.get('use_wide', False),
        batch_norm=params.get('batch_norm', False),
        net_dropout=params.get('net_dropout', 0),
        wide_embedding_dim=params.get('embedding_size', 8),
        deep_embedding_dim=params.get('embedding_size', 8),
        wide_combine_schema_path=params.get('wide_combine_schema_path'),
        deep_combine_schema_path=params.get('combine_schema_path'),
        dnn_hidden_units=params.get('dnn_hidden_units', [512, 256, 64]),
        dnn_hidden_activations="relu",
        **_ftrl_params(params),
    )


@register("DCN", "dcn")
def build_dcn(params: dict):
    """DCN：Deep & Cross Network，显式特征交叉"""
    return DCN(
        use_wide=params.get('use_wide', False),
        batch_norm=params.get('batch_norm', False),
        net_dropout=params.get('net_dropout', 0),
        wide_embedding_dim=params.get('embedding_size', 8),
        deep_embedding_dim=params.get('embedding_size', 8),
        wide_combine_schema_path=params.get('wide_combine_schema_path'),
        deep_combine_schema_path=params.get('combine_schema_path'),
        dnn_hidden_units=params.get('dnn_hidden_units', [512, 256, 64]),
        dnn_activations="relu",
        **_ftrl_params(params),
    )


@register("FFM", "ffm")
def build_ffm(params: dict):
    """FFM：Field-aware Factorization Machine"""
    return FFM(
        wide_embedding_dim=params.get('embedding_size', 8),
        deep_embedding_dim=params.get('embedding_size', 8),
        wide_combine_schema_path=params.get('wide_combine_schema_path'),
        deep_combine_schema_path=params.get('combine_schema_path'),
    )


@register("fwfm", "FwFM")
def build_fwfm(params: dict):
    """FwFM：Field-weighted Factorization Machine"""
    return FwFM(
        use_wide=params.get('use_wide', False),
        use_dnn=params.get('use_dnn', True),
        batch_norm=params.get('batch_norm', False),
        net_dropout=params.get('net_dropout', 0),
        wide_embedding_dim=params.get('embedding_size', 8),
        deep_embedding_dim=params.get('embedding_size', 8),
        wide_combine_schema_path=params.get('wide_combine_schema_path'),
        deep_combine_schema_path=params.get('combine_schema_path'),
        dnn_hidden_units=params.get('dnn_hidden_units', [512, 256, 64]),
        dnn_hidden_activations="relu",
        **_ftrl_params(params),
    )


@register("masknet", "MaskNet")
def build_masknet(params: dict):
    """MaskNet：特征掩码网络"""
    return MaskNet(
        use_wide=params.get('use_wide', False),
        batch_norm=params.get('batch_norm', False),
        net_dropout=params.get('net_dropout', 0),
        wide_embedding_dim=params.get('embedding_size', 8),
        deep_embedding_dim=params.get('embedding_size', 8),
        wide_combine_schema_path=params.get('wide_combine_schema_path'),
        deep_combine_schema_path=params.get('combine_schema_path'),
        dnn_hidden_units=params.get('dnn_hidden_units', [512, 256, 64]),
        dnn_hidden_activations="relu",
        **_ftrl_params(params),
    )
