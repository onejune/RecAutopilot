#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_registry.py - 模型注册表

统一管理所有支持的模型类型。
新增模型只需在 src/models/ 下注册，不需要修改 dnn_trainFlow.py。

用法:
    # 注册模型（在 src/models/*.py 中使用装饰器）
    @register("MyModel")
    def build_mymodel(params):
        return MyModel(...)

    # 构建模型
    from model_registry import build_model
    model = build_model("WideDeep", params)

    # 查看所有可用模型
    from model_registry import list_models
    print(list_models())
"""
from typing import Dict, Callable, Any

# 全局注册表：模型名（小写）→ 构建函数
_REGISTRY: Dict[str, Callable] = {}


def register(*names: str):
    """
    装饰器：注册模型构建函数。
    支持多个别名，名称不区分大小写。

    示例:
        @register("WideDeep", "widedeep")
        def build_widedeep(params: dict):
            return WideDeep(...)
    """
    def decorator(fn: Callable) -> Callable:
        for name in names:
            _REGISTRY[name.lower()] = fn
        return fn
    return decorator


def build_model(name: str, params: dict) -> Any:
    """
    根据模型名称构建模型实例。

    Args:
        name: 模型名称（不区分大小写）
        params: 完整配置参数字典

    Returns:
        模型实例

    Raises:
        ValueError: 模型名称未注册
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = sorted(set(_REGISTRY.keys()))
        raise ValueError(
            f"未知模型类型: '{name}'。\n"
            f"可用模型: {available}\n"
            f"新增模型请在 src/models/ 下注册。"
        )
    return _REGISTRY[key](params)


def list_models() -> list:
    """返回所有已注册的模型名称（去重，排序）"""
    # 去掉重复别名，只返回唯一的构建函数对应的名称
    seen_fns = set()
    result = []
    for name, fn in sorted(_REGISTRY.items()):
        if fn not in seen_fns:
            seen_fns.add(fn)
            result.append(name)
    return result


def is_registered(name: str) -> bool:
    """检查模型名称是否已注册"""
    return name.lower() in _REGISTRY


# ============================================================
# 自动加载 src/models/ 下所有模块（触发 @register 装饰器）
# ============================================================
import os
import importlib

def _auto_load_models():
    """自动导入 src/models/ 下所有 .py 文件，触发模型注册"""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.isdir(models_dir):
        return
    for fname in sorted(os.listdir(models_dir)):
        if fname.endswith('.py') and not fname.startswith('_'):
            module_name = fname[:-3]
            try:
                importlib.import_module(f'models.{module_name}')
            except Exception as e:
                import warnings
                warnings.warn(f"[model_registry] 加载 models/{fname} 失败: {e}")

_auto_load_models()
