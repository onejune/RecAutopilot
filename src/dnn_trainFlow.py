#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
dnn_trainFlow.py - DNN 模型训练流程

继承 BaseTrainFlow，实现 _build_model_module。
模型构建通过 model_registry 注册表完成，新增模型只需在 src/models/ 下注册，
无需修改本文件。

支持的模型见 src/models/ 目录，或运行:
    from model_registry import list_models; print(list_models())
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
print('sys.path:' + str(sys.path))

import metaspore as ms
print("metaspore.__file__ =", ms.__file__)

from base_trainFlow import BaseTrainFlow
from movas_logger import MovasLogger
# 导入注册表（自动触发 src/models/ 下所有模型的注册）
from model_registry import build_model, list_models


class DNNModelTrainFlow(BaseTrainFlow):
    """
    DNN 模型训练流程

    继承 BaseTrainFlow，通过 model_registry 注册表构建模型。
    新增模型无需修改本文件，只需在 src/models/ 下添加注册。

    当前支持的模型类型见 src/models/ 目录。
    """

    def _build_model_module(self):
        """
        通过注册表构建模型实例。
        model_type 在 yaml 配置中指定，不区分大小写。
        """
        configed_model = self.params.get('model_type', "WideDeep")
        MovasLogger.add_log(content=f"Building model module: {configed_model}")
        MovasLogger.add_log(content=f"Available models: {list_models()}")

        self.model_module = build_model(configed_model, self.params)
        self.configed_model = configed_model


if __name__ == "__main__":
    args = DNNModelTrainFlow.parse_args()
    print(f'DNNModelTrainFlow: debug_args={args}')
    trainer = DNNModelTrainFlow(config_path=args.conf)
    trainer.run_complete_flow(args)
    MovasLogger.save_to_local()
