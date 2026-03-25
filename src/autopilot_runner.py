#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
autopilot_runner.py - RecAutoPilot 自动化实验运行器

基于 MetaSpore 的 DNN 训练流程，增加：
1. 业务类型过滤（shein, ae*, shopee*, lazada*）
2. 实验配置合并（base.yaml + experiment.yaml）
3. 实验结果记录
"""
import os
import sys
import json
import yaml
import shutil
import argparse
from datetime import datetime, date

# 添加 MetaSpore 路径
METASPORE_DIR = "/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python"
sys.path.insert(0, METASPORE_DIR)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from pyspark.sql import functions as F
from dnn_trainFlow import DNNModelTrainFlow
from movas_logger import MovasLogger


class AutoPilotTrainFlow(DNNModelTrainFlow):
    """
    RecAutoPilot 训练流程
    
    继承 DNNModelTrainFlow，重写数据读取逻辑以支持业务过滤
    """
    
    def __init__(self, config_path: str, base_config_path: str = None,
                 output_dir: str = None, exp_conf_path: str = None):
        # 合并配置
        self.merged_config_path = self._merge_configs(base_config_path, config_path,
                                                       output_dir=output_dir)
        super().__init__(self.merged_config_path)
        # 保存 exp_conf 路径，用于读取 readme
        self._exp_conf_path = exp_conf_path or config_path
        self._load_exp_readme()
        
        # 业务过滤配置
        self.business_type_filter = self.params.get('business_type_filter', 
            ['shein', 'ae', 'shopee', 'lazada'])
        
        # 实验跟踪
        self.exp_id = None
        self.exp_start_time = None
    
    def _merge_configs(self, base_path: str, exp_path: str,
                       output_dir: str = None) -> str:
        """合并基础配置和实验配置"""
        merged = {}
        
        # 加载基础配置
        if base_path and os.path.exists(base_path):
            with open(base_path, 'r') as f:
                merged.update(yaml.safe_load(f) or {})
        
        # 加载实验配置（覆盖基础配置）
        with open(exp_path, 'r') as f:
            merged.update(yaml.safe_load(f) or {})
        
        # --output_dir 优先级最高，覆盖 model_out_base_path
        if output_dir:
            merged['model_out_base_path'] = output_dir
        
        # 写入临时合并文件（按实验配置文件名区分，避免并行时互相覆盖）
        exp_name = os.path.splitext(os.path.basename(exp_path))[0]
        merged_path = f'./conf/_merged_{exp_name}.yaml'
        os.makedirs(os.path.dirname(merged_path), exist_ok=True)
        with open(merged_path, 'w') as f:
            yaml.dump(merged, f, default_flow_style=False)
        
        return merged_path

    def _load_exp_readme(self):
        """
        从实验 conf 目录下的 readme 文件加载实验描述，
        覆盖 base_trainFlow._load_config 中读取 ./readme 的逻辑。
        优先级：
          1. <exp_conf_dir>/<exp_name>/readme  (yaml 同名子目录，如 exp_fea_A/readme)
          2. <exp_conf_dir>/readme             (yaml 所在目录)
          3. ./readme                          (根目录兜底)
        """
        readme_content = ''
        candidates = []
        if self._exp_conf_path:
            abs_path = os.path.abspath(self._exp_conf_path)
            exp_dir = os.path.dirname(abs_path)
            # 1. yaml 同名子目录：去掉 .yaml 后缀得到子目录名
            exp_name = os.path.splitext(os.path.basename(abs_path))[0]
            candidates.append(os.path.join(exp_dir, exp_name, 'readme'))
            # 2. yaml 所在目录
            candidates.append(os.path.join(exp_dir, 'readme'))
        # 3. 根目录兜底
        candidates.append('./readme')

        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        readme_content = f.read().strip()
                    break
                except Exception:
                    pass

        if readme_content:
            self.exp_readme = '实验信息:\n' + readme_content
    
    def _read_dataset_by_date(self, base_path: str, date_str: str):
        """
        重写数据读取，增加业务类型过滤
        """
        # 支持两种目录格式：part=YYYY-MM-DD 或 YYYY-MM-DD
        data_path = os.path.join(base_path, f"part={date_str}")
        if not os.path.exists(data_path):
            data_path = os.path.join(base_path, date_str)
        df = self.spark_session.read.parquet(data_path)
        
        # 选择需要的特征
        df = df.select(*self.used_fea_list)
        MovasLogger.log(f'[{date_str}] 原始样本数: {df.count()}')
        
        # 业务类型过滤
        df = self._filter_business_type(df)
        MovasLogger.log(f'[{date_str}] 业务过滤后样本数: {df.count()}')
        
        # 类型转换
        for col_name in df.columns:
            if col_name == 'label':
                df = df.withColumn(col_name, F.col(col_name).cast("float"))
            else:
                df = df.withColumn(col_name, F.col(col_name).cast("string"))
        
        df = df.withColumn("domain_id", F.lit(0))
        
        # 采样
        df = self.random_sample(df)
        df = df.fillna('none')
        MovasLogger.log(f'[{date_str}] 采样后样本数: {df.count()}')
        
        return df
    
    def _filter_business_type(self, df):
        """
        过滤业务类型：shein, ae*, shopee*, lazada*
        """
        if 'business_type' not in df.columns:
            MovasLogger.log('警告: business_type 列不存在，跳过业务过滤')
            return df
        
        # 构建过滤条件
        conditions = []
        for bt in self.business_type_filter:
            if bt in ['shein']:
                # 精确匹配
                conditions.append(F.col("business_type") == bt)
            else:
                # 前缀匹配
                conditions.append(F.col("business_type").startswith(bt))
        
        # 合并条件
        if conditions:
            filter_condition = conditions[0]
            for cond in conditions[1:]:
                filter_condition = filter_condition | cond
            df = df.filter(filter_condition)
        
        return df
    
    def random_sample(self, df):
        """
        采样策略：
        - 正样本全保留
        - 负样本：shein 采样 1%，其他采样 10%
        """
        return df.filter(
            (F.col("label") == 1) |
            (
                (F.col("label") == 0) &
                F.when(
                    (F.col("business_type") == 'shein'),
                    F.rand(seed=42) < 0.01
                ).otherwise(
                    F.rand(seed=42) < 0.1
                )
            )
        )
    
    def set_experiment_id(self, exp_id: str):
        """设置实验 ID"""
        self.exp_id = exp_id
        self.exp_start_time = datetime.now()
    
    def get_experiment_summary(self) -> dict:
        """获取实验摘要"""
        return {
            'exp_id': self.exp_id,
            'start_time': self.exp_start_time.isoformat() if self.exp_start_time else None,
            'model_type': self.params.get('model_type'),
            'embedding_size': self.params.get('embedding_size'),
            'dnn_hidden_units': self.params.get('dnn_hidden_units'),
            'batch_size': self.params.get('batch_size'),
            'train_dates': f"{self.params.get('train_start_date')} ~ {self.params.get('train_end_date')}",
            'validation_date': str(self.params.get('validation_date')),
        }


def generate_exp_id() -> str:
    """生成实验 ID"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"exp_{timestamp}"


class DateTimeEncoder(json.JSONEncoder):
    """JSON 编码器，支持 datetime 和 date 类型"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def save_experiment_config(exp_dir: str, config: dict, hypothesis: str = None):
    """保存实验配置"""
    os.makedirs(exp_dir, exist_ok=True)
    
    config_data = {
        'exp_id': os.path.basename(exp_dir),
        'timestamp': datetime.now().isoformat(),
        'hypothesis': hypothesis,
        'config': config
    }
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)


def update_leaderboard(leaderboard_path: str, exp_id: str, metrics: dict, config_summary: str):
    """更新排行榜"""
    leaderboard = {'experiments': []}
    
    if os.path.exists(leaderboard_path):
        with open(leaderboard_path, 'r') as f:
            leaderboard = json.load(f)
    
    # 添加新实验
    exp_entry = {
        'exp_id': exp_id,
        'timestamp': datetime.now().isoformat(),
        'overall_auc': metrics.get('overall', {}).get('auc', 0),
        'overall_pcoc': metrics.get('overall', {}).get('pcoc', 0),
        'config_summary': config_summary
    }
    leaderboard['experiments'].append(exp_entry)
    
    # 按 AUC 排序
    leaderboard['experiments'].sort(key=lambda x: x.get('overall_auc', 0), reverse=True)
    
    # 更新最佳记录
    if leaderboard['experiments']:
        best = leaderboard['experiments'][0]
        leaderboard['best_overall_auc'] = {
            'exp_id': best['exp_id'],
            'auc': best['overall_auc'],
            'config_summary': best['config_summary']
        }
    
    with open(leaderboard_path, 'w') as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='RecAutoPilot 实验运行器')
    parser.add_argument('--base_conf', type=str, default='./conf/base.yaml',
                        help='基础配置文件路径')
    parser.add_argument('--exp_conf', type=str, default='./conf/experiment.yaml',
                        help='实验配置文件路径')
    parser.add_argument('--name', type=str, default='autopilot',
                        help='实验名称')
    parser.add_argument('--eval_keys', type=str, default='business_type',
                        help='评估分组键')
    parser.add_argument('--hypothesis', type=str, default=None,
                        help='实验假设')
    parser.add_argument('--exp_id', type=str, default=None,
                        help='实验 ID（不指定则自动生成）')
    parser.add_argument('--validation', type=bool, default=False,
                        help='仅验证模式')
    parser.add_argument('--model_date', type=str, default=None,
                        help='验证模式下的模型日期')
    parser.add_argument('--sample_date', type=str, default=None,
                        help='验证模式下的样本日期')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='模型输出根目录（覆盖 base.yaml 中的 model_out_base_path，用于并行实验隔离）')
    
    args = parser.parse_args()
    
    # 生成实验 ID
    exp_id = args.exp_id or generate_exp_id()
    exp_dir = f'./experiments/{exp_id}'
    
    print(f"=" * 60)
    print(f"RecAutoPilot 实验: {exp_id}")
    print(f"=" * 60)
    
    # 创建训练流程
    trainer = AutoPilotTrainFlow(args.exp_conf, args.base_conf,
                                 output_dir=args.output_dir,
                                 exp_conf_path=args.exp_conf)
    trainer.set_experiment_id(exp_id)
    
    # 保存实验配置
    save_experiment_config(exp_dir, trainer.params, args.hypothesis)
    
    # 复制当前特征配置
    combine_schema_path = trainer.params.get('combine_schema_path', './conf/combine_schema')
    if os.path.exists(combine_schema_path):
        shutil.copy(combine_schema_path, os.path.join(exp_dir, 'combine_schema'))
    
    # 运行训练
    class Args:
        pass
    
    run_args = Args()
    run_args.conf = trainer.merged_config_path
    # 优先用 --name 参数，否则从 yaml 的 experiment_name 字段读取，最后兜底 'autopilot'
    run_args.name = args.name if args.name != 'autopilot' else trainer.params.get('experiment_name', 'autopilot')
    run_args.eval_keys = args.eval_keys
    run_args.validation = args.validation
    run_args.model_date = args.model_date
    run_args.sample_date = args.sample_date
    
    trainer.run_complete_flow(run_args)
    
    # 保存实验摘要
    summary = trainer.get_experiment_summary()
    with open(os.path.join(exp_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n实验完成: {exp_id}")
    print(f"结果保存在: {exp_dir}")


if __name__ == '__main__':
    main()
