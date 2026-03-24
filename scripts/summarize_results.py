#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
summarize_results.py - 实验结果汇总工具

从 leaderboard.json 和 experiments/ 目录读取实验结果，
生成对比表格。

用法:
    python scripts/summarize_results.py                    # 汇总所有实验
    python scripts/summarize_results.py --top 10           # 只看 top10
    python scripts/summarize_results.py --filter fea       # 按名称过滤
    python scripts/summarize_results.py --baseline deeper_dnn  # 标注 baseline
"""
import os
import sys
import json
import argparse
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_leaderboard(path: str) -> list:
    """加载排行榜数据"""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        lb = json.load(f)
    return lb.get('experiments', [])


def load_exp_details(exp_dir: str, exp_id: str) -> dict:
    """加载单个实验的详细信息"""
    config_file = os.path.join(exp_dir, exp_id, 'config.json')
    if not os.path.exists(config_file):
        return {}
    with open(config_file) as f:
        return json.load(f)


def format_delta(auc: float, baseline_auc: float) -> str:
    """格式化 AUC 与 baseline 的差值"""
    if baseline_auc == 0:
        return ''
    delta = auc - baseline_auc
    sign = '+' if delta >= 0 else ''
    return f"{sign}{delta:.4f}"


def main():
    parser = argparse.ArgumentParser(description='实验结果汇总')
    parser.add_argument('--top', type=int, default=30, help='显示 top N 实验')
    parser.add_argument('--filter', type=str, default=None, help='按实验名过滤（子字符串匹配）')
    parser.add_argument('--baseline', type=str, default='deeper_dnn', help='baseline 实验名')
    parser.add_argument('--sort', type=str, default='auc',
                        choices=['auc', 'pcoc', 'time'], help='排序方式')
    args = parser.parse_args()

    lb_path = os.path.join(PROJECT_DIR, 'leaderboard.json')
    exp_dir = os.path.join(PROJECT_DIR, 'experiments')

    experiments = load_leaderboard(lb_path)

    if not experiments:
        print("暂无实验结果（leaderboard.json 为空或不存在）")
        return

    # 过滤
    if args.filter:
        experiments = [e for e in experiments
                       if args.filter.lower() in e.get('exp_id', '').lower()
                       or args.filter.lower() in e.get('config_summary', '').lower()]

    # 排序
    if args.sort == 'auc':
        experiments = sorted(experiments, key=lambda x: x.get('overall_auc', 0), reverse=True)
    elif args.sort == 'time':
        experiments = sorted(experiments, key=lambda x: x.get('timestamp', ''), reverse=True)

    experiments = experiments[:args.top]

    # 找 baseline AUC
    baseline_auc = 0.0
    all_exps = load_leaderboard(lb_path)
    for e in all_exps:
        if args.baseline in e.get('exp_id', '') or args.baseline in e.get('config_summary', ''):
            baseline_auc = e.get('overall_auc', 0)
            break

    # 打印表格
    col_w = [5, 32, 8, 8, 8, 35]
    header = f"{'Rank':<{col_w[0]}} {'Exp ID':<{col_w[1]}} {'AUC':<{col_w[2]}} {'ΔAUC':<{col_w[3]}} {'PCOC':<{col_w[4]}} {'Config'}"
    sep = '-' * (sum(col_w) + 20)

    print(f"\n{'='*80}")
    print(f"实验结果汇总  (baseline={args.baseline}, AUC={baseline_auc:.4f})")
    print(f"{'='*80}")
    print(header)
    print(sep)

    for i, exp in enumerate(experiments, 1):
        exp_id = exp.get('exp_id', '')
        auc = exp.get('overall_auc', 0)
        pcoc = exp.get('overall_pcoc', 0)
        config = exp.get('config_summary', '')
        delta = format_delta(auc, baseline_auc)

        # 标注 baseline
        marker = ' ◀ baseline' if args.baseline in exp_id else ''

        print(f"{i:<{col_w[0]}} {exp_id:<{col_w[1]}} {auc:<{col_w[2]}.4f} {delta:<{col_w[3]}} {pcoc:<{col_w[4]}.4f} {config}{marker}")

    print(sep)
    print(f"共 {len(experiments)} 条记录\n")


if __name__ == '__main__':
    main()
