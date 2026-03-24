#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_runner.py - 批量实验调度器

支持：
- 串行/并行实验组（同 group 内串行，不同 group 并行）
- 自动生成日志文件
- 噪音日志过滤
- 实验失败后继续运行其他组

用法:
    python scripts/batch_runner.py --plan conf/plans/phase2_feature.yaml
"""
import os
import sys
import yaml
import argparse
import subprocess
import time
import threading
from collections import defaultdict

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_ENV = "/root/anaconda3/envs/spore/bin/python"
METASPORE_DIR = "/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python"

# 过滤掉的噪音日志关键词
NOISE_PATTERNS = ['bkdr_hash', 'add expr', 'StringBKDR']


def run_experiment(exp: dict, log_dir: str) -> int:
    """
    运行单个实验，实时输出日志到终端和文件。

    Returns:
        进程退出码（0 表示成功）
    """
    name = exp['name']
    exp_conf = exp.get('exp_conf', './conf/experiment.yaml')
    hypothesis = exp.get('hypothesis', '')
    output_dir = exp.get('output_dir', f'./output/{name}')
    eval_keys = exp.get('eval_keys', 'business_type')

    log_file = os.path.join(log_dir, f"{name}.log")

    cmd = [
        PYTHON_ENV, 'src/autopilot_runner.py',
        '--base_conf', './conf/base.yaml',
        '--exp_conf', exp_conf,
        '--name', name,
        '--eval_keys', eval_keys,
        '--output_dir', output_dir,
    ]
    if hypothesis:
        cmd += ['--hypothesis', hypothesis]

    print(f"\n[{name}] ▶ 启动实验，日志: {log_file}")
    print(f"[{name}]   exp_conf: {exp_conf}")
    print(f"[{name}]   假设: {hypothesis or '(无)'}")

    with open(log_file, 'w', buffering=1) as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=PROJECT_DIR,
            env={**os.environ,
                 'PYTHONPATH': f"{METASPORE_DIR}:{PROJECT_DIR}/src:{os.environ.get('PYTHONPATH', '')}",
                 'PYSPARK_PYTHON': PYTHON_ENV,
                 'PYSPARK_DRIVER_PYTHON': PYTHON_ENV,
                 'PYTHONUNBUFFERED': '1'},
        )
        for line in proc.stdout:
            # 过滤噪音日志
            if not any(pat in line for pat in NOISE_PATTERNS):
                sys.stdout.write(f"[{name}] {line}")
                sys.stdout.flush()
                f.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"\n[{name}] ❌ 实验失败 (exit={proc.returncode})")
    else:
        print(f"\n[{name}] ✅ 实验完成")

    return proc.returncode


def run_group(group_id: int, experiments: list, log_dir: str):
    """串行运行一组实验，任一失败则停止本组后续"""
    print(f"\n{'='*50}")
    print(f"[Group {group_id}] 开始，共 {len(experiments)} 个实验")
    print(f"{'='*50}")
    for exp in experiments:
        rc = run_experiment(exp, log_dir)
        if rc != 0:
            print(f"[Group {group_id}] 实验 '{exp['name']}' 失败，跳过本组剩余实验")
            break
    print(f"\n[Group {group_id}] 完成")


def main():
    parser = argparse.ArgumentParser(description='RecAutoPilot 批量实验调度器')
    parser.add_argument('--plan', required=True, help='实验计划文件路径（YAML）')
    args = parser.parse_args()

    # 加载实验计划
    plan_path = args.plan
    if not os.path.isabs(plan_path):
        plan_path = os.path.join(PROJECT_DIR, plan_path)

    with open(plan_path) as f:
        plan = yaml.safe_load(f)

    gap_seconds = plan.get('gap_seconds', 30)
    experiments = plan.get('experiments', [])

    if not experiments:
        print("计划文件中没有实验，退出")
        return

    # 按 group 分组（默认 group=1）
    groups = defaultdict(list)
    for exp in experiments:
        group_id = exp.get('group', 1)
        groups[group_id].append(exp)

    log_dir = os.path.join(PROJECT_DIR, 'log')
    os.makedirs(log_dir, exist_ok=True)

    # 确保中断控制文件存在
    interrupt_flag = os.path.join(PROJECT_DIR, 'train_interrupt.flag')
    if not os.path.exists(interrupt_flag):
        open(interrupt_flag, 'w').close()

    print(f"\n{'='*60}")
    print(f"RecAutoPilot 批量实验启动")
    print(f"计划文件: {plan_path}")
    print(f"共 {len(groups)} 组，{len(experiments)} 个实验")
    print(f"并行组间隔: {gap_seconds}s")
    print(f"{'='*60}")

    # 启动各组线程（不同组并行，同组内串行）
    threads = []
    for i, (gid, exps) in enumerate(sorted(groups.items())):
        t = threading.Thread(
            target=run_group,
            args=(gid, exps, log_dir),
            name=f"Group-{gid}",
            daemon=False,
        )
        t.start()
        threads.append(t)
        if i < len(groups) - 1:
            print(f"\n[Scheduler] Group {gid} 已启动，{gap_seconds}s 后启动下一组...")
            time.sleep(gap_seconds)

    # 等待所有组完成
    for t in threads:
        t.join()

    print(f"\n{'='*60}")
    print(f"所有实验完成！时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
