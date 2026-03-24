#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
batch_runner.py - 批量实验调度器

支持：
- 串行/并行实验组（同 group 内串行，不同 group 并行）
- 失败自动重试（最多 MAX_RETRY 次）
- 错误检测：区分 Spark 资源失败（可重试）和代码错误（不重试）
- 自动生成日志文件

用法:
    python scripts/batch_runner.py --plan conf/plans/phase3_architecture.yaml
"""
import os
import sys
import yaml
import argparse
import subprocess
import time
import threading
from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON_ENV = "/root/anaconda3/envs/spore/bin/python"
METASPORE_DIR = "/mnt/workspace/git_project/movas_hub/DeepForgeX/MetaSpore/python"

# 过滤掉的噪音日志关键词
NOISE_PATTERNS = ['bkdr_hash', 'add expr', 'StringBKDR']

# 可重试的错误关键词（Spark 资源/网络类瞬时故障）
RETRYABLE_ERRORS = [
    'DAGSchedulerEventProcessLoop',
    'Job aborted due to stage failure',
    'Could not recover from a failed barrier',
    'SparkException',
    'Connection refused',
    'Lost executor',
    'ExecutorLostFailure',
    'TaskSetManager',
    'WARN BarrierTaskContext',
]

# 不可重试的错误关键词（代码/配置错误，重试无意义）
FATAL_ERRORS = [
    'ModuleNotFoundError',
    'ImportError',
    'AttributeError',
    'KeyError',
    'FileNotFoundError',
    'yaml.YAMLError',
    'SyntaxError',
]

MAX_RETRY = 3          # 最大重试次数
RETRY_WAIT = 60        # 重试前等待秒数（让 Spark 资源释放）


def classify_failure(log_file: str) -> str:
    """
    分析日志，判断失败类型。
    返回: 'retryable' | 'fatal' | 'unknown'
    """
    if not os.path.exists(log_file):
        return 'unknown'
    try:
        # 只读最后 200 行，避免读大文件
        with open(log_file, 'r') as f:
            lines = f.readlines()
        tail = ''.join(lines[-200:])

        for kw in FATAL_ERRORS:
            if kw in tail:
                return 'fatal'
        for kw in RETRYABLE_ERRORS:
            if kw in tail:
                return 'retryable'
    except Exception:
        pass
    return 'unknown'


def run_experiment(exp: dict, log_dir: str) -> bool:
    """
    运行单个实验，失败时自动重试（可重试错误最多 MAX_RETRY 次）。
    返回: True 表示最终成功，False 表示最终失败
    """
    name = exp['name']
    exp_conf = exp.get('exp_conf', './conf/experiment.yaml')
    hypothesis = exp.get('hypothesis', '')
    output_dir = exp.get('output_dir', f'./output/{name}')
    eval_keys = exp.get('eval_keys', 'business_type')

    base_log = os.path.join(log_dir, f"{name}.log")

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

    env = {
        **os.environ,
        'PYTHONPATH': f"{METASPORE_DIR}:{PROJECT_DIR}/src:{os.environ.get('PYTHONPATH', '')}",
        'PYSPARK_PYTHON': PYTHON_ENV,
        'PYSPARK_DRIVER_PYTHON': PYTHON_ENV,
        'PYTHONUNBUFFERED': '1',
    }

    for attempt in range(1, MAX_RETRY + 1):
        # 每次重试写独立日志（方便排查），最终合并到 base_log
        log_file = base_log if attempt == 1 else f"{base_log}.retry{attempt}"
        ts = time.strftime('%Y-%m-%d %H:%M:%S')

        if attempt == 1:
            print(f"\n[{name}] ▶ 启动实验 ({ts})，日志: {log_file}")
        else:
            print(f"\n[{name}] 🔄 第 {attempt} 次重试 ({ts})，日志: {log_file}")

        with open(log_file, 'w', buffering=1) as f:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=PROJECT_DIR,
                env=env,
            )
            for line in proc.stdout:
                if not any(pat in line for pat in NOISE_PATTERNS):
                    sys.stdout.write(f"[{name}] {line}")
                    sys.stdout.flush()
                    f.write(line)
            proc.wait()

        if proc.returncode == 0:
            print(f"\n[{name}] ✅ 实验完成 (attempt={attempt})")
            return True

        # 失败：分析原因
        failure_type = classify_failure(log_file)
        print(f"\n[{name}] ❌ 失败 (exit={proc.returncode}, type={failure_type}, attempt={attempt}/{MAX_RETRY})")

        if failure_type == 'fatal':
            print(f"[{name}] 代码/配置错误，不重试")
            break

        if attempt < MAX_RETRY:
            print(f"[{name}] 等待 {RETRY_WAIT}s 后重试（让 Spark 资源释放）...")
            time.sleep(RETRY_WAIT)
        else:
            print(f"[{name}] 已达最大重试次数 {MAX_RETRY}，放弃")

    return False


def run_group(group_id: int, experiments: list, log_dir: str, results: dict):
    """串行运行一组实验，记录每个实验的成功/失败"""
    print(f"\n{'='*50}")
    print(f"[Group {group_id}] 开始，共 {len(experiments)} 个实验")
    print(f"{'='*50}")

    for exp in experiments:
        ok = run_experiment(exp, log_dir)
        results[exp['name']] = 'success' if ok else 'failed'
        if not ok:
            print(f"[Group {group_id}] '{exp['name']}' 最终失败，跳过本组剩余实验")
            # 标记后续实验为 skipped
            idx = experiments.index(exp)
            for skipped in experiments[idx + 1:]:
                results[skipped['name']] = 'skipped'
            break

    print(f"\n[Group {group_id}] 完成")


def main():
    parser = argparse.ArgumentParser(description='RecAutoPilot 批量实验调度器')
    parser.add_argument('--plan', required=True, help='实验计划文件路径（YAML）')
    args = parser.parse_args()

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

    groups = defaultdict(list)
    for exp in experiments:
        groups[exp.get('group', 1)].append(exp)

    log_dir = os.path.join(PROJECT_DIR, 'log')
    os.makedirs(log_dir, exist_ok=True)

    # 确保中断控制文件存在
    interrupt_flag = os.path.join(PROJECT_DIR, 'train_interrupt.flag')
    if not os.path.exists(interrupt_flag):
        open(interrupt_flag, 'w').close()

    print(f"\n{'='*60}")
    print(f"RecAutoPilot 批量实验启动")
    print(f"计划文件: {plan_path}")
    print(f"共 {len(groups)} 组，{len(experiments)} 个实验，最大重试 {MAX_RETRY} 次")
    print(f"并行组间隔: {gap_seconds}s")
    print(f"{'='*60}")

    results = {}  # name -> 'success' | 'failed' | 'skipped'
    threads = []

    for i, (gid, exps) in enumerate(sorted(groups.items())):
        t = threading.Thread(
            target=run_group,
            args=(gid, exps, log_dir, results),
            name=f"Group-{gid}",
            daemon=False,
        )
        t.start()
        threads.append(t)
        if i < len(groups) - 1:
            print(f"\n[Scheduler] Group {gid} 已启动，{gap_seconds}s 后启动下一组...")
            time.sleep(gap_seconds)

    for t in threads:
        t.join()

    # 汇总结果
    print(f"\n{'='*60}")
    print(f"所有实验完成！时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    success = [n for n, s in results.items() if s == 'success']
    failed  = [n for n, s in results.items() if s == 'failed']
    skipped = [n for n, s in results.items() if s == 'skipped']
    print(f"✅ 成功: {len(success)} — {success}")
    if failed:
        print(f"❌ 失败: {len(failed)} — {failed}")
    if skipped:
        print(f"⏭  跳过: {len(skipped)} — {skipped}")


if __name__ == '__main__':
    main()
