#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

from experiment_runner import build_notifier, send_notification


ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = ROOT / "experiment_runner"
RUNNER_DIR.mkdir(exist_ok=True)


def run_job(job: dict, logs_dir: Path) -> dict:
    name = job["name"]
    cwd = job["cwd"]
    env = dict(os.environ)
    if "gpu" in job and job["gpu"] is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    cmd = job["cmd"]
    log_path = logs_dir / f"{name}.log"
    start = time.time()
    proc = subprocess.run(
        ["/bin/bash", "-lc", cmd],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
    )
    elapsed = time.time() - start
    log_path.write_text(proc.stdout, encoding="utf-8")
    return {
        "name": name,
        "cwd": cwd,
        "cmd": cmd,
        "gpu": job.get("gpu"),
        "returncode": proc.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "log_path": str(log_path),
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def parse_gpu_spec(gpu_value):
    if gpu_value is None or gpu_value == "":
        return tuple()
    if isinstance(gpu_value, int):
        return (gpu_value,)
    if isinstance(gpu_value, str):
        return tuple(int(x.strip()) for x in gpu_value.split(",") if x.strip())
    raise TypeError(f"Unsupported gpu spec: {gpu_value!r}")


def allocate_gpus(gpu_pool, busy_gpus, count):
    free_gpus = [gpu for gpu in gpu_pool if gpu not in busy_gpus]
    if len(free_gpus) < count:
        return None
    return tuple(free_gpus[:count])


def get_job_gpu_request(job, gpu_pool, busy_gpus):
    explicit = parse_gpu_spec(job.get("gpu"))
    if explicit:
        if any(gpu in busy_gpus for gpu in explicit):
            return None
        return explicit

    gpus_required = int(job.get("gpus_required", 0) or 0)
    if gpus_required <= 0:
        return tuple()
    return allocate_gpus(gpu_pool, busy_gpus, gpus_required)


def main():
    parser = argparse.ArgumentParser(description="并行批量实验执行器，结束后自动发微信通知")
    parser.add_argument("--jobs", required=True, help="JSON jobs file")
    parser.add_argument("--token-config", default="", help="加密微信 token 文件路径")
    parser.add_argument("--title", default="并行实验批处理完成", help="通知标题")
    parser.add_argument("--max-workers", type=int, default=8, help="最大并发数")
    parser.add_argument("--gpu-pool", default="0,1,2,3,4,5,6,7", help="可调度 GPU 池，如 0,1,2,3")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="调度轮询间隔（秒）")
    args = parser.parse_args()

    jobs_path = Path(args.jobs)
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))
    gpu_pool = [int(x.strip()) for x in args.gpu_pool.split(",") if x.strip()]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNNER_DIR / f"parallel_run_{ts}"
    logs_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    notifier = None
    notify_error = None
    if args.token_config:
        try:
            notifier = build_notifier(args.token_config)
        except Exception as e:
            notify_error = str(e)

    results = []
    pending_jobs = list(jobs)
    running = {}

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        while pending_jobs or running:
            launched = False
            busy_gpus = set()
            for req in running.values():
                busy_gpus.update(req)

            idx = 0
            while idx < len(pending_jobs) and len(running) < args.max_workers:
                job = pending_jobs[idx]
                gpu_request = get_job_gpu_request(job, gpu_pool, busy_gpus)
                if gpu_request is None:
                    idx += 1
                    continue

                scheduled_job = dict(job)
                if gpu_request:
                    scheduled_job["gpu"] = ",".join(str(g) for g in gpu_request)
                future = ex.submit(run_job, scheduled_job, logs_dir)
                running[future] = gpu_request
                busy_gpus.update(gpu_request)
                pending_jobs.pop(idx)
                launched = True

            if not running:
                if pending_jobs and not launched:
                    raise RuntimeError(
                        "No runnable jobs found. Check gpu/gpus_required settings against --gpu-pool."
                    )
                continue

            done, _ = wait(running.keys(), timeout=args.poll_interval, return_when=FIRST_COMPLETED)
            if not done:
                continue

            for fut in done:
                result = fut.result()
                results.append(result)
                print(json.dumps(result, ensure_ascii=False))
                running.pop(fut, None)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    success = sum(1 for r in results if r["returncode"] == 0)
    failed = len(results) - success
    body = [
        f"任务数: {len(results)}",
        f"成功: {success}",
        f"失败: {failed}",
        f"summary: {summary_path}",
        f"logs: {logs_dir}",
    ]
    if notify_error:
        body.append(f"通知初始化失败: {notify_error}")
    body = "\n".join(body)
    send_notification(notifier, args.title, body)
    print(body)


if __name__ == "__main__":
    main()
