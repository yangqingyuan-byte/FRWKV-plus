#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import base64
import getpass
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from notify_wechat import WeChatNotifier

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
except ModuleNotFoundError:
    Fernet = None


ROOT = Path(__file__).resolve().parents[1]
RUNNER_DIR = ROOT / "experiment_runner"
RUNNER_DIR.mkdir(exist_ok=True)


def generate_key_from_password(password: bytes, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend(),
    )
    return base64.urlsafe_b64encode(kdf.derive(password))


def decrypt_token(config_file: str) -> str:
    if Fernet is None:
        raise RuntimeError("cryptography 未安装，无法读取加密 token")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"通知 token 配置文件不存在: {config_file}")
    data = Path(config_file).read_bytes()
    parts = data.split(b"\n", 1)
    if len(parts) != 2:
        raise ValueError("token 配置文件格式错误")
    salt, encrypted_token = parts
    import socket
    hostname = socket.gethostname()
    username = getpass.getuser()
    password = f"{hostname}_{username}_gpu_monitor_2026"
    key = generate_key_from_password(password.encode("utf-8"), salt)
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_token).decode("utf-8")


def build_notifier(token_config: Optional[str]):
    if not token_config:
        return None
    token = decrypt_token(token_config)
    return WeChatNotifier(method="serverchan", sendkey=token)


def send_notification(notifier, title: str, body: str):
    if notifier is None:
        print("[notify skipped] no token config provided")
        return
    ok, msg = notifier.send(title, body)
    print(f"[notify] {ok} {msg}")


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


def main():
    parser = argparse.ArgumentParser(description="批量实验执行器，结束后自动发微信通知")
    parser.add_argument("--jobs", required=True, help="JSON jobs file")
    parser.add_argument("--token-config", default="", help="加密微信 token 文件路径")
    parser.add_argument("--title", default="实验批处理完成", help="通知标题")
    args = parser.parse_args()

    jobs_path = Path(args.jobs)
    jobs = json.loads(jobs_path.read_text(encoding="utf-8"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNNER_DIR / f"run_{ts}"
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
    for job in jobs:
        print(f"[run] {job['name']}")
        result = run_job(job, logs_dir)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

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
