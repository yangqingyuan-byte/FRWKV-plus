#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Optional


PYTHON_TRAIN_RE = re.compile(r"\bpython(?:\s+-m\s+adaptive_phasegate_kbs\.train|\s+train\.py)\b")
BATCH_SIZE_RE = re.compile(r"--batch_size\s+(\d+)")


def parse_batch_to_gpus(spec: str) -> dict[int, int]:
    mapping = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        batch_size, gpu_count = item.split(":", 1)
        mapping[int(batch_size)] = int(gpu_count)
    return mapping


def detect_batch_size(cmd: str) -> Optional[int]:
    match = BATCH_SIZE_RE.search(cmd)
    if match is None:
        return None
    return int(match.group(1))


def ensure_flag(cmd: str, flag: str) -> str:
    return cmd if flag in cmd else f"{cmd} {flag}"


def rewrite_command(cmd: str, gpus_required: int) -> str:
    if gpus_required <= 1:
        return ensure_flag(cmd, "--ddp_batch_size_mode global")

    torchrun_cmd = f"torchrun --standalone --nproc_per_node={gpus_required} -m adaptive_phasegate_kbs.train"
    if not PYTHON_TRAIN_RE.search(cmd):
        raise ValueError(f"Unsupported train command format: {cmd}")
    rewritten = PYTHON_TRAIN_RE.sub(torchrun_cmd, cmd, count=1)
    rewritten = ensure_flag(rewritten, "--ddp_batch_size_mode global")
    return rewritten


def main():
    parser = argparse.ArgumentParser(description="Rewrite single-GPU train jobs into DDP-ready jobs.")
    parser.add_argument("--input", required=True, help="Input jobs JSON")
    parser.add_argument("--output", required=True, help="Output jobs JSON")
    parser.add_argument(
        "--batch-to-gpus",
        default="16:1,32:2",
        help="Mapping like 16:1,32:2,64:4 meaning batch_size -> gpus_required",
    )
    parser.add_argument(
        "--default-gpus-per-job",
        type=int,
        default=1,
        help="Fallback gpus_required when batch_size not found in mapping",
    )
    parser.add_argument(
        "--append-flags",
        default="",
        help="Extra flags appended to every rewritten command",
    )
    args = parser.parse_args()

    mapping = parse_batch_to_gpus(args.batch_to_gpus)
    jobs = json.loads(Path(args.input).read_text(encoding="utf-8"))

    rewritten_jobs = []
    for job in jobs:
        job = dict(job)
        batch_size = detect_batch_size(job["cmd"])
        gpus_required = mapping.get(batch_size, args.default_gpus_per_job)
        job["cmd"] = rewrite_command(job["cmd"], gpus_required)
        if args.append_flags.strip():
            job["cmd"] = f"{job['cmd']} {args.append_flags.strip()}"
        job["gpus_required"] = gpus_required
        job.pop("gpu", None)
        rewritten_jobs.append(job)

    Path(args.output).write_text(
        json.dumps(rewritten_jobs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved {len(rewritten_jobs)} rewritten jobs to {args.output}")


if __name__ == "__main__":
    main()
