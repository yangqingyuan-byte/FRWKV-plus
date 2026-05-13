#!/usr/bin/env python3
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT
OUT_PATH = ROOT / "jobs" / "experiment_runner_kbs_tuning_confirm_jobs_gpu47.json"
GPU_POOL = [4, 5, 6, 7]


def build_cmd(
    *,
    model_type: str,
    data_path: str,
    num_nodes: int,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    dropout_n: float,
    d_model: int,
    d_ff: int,
    n_heads: int,
    e_layers: int,
    temp_patch_len: int,
    temp_stride: int,
    embed_size: int,
    phase_period_len: int,
    phase_num_routers: int,
    phase_alpha_init: float,
    phase_trust_bias_init: float,
    epochs: int,
    es_patience: int,
    loss_mode: str,
    lossfun_alpha: float,
    weight_decay: float,
    seed: int,
    model_tag: str,
    num_workers: int = 4,
):
    python_cmd = [
        "python -m adaptive_phasegate_kbs.train",
        f"--model_type {model_type}",
        "--use_embeddings false",
        f"--data_path {data_path}",
        f"--num_nodes {num_nodes}",
        f"--seq_len {seq_len}",
        f"--pred_len {pred_len}",
        f"--batch_size {batch_size}",
        f"--learning_rate {learning_rate}",
        f"--dropout {dropout}",
        f"--dropout_n {dropout_n}",
        f"--d_model {d_model}",
        f"--d_ff {d_ff}",
        f"--n_heads {n_heads}",
        f"--e_layers {e_layers}",
        f"--temp_patch_len {temp_patch_len}",
        f"--temp_stride {temp_stride}",
        f"--embed_size {embed_size}",
        f"--phase_period_len {phase_period_len}",
        f"--phase_num_routers {phase_num_routers}",
        f"--phase_alpha_init {phase_alpha_init}",
        f"--phase_trust_bias_init {phase_trust_bias_init}",
        f"--epochs {epochs}",
        f"--es_patience {es_patience}",
        f"--loss_mode {loss_mode}",
        f"--lossfun_alpha {lossfun_alpha}",
        f"--weight_decay {weight_decay}",
        f"--seed {seed}",
        f"--model_tag {model_tag}",
        f"--num_workers {num_workers}",
        "--speed_mode strict",
    ]
    setup = [
        'export PYTHONPATH="${PWD}/src:${PYTHONPATH}"',
        "export OMP_NUM_THREADS=4",
        "export MKL_NUM_THREADS=4",
    ]
    return " && ".join(setup + [" ".join(python_cmd)])


def add_job(jobs, idx, name, cmd):
    jobs.append(
        {
            "name": name,
            "cwd": ".",
            "gpu": GPU_POOL[idx % len(GPU_POOL)],
            "cmd": cmd,
        }
    )


def main():
    jobs = []
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    idx = 0
    seeds = [2032, 2033, 2034, 2035, 2036]

    etth2_base = dict(
        model_type="frwkv_crossbranchphasegate_adaptive",
        data_path="ETTh2",
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        batch_size=32,
        learning_rate=1e-4,
        dropout=0.2,
        dropout_n=0.2,
        d_model=512,
        d_ff=512,
        n_heads=8,
        e_layers=2,
        temp_patch_len=16,
        temp_stride=8,
        embed_size=8,
        epochs=40,
        es_patience=8,
        loss_mode="L1",
        lossfun_alpha=0.5,
        weight_decay=1e-3,
    )
    etth2_cfgs = [
        ("best", 48, 2, 0.02, -3.5),
        ("conservative", 12, 1, 0.0, -3.5),
    ]
    for label, pp, routers, alpha, bias in etth2_cfgs:
        for seed in seeds:
            tag = f"kbs_confirm_etth2_pl96_{label}_pp{pp}_r{routers}_a{alpha}_b{bias}_seed{seed}"
            cmd = build_cmd(
                phase_period_len=pp,
                phase_num_routers=routers,
                phase_alpha_init=alpha,
                phase_trust_bias_init=bias,
                seed=seed,
                model_tag=tag,
                **etth2_base,
            )
            add_job(jobs, idx, tag, cmd)
            idx += 1

    ili_base = dict(
        model_type="frwkv_crossbranchphasegate_adaptive",
        data_path="ILI.csv",
        num_nodes=7,
        seq_len=36,
        pred_len=36,
        batch_size=16,
        learning_rate=1e-4,
        dropout=0.2,
        dropout_n=0.2,
        d_model=512,
        d_ff=512,
        n_heads=8,
        e_layers=2,
        temp_patch_len=12,
        temp_stride=6,
        embed_size=8,
        epochs=90,
        es_patience=15,
        loss_mode="L1",
        lossfun_alpha=0.5,
        weight_decay=1e-3,
    )
    ili_cfgs = [
        ("best", 12, 4, 0.01, -4.5),
        ("conservative", 12, 4, 0.0, -4.5),
    ]
    for label, pp, routers, alpha, bias in ili_cfgs:
        for seed in seeds:
            tag = f"kbs_confirm_ili_pl36_{label}_pp{pp}_r{routers}_a{alpha}_b{bias}_seed{seed}"
            cmd = build_cmd(
                phase_period_len=pp,
                phase_num_routers=routers,
                phase_alpha_init=alpha,
                phase_trust_bias_init=bias,
                seed=seed,
                model_tag=tag,
                **ili_base,
            )
            add_job(jobs, idx, tag, cmd)
            idx += 1

    OUT_PATH.write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(jobs)} jobs to {OUT_PATH}")


if __name__ == "__main__":
    main()
