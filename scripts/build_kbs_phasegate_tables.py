#!/usr/bin/env python3
import json
import re
import statistics as st
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
EXPERIMENT_LOG = ROOT / "results" / "paper_results.jsonl"
OUT_MD = DOCS / "KBS_phasegate_tables_draft.md"


PAT_ADAPTIVE_MS = re.compile(r"^kbs_adaptive_ms_(?P<data>.+)_pl(?P<pred>\d+)_seed(?P<seed>\d+)$")
PAT_ABLATION = re.compile(
    r"^kbs_ablation_(?P<model>.+)_(?P<data>etth2|exchange|ili)_pl(?P<pred>\d+)_seed(?P<seed>\d+)$"
)
PAT_TUNE = re.compile(
    r"^kbs_tune_(?P<data>etth2|ili)_pl(?P<pred>\d+)_pp(?P<pp>\d+)_r(?P<r>\d+)_a(?P<a>-?\d+(?:\.\d+)?)_b(?P<b>-?\d+(?:\.\d+)?)$"
)
PAT_CONFIRM = re.compile(
    r"^kbs_confirm_(?P<data>etth2|ili)_pl(?P<pred>\d+)_(?P<label>best|conservative)_pp(?P<pp>\d+)_r(?P<r>\d+)_a(?P<a>-?\d+(?:\.\d+)?)_b(?P<b>-?\d+(?:\.\d+)?)_seed(?P<seed>\d+)$"
)
PAT_MATCH = re.compile(
    r"^kbs_match_(?P<model>crossbranchphasegate|frwkv)_(?P<data>etth2|ili)_pl(?P<pred>\d+)_seed(?P<seed>\d+)$"
)


def load_rows():
    rows = []
    for line in EXPERIMENT_LOG.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return st.mean(values), st.pstdev(values)


def format_mean_std(values):
    mean, std = mean_std(values)
    return f"{mean:.6f} ± {std:.6f}"


def main():
    rows = load_rows()

    adaptive_ms = {}
    ablation = {}
    tuning = {}
    confirm = {}
    match = {}

    for row in rows:
        tag = row.get("model_tag", "")

        m = PAT_ADAPTIVE_MS.match(tag)
        if m:
            key = (m.group("data").lower(), int(m.group("pred")))
            adaptive_ms.setdefault(key, []).append(row)
            continue

        m = PAT_ABLATION.match(tag)
        if m:
            key = (m.group("data"), int(m.group("pred")), m.group("model"))
            ablation.setdefault(key, []).append(row)
            continue

        m = PAT_TUNE.match(tag)
        if m:
            key = (m.group("data"), int(m.group("pred")))
            tuning.setdefault(key, []).append(
                {
                    "pp": int(m.group("pp")),
                    "r": int(m.group("r")),
                    "a": float(m.group("a")),
                    "b": float(m.group("b")),
                    "mse": row["test_mse"],
                    "mae": row["test_mae"],
                }
            )
            continue

        m = PAT_CONFIRM.match(tag)
        if m:
            key = (m.group("data"), int(m.group("pred")), m.group("label"))
            confirm.setdefault(key, []).append(row)
            continue

        m = PAT_MATCH.match(tag)
        if m:
            key = (m.group("data"), int(m.group("pred")), m.group("model"))
            match.setdefault(key, []).append(row)
            continue

    lines = []
    lines.append("# Legacy KBS PhaseGate Tables Draft")
    lines.append("")
    lines.append(
        "This file is generated from `results/paper_results.jsonl`, an April-era curated log retained for historical auditability."
    )
    lines.append(
        "For the current manuscript evidence chain, use `results/final_evidence/` and run `python scripts/verify_release_repro.py`."
    )
    lines.append("")
    lines.append("## Strong-Regime Multi-Seed Completion")
    lines.append("")
    lines.append("| Dataset | Pred | Seeds | Adaptive PhaseGate MSE | Adaptive PhaseGate MAE |")
    lines.append("| --- | --- | --- | --- | --- |")
    for (data, pred) in sorted(adaptive_ms):
        vals = adaptive_ms[(data, pred)]
        lines.append(
            f"| {data} | {pred} | {len(vals)} | "
            f"{format_mean_std([x['test_mse'] for x in vals])} | "
            f"{format_mean_std([x['test_mae'] for x in vals])} |"
        )

    lines.append("")
    lines.append("## Five-Seed Ablation")
    lines.append("")
    lines.append("| Setting | FRWKV | CrossBranchGate | CrossBranchPhaseGate | Adaptive PhaseGate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for setting in sorted(set((d, p) for d, p, _ in ablation)):
        d, p = setting
        row = [f"{d}-{p}"]
        for model in [
            "frwkv",
            "frwkv_crossbranchgate",
            "frwkv_crossbranchphasegate",
            "frwkv_crossbranchphasegate_adaptive",
        ]:
            vals = ablation[(d, p, model)]
            row.append(format_mean_std([x["test_mse"] for x in vals]))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Focused Tuning Best Single-Seed Candidates")
    lines.append("")
    lines.append("| Setting | phase\\_period\\_len | routers | alpha | trust bias | MSE | MAE |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for setting in sorted(tuning):
        best = sorted(tuning[setting], key=lambda x: (x["mse"], x["mae"]))[0]
        lines.append(
            f"| {setting[0]}-{setting[1]} | {best['pp']} | {best['r']} | {best['a']} | {best['b']} | "
            f"{best['mse']:.6f} | {best['mae']:.6f} |"
        )

    if confirm:
        lines.append("")
        lines.append("## Confirmation Runs")
        lines.append("")
        lines.append("| Setting | Label | Seeds | MSE | MAE |")
        lines.append("| --- | --- | --- | --- | --- |")
        for key in sorted(confirm):
            d, p, label = key
            vals = confirm[key]
            lines.append(
                f"| {d}-{p} | {label} | {len(vals)} | "
                f"{format_mean_std([x['test_mse'] for x in vals])} | "
                f"{format_mean_std([x['test_mae'] for x in vals])} |"
            )

    if match:
        lines.append("")
        lines.append("## Matched Baseline Confirmation")
        lines.append("")
        lines.append("| Setting | Model | Seeds | MSE | MAE |")
        lines.append("| --- | --- | --- | --- | --- |")
        for key in sorted(match):
            d, p, model = key
            vals = match[key]
            lines.append(
                f"| {d}-{p} | {model} | {len(vals)} | "
                f"{format_mean_std([x['test_mse'] for x in vals])} | "
                f"{format_mean_std([x['test_mae'] for x in vals])} |"
            )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
