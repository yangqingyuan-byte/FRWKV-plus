#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FINAL = RESULTS / "final_evidence"
CONFIG = ROOT / "src" / "adaptive_phasegate_kbs" / "configs" / "kbs_ours_recipes.json"


SELECTED_PROVENANCE = FINAL / "KBS_selected_system_provenance_audit_2026-05-10.csv"
TABLE4_PROVENANCE = FINAL / "KBS_table4_selected_single_run_provenance_2026-05-11.csv"
MATCHED16_JSON = FINAL / "KBS_full_family_matched16_final_analysis_2026-05-09.json"
MATCHED16_CSV = FINAL / "KBS_full_family_matched16_final_analysis_2026-05-09.csv"
MATCHED16_RAW_ROWS = FINAL / "KBS_full_family_matched16_final_raw_rows_2026-05-09.csv"
RUNTIME_PROFILE = FINAL / "KBS_main_model_runtime_profile_results_2026-05-11.csv"
SOURCE_SNAPSHOT_MANIFEST = FINAL / "source_snapshot_manifest_2026-05-13.csv"


EXPECTED_SELECTED_VALUES = {
    ("ETTh2", 96): (0.277616, 0.327223, "status_json_exact"),
    ("ILI", 36): (1.391716, 0.714074, "status_json_exact"),
    ("Exchange", 720): (0.840333, 0.689541, "summary_log_exact"),
}

EXPECTED_TABLE4_VALUES = {
    ("ETTh1", 336, "Ours(selected)"): (2025, 0.462054, 0.440882),
    ("ETTh2", 96, "Ours(selected)"): (2034, 0.277616, 0.327223),
    ("ILI", 36, "Ours(selected)"): (2025, 1.391716, 0.714074),
    ("Weather", 720, "FRWKV"): (2028, 0.341823, 0.335278),
}

EXPECTED_MODELS = {
    "frwkv",
    "frwkv_crossbranchgate",
    "frwkv_crossbranchphasegate",
    "frwkv_crossbranchphasegate_fullcontextdelta",
    "frwkv_crossbranchperiodicpositiongate_adaptive",
    "frwkv_crossbranchphasegate_adaptive",
    "frwkv_crossbranchphasegate_adaptive_linearproj",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"missing required evidence file: {path}")
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def read_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"missing required evidence file: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def assert_close(label: str, value: float, expected: float, tol: float = 1e-9) -> None:
    if not math.isclose(value, expected, rel_tol=0.0, abs_tol=tol):
        raise SystemExit(f"{label}: expected {expected}, got {value}")


def verify_selected_provenance() -> None:
    rows = read_csv(SELECTED_PROVENANCE)
    if len(rows) != 28:
        raise SystemExit(f"selected provenance row count mismatch: expected 28, got {len(rows)}")

    bad_status = [
        row
        for row in rows
        if not row["trace_status"].endswith("_exact") or row["candidate_count_rounded_exact"] in {"", "0"}
    ]
    if bad_status:
        settings = [(row["dataset"], row["horizon"], row["trace_status"]) for row in bad_status]
        raise SystemExit(f"selected provenance contains non-exact traces: {settings}")

    indexed = {(row["dataset"], int(row["horizon"])): row for row in rows}
    for key, (mse, mae, status) in EXPECTED_SELECTED_VALUES.items():
        row = indexed.get(key)
        if row is None:
            raise SystemExit(f"missing selected provenance row: {key}")
        assert_close(f"{key} selected MSE", float(row["test_mse"]), mse)
        assert_close(f"{key} selected MAE", float(row["test_mae"]), mae)
        if row["trace_status"] != status:
            raise SystemExit(f"{key} trace status mismatch: expected {status}, got {row['trace_status']}")

    print("ok selected-system provenance: 28 exact traced cells")


def verify_table4_provenance() -> None:
    rows = read_csv(TABLE4_PROVENANCE)
    if len(rows) != 76:
        raise SystemExit(f"Table 4 provenance row count mismatch: expected 76, got {len(rows)}")

    missing_source = [row for row in rows if not row["source_file"] or not row["selection_rule"]]
    if missing_source:
        raise SystemExit(f"Table 4 provenance rows missing source metadata: {len(missing_source)}")

    ours_rows = [row for row in rows if row["model"] == "Ours(selected)"]
    if len(ours_rows) != 19:
        raise SystemExit(f"Table 4 Ours(selected) row count mismatch: expected 19, got {len(ours_rows)}")

    indexed = {(row["dataset"], int(row["horizon"]), row["model"]): row for row in rows}
    for key, (seed, mse, mae) in EXPECTED_TABLE4_VALUES.items():
        row = indexed.get(key)
        if row is None:
            raise SystemExit(f"missing Table 4 provenance row: {key}")
        if int(row["seed"]) != seed:
            raise SystemExit(f"{key} seed mismatch: expected {seed}, got {row['seed']}")
        assert_close(f"{key} MSE", float(row["mse"]), mse)
        assert_close(f"{key} MAE", float(row["mae"]), mae)

    print("ok selected single-run provenance: 76 traced Table 4 rows")


def verify_matched16_analysis() -> None:
    summary = read_json(MATCHED16_JSON)
    if not MATCHED16_CSV.exists():
        raise SystemExit(f"missing matched16 CSV: {MATCHED16_CSV}")
    raw_rows = read_csv(MATCHED16_RAW_ROWS)

    load_meta = summary["load_meta"]
    validation = summary["validation"]
    if load_meta["deduped_rows"] != 2240 or validation["observed_rows"] != 2240:
        raise SystemExit(f"matched16 row count mismatch: {load_meta}, {validation}")
    if len(raw_rows) != 2240:
        raise SystemExit(f"matched16 raw row count mismatch: expected 2240, got {len(raw_rows)}")
    if validation["expected_rows"] != 2240:
        raise SystemExit(f"matched16 expected row count changed: {validation['expected_rows']}")
    for key in ["duplicate_setting_seed_rows", "missing_setting_seed_rows", "bad_seed_groups"]:
        if validation[key]:
            raise SystemExit(f"matched16 validation failure in {key}: {validation[key]}")

    winner_counts = summary["winner_counts"]
    average_ranks = summary["average_ranks"]
    expected_winners = {
        ("mse", "Adaptive PhaseGate"): 8,
        ("mse", "FRWKV"): 7,
        ("mae", "Adaptive PhaseGate"): 7,
        ("mae", "FRWKV"): 12,
    }
    for (metric, model), expected in expected_winners.items():
        actual = winner_counts[metric][model]
        if actual != expected:
            raise SystemExit(f"{metric} winner count for {model}: expected {expected}, got {actual}")
    assert_close("Adaptive PhaseGate MSE average rank", average_ranks["mse"]["Adaptive PhaseGate"], 2.9642857142857144)
    assert_close("FRWKV MAE average rank", average_ranks["mae"]["FRWKV"], 2.4642857142857144)

    claim = summary["result_to_claim"]
    if claim["claim_supported"] != "partial":
        raise SystemExit(f"unexpected claim support label: {claim['claim_supported']}")

    raw_keys = {
        (row["dataset"], int(row["horizon"]), row["model"], int(row["seed"]))
        for row in raw_rows
    }
    if len(raw_keys) != 2240:
        raise SystemExit("matched16 raw rows contain duplicate dataset/horizon/model/seed keys")

    print("ok matched 16-seed ablation: 2240 raw rows, no missing seeds, claim label partial")


def verify_runtime_profile() -> None:
    rows = read_csv(RUNTIME_PROFILE)
    if len(rows) != 28:
        raise SystemExit(f"runtime profile row count mismatch: expected 28, got {len(rows)}")
    bad = [row for row in rows if row["status"] != "completed"]
    if bad:
        settings = [(row["dataset"], row["horizon"], row["status"]) for row in bad]
        raise SystemExit(f"runtime profile contains unfinished rows: {settings}")
    if {row["run_id"] for row in rows} != {"parallel_run_20260511_121021"}:
        raise SystemExit("runtime profile run_id set does not match the final profiled campaign")

    print("ok runtime profile evidence: 28 completed runs")


def verify_source_snapshots() -> None:
    rows = read_csv(SOURCE_SNAPSHOT_MANIFEST)
    if len(rows) != 112:
        raise SystemExit(f"source snapshot manifest row count mismatch: expected 112, got {len(rows)}")
    for row in rows:
        path = ROOT / row["snapshot_file"]
        if not path.exists():
            raise SystemExit(f"missing source snapshot: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise SystemExit(f"source snapshot checksum mismatch: {path}")
        if path.stat().st_size != int(row["bytes"]):
            raise SystemExit(f"source snapshot byte-size mismatch: {path}")

    print("ok raw source snapshots: 112 referenced status/log files with checksums")


def verify_recipe_config() -> None:
    config = read_json(CONFIG)
    recipes = config.get("recipes", {})
    main_group = config.get("recipe_groups", {}).get("paper_current.main_table_all", [])
    if len(main_group) != 28:
        raise SystemExit(f"main table recipe group size mismatch: expected 28, got {len(main_group)}")
    missing_recipes = [name for name in main_group if name not in recipes]
    if missing_recipes:
        raise SystemExit(f"main table recipe group references missing recipes: {missing_recipes}")
    model_types = {recipes[name]["args"]["model_type"] for name in main_group}
    unsupported = model_types - EXPECTED_MODELS
    if unsupported:
        raise SystemExit(f"recipe config contains unsupported model types: {sorted(unsupported)}")

    etth2_96 = recipes["paper_current.adaptive.etth2_pl96"]["args"]
    expected = {
        "model_type": "frwkv_crossbranchperiodicpositiongate_adaptive",
        "data_path": "ETTh2",
        "pred_len": 96,
        "period_position_len": 48,
        "period_position_num_routers": 2,
        "period_context_alpha_init": 0.02,
        "period_context_trust_bias_init": -4.0,
        "embed_size": 16,
        "loss_mode": "L1",
        "lossfun_alpha": 0.5,
    }
    for key, expected_value in expected.items():
        actual = etth2_96[key]
        if actual != expected_value:
            raise SystemExit(f"ETTh2-96 recipe {key}: expected {expected_value}, got {actual}")

    print("ok packaged recipe config: 28 current main-table recipes")


def main() -> None:
    verify_selected_provenance()
    verify_table4_provenance()
    verify_matched16_analysis()
    verify_runtime_profile()
    verify_source_snapshots()
    verify_recipe_config()
    print("release evidence integrity check passed")


if __name__ == "__main__":
    main()
