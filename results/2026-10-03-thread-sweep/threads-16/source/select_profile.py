"""Export a constrained choice for every measured workload, including no-match cases."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def number(row, key):
    try:
        value = float(row[key])
        return value if math.isfinite(value) and value >= 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def choose(rows, groups, option, *, max_error, max_p95=None, max_memory=None,
           max_size=None, max_spread=None, min_trials=3, max_accuracy_drop=None, only_options=None):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in groups), []).append(row)
    decisions = []
    for key, candidates in grouped.items():
        valid, rejected = [], []
        for row in candidates:
            reasons = []
            if only_options is not None and row[option] not in only_options:
                reasons.append("option_not_allowed")
            if row["status"] != "ok":
                reasons.append(row["status"])
            count, expected = number(row, "successful_trials"), number(row, "expected_trials")
            if count is None or expected is None or count < min_trials or count != expected:
                reasons.append("insufficient_complete_trials")
            for metric in ("median_ms", "p95_ms", "relative_l2_error", "spread_ratio"):
                value = number(row, metric)
                if value is None or (metric in ("median_ms", "p95_ms", "spread_ratio") and value <= 0):
                    reasons.append("invalid_" + metric)
            for metric, limit in (("relative_l2_error", max_error), ("p95_ms", max_p95),
                                  ("peak_extra_mib", max_memory), ("model_state_mib", max_size),
                                  ("spread_ratio", max_spread), ("accuracy_drop_pp", max_accuracy_drop)):
                if limit is not None:
                    value = number(row, metric)
                    if value is None or value > limit:
                        reasons.append(metric + "_constraint")
            if reasons:
                rejected.append({"option": row[option], "reasons": reasons})
            else:
                valid.append(row)
        decision = {"workload": dict(zip(groups, key)), "status": "selected" if valid else "no_match", "rejected": rejected}
        if valid:
            best = min(valid, key=lambda r: (float(r["median_ms"]), r[option]))
            decision.update(selected=best[option], measured_median_ms=float(best["median_ms"]),
                            worst_trial_p95_ms=float(best["p95_ms"]), worst_relative_l2_error=float(best["relative_l2_error"]),
                            feasible_options=[r[option] for r in valid])
        decisions.append(decision)
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-error", type=float, default=0.001)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--max-extra-mib", type=float)
    parser.add_argument("--max-model-mib", type=float)
    parser.add_argument("--max-spread-ratio", type=float)
    parser.add_argument("--max-accuracy-drop-pp", type=float,
                        help="Maximum measured validation accuracy loss in percentage points; requires model evaluation data")
    parser.add_argument("--min-trials", type=int, default=3)
    parser.add_argument("--only-options", nargs="+", help="Allow only these measured variants, e.g. cpu_fp32 cpu_int8_dynamic")
    args = parser.parse_args()
    limits = [args.max_relative_error, args.max_p95_ms, args.max_extra_mib, args.max_model_mib, args.max_spread_ratio, args.max_accuracy_drop_pp]
    if any(v is not None and (not math.isfinite(v) or v < 0) for v in limits) or args.min_trials < 1:
        parser.error("constraints must be finite and nonnegative; min-trials must be positive")
    metadata = json.loads((args.run / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 2 or "completed_utc" not in metadata:
        parser.error("requires a completed schema-v2 profile; legacy files are not interchangeable")
    summary = args.run / "summary.csv"
    with summary.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    decisions = choose(rows, metadata["group_fields"], metadata["option_field"], max_error=args.max_relative_error,
                       max_p95=args.max_p95_ms, max_memory=args.max_extra_mib, max_size=args.max_model_mib,
                       max_spread=args.max_spread_ratio, min_trials=args.min_trials, max_accuracy_drop=args.max_accuracy_drop_pp,
                       only_options=args.only_options)
    result = {"schema_version": 2, "environment": metadata["environment"],
              "environment_fingerprint": metadata["environment_fingerprint"],
              "summary_sha256": hashlib.sha256(summary.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
              "summary_hash_normalization": "LF newlines",
              "constraints": {k: v for k, v in vars(args).items() if k not in ("run", "output")},
              "scope": "Measured choices only, for exact workload and environment. P95 is an empirical bound, not an SLA guarantee. Unknown shapes require measurement.",
              "decisions": decisions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{sum(d['status'] == 'selected' for d in decisions)}/{len(decisions)} workloads have a feasible choice; saved {args.output}")


if __name__ == "__main__":
    main()
