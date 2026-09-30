"""Select the fastest measured precision per device and batch under quality/size limits."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/results.csv"))
    parser.add_argument("--max-relative-error", type=float, default=0.05)
    parser.add_argument("--max-model-mib", type=float, default=None)
    args = parser.parse_args()
    if args.max_relative_error < 0 or (args.max_model_mib is not None and args.max_model_mib < 0):
        parser.error("constraints must be nonnegative")
    with args.results.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        if float(row["relative_l2_error"]) > args.max_relative_error:
            continue
        if args.max_model_mib is not None and float(row["model_state_mib"]) > args.max_model_mib:
            continue
        groups.setdefault((row["device"], int(row["batch"])), []).append(row)
    print("device,batch,selected_variant,median_ms,relative_l2_error,model_state_mib")
    for key in sorted(groups):
        best = min(groups[key], key=lambda row: float(row["median_ms"]))
        print(",".join((key[0], str(key[1]), best["variant"], best["median_ms"],
                        best["relative_l2_error"], best["model_state_mib"])))
    if not groups:
        print("No measured configuration satisfies the constraints.")


if __name__ == "__main__":
    main()
