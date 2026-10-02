"""Utilities for repeated, synchronized request-latency measurements."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import statistics
import subprocess
import time

import torch


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def measure(fn, device, warmup, samples):
    """One blocking request per sample; includes Python/dispatch, excludes setup."""
    def sync():
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        sync()
        values = []
        for _ in range(samples):
            start = time.perf_counter_ns()
            output = fn()
            sync()
            values.append((time.perf_counter_ns() - start) / 1e6)
            del output
        extra = None
        if torch.device(device).type == "cuda":
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            output = fn()
            sync()
            extra = max(0, torch.cuda.max_memory_allocated(device) - baseline) / 2**20
            del output
    return {"median_ms": statistics.median(values), "p95_ms": percentile(values, 0.95),
            "peak_extra_mib": extra, "samples_ms": values}


def error_metrics(reference, output):
    a, b = reference.float(), output.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ArithmeticError("Non-finite reference or output")
    diff = b - a
    return {"max_abs_error": float(diff.abs().max()),
            "relative_l2_error": float(torch.linalg.vector_norm(diff) /
                                       torch.linalg.vector_norm(a).clamp_min(1e-12))}


def environment():
    cpu = platform.processor()
    if platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    try:
        driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        driver = None
    return {"python": platform.python_version(), "platform": platform.platform(), "cpu": cpu,
            "cpu_threads": torch.get_num_threads(), "torch": str(torch.__version__),
            "cuda_runtime": torch.version.cuda, "driver": driver,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "gpu_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
            "quantized_engine": torch.backends.quantized.engine, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}


def fingerprint(env):
    return hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()


def start_run(output, args, groups, option, description):
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("metadata.json", "samples.jsonl", "results.csv")):
        raise FileExistsError(f"Refusing to overwrite a previous run: {output}")
    env = environment()
    metadata = {"schema_version": 2, "started_utc": datetime.now(timezone.utc).isoformat(),
                "environment": env, "environment_fingerprint": fingerprint(env),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "group_fields": groups, "option_field": option, "description": description,
                "timing": "Synchronized wall-clock per request, including host dispatch; nearest-rank P95. Setup, warmup and model loading excluded.",
                "aggregation": "Median of trial medians; worst trial P95 and worst error/resource usage. Not a confidence interval.",
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}}
    write_json(output / "metadata.json", metadata)
    return metadata


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def record(output, row):
    with (output / "samples.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r if k != "samples_ms"))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, groups, option, trials):
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in groups + [option]), []).append(row)
    summary = []
    for key, observed in grouped.items():
        good = [r for r in observed if r["status"] == "ok"]
        unique_trials = {r["trial"] for r in good}
        status = "ok" if len(good) == trials and unique_trials == set(range(trials)) else "incomplete"
        if not good:
            status = observed[0]["status"]
        item = dict(zip(groups + [option], key))
        item.update(status=status, successful_trials=len(unique_trials), expected_trials=trials,
                    reason="; ".join(sorted({r.get("reason", "") for r in observed if r.get("reason")})))
        if good:
            medians = [r["median_ms"] for r in good]
            item.update(median_ms=statistics.median(medians), min_median_ms=min(medians), max_median_ms=max(medians),
                        spread_ratio=max(medians) / min(medians), p95_ms=max(r["p95_ms"] for r in good),
                        relative_l2_error=max(r["relative_l2_error"] for r in good),
                        max_abs_error=max(r["max_abs_error"] for r in good))
            for metric in ("peak_extra_mib", "model_state_mib"):
                values = [r.get(metric) for r in good]
                item[metric] = max(values) if all(v is not None for v in values) else None
            if all("accuracy_drop_pp" in r for r in good):
                item["accuracy_drop_pp"] = max(r["accuracy_drop_pp"] for r in good)
                item["accuracy"] = min(r["accuracy"] for r in good)
            item["items_per_second"] = good[0]["items_per_request"] * 1000 / item["median_ms"]
        summary.append(item)
    return summary


def finish_run(output, metadata, rows):
    summary = summarize(rows, metadata["group_fields"], metadata["option_field"], metadata["arguments"]["trials"])
    write_csv(output / "results.csv", rows)
    write_csv(output / "summary.csv", summary)
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "metadata.json", metadata)
    return summary
