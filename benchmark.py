"""Benchmark an MLP with CPU dynamic INT8 and optional GPU FP16 inference."""

from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import platform
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(768, 3072), nn.GELU(), nn.Linear(3072, 768))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def state_size_mib(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return round(buffer.tell() / 2**20, 3)


def timing(model: nn.Module, x: torch.Tensor, warmup: int, repeats: int) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            for start, end in zip(starts, ends):
                start.record()
                model(x)
                end.record()
            torch.cuda.synchronize()
            values = [start.elapsed_time(end) for start, end in zip(starts, ends)]
        else:
            values = []
            for _ in range(repeats):
                start = time.perf_counter()
                model(x)
                values.append((time.perf_counter() - start) * 1000)
    return statistics.median(values), statistics.pstdev(values)


def accuracy(reference: torch.Tensor, output: torch.Tensor) -> dict[str, float]:
    a, b = reference.float().flatten(), output.float().flatten()
    delta = b - a
    return {
        "max_abs_error": float(delta.abs().max()),
        "relative_l2_error": float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a)),
        "cosine_similarity": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
    }


def select_quantization_engine(supported: list[str]) -> str:
    """Retain the original preference, with oneDNN for builds that only ship it."""
    engine = next((e for e in ("x86", "fbgemm", "qnnpack", "onednn") if e in supported), None)
    if engine is None:
        raise RuntimeError(f"No supported dynamic quantization backend: {supported}")
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8, 32, 128])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if any(n < 1 for n in args.batch_sizes) or args.threads < 1 or args.warmup < 0 or args.repeats < 2:
        parser.error("batch sizes and threads must be positive; repeats must be at least 2")
    torch.manual_seed(2026)
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False

    fp32_cpu = TinyMLP().eval()
    supported = torch.backends.quantized.supported_engines
    engine = select_quantization_engine(supported)
    torch.backends.quantized.engine = engine
    int8_cpu = quantize_dynamic(copy.deepcopy(fp32_cpu), {nn.Linear}, dtype=torch.qint8).eval()
    use_cuda = torch.cuda.is_available() and not args.cpu_only
    fp32_cuda = copy.deepcopy(fp32_cpu).cuda().eval() if use_cuda else None
    fp16_cuda = copy.deepcopy(fp32_cpu).cuda().half().eval() if use_cuda else None
    model_sizes = {"cpu_fp32": state_size_mib(fp32_cpu), "cpu_int8_dynamic": state_size_mib(int8_cpu)}
    if use_cuda:
        model_sizes.update(cuda_fp32=state_size_mib(fp32_cuda), cuda_fp16=state_size_mib(fp16_cuda))

    rows: list[dict] = []
    for batch in args.batch_sizes:
        x_cpu = torch.randn(batch, 768)
        with torch.inference_mode():
            ref_cpu = fp32_cpu(x_cpu)
        for name, model, x, reference in (("cpu_fp32", fp32_cpu, x_cpu, ref_cpu),
                                          ("cpu_int8_dynamic", int8_cpu, x_cpu, ref_cpu)):
            with torch.inference_mode():
                output = model(x)
            median, stdev = timing(model, x, args.warmup, args.repeats)
            rows.append({"batch": batch, "variant": name, "device": "cpu",
                         "median_ms": median, "stdev_ms": stdev,
                         "model_state_mib": model_sizes[name], **accuracy(reference, output)})
            print(f"batch={batch:3d} {name:16s} {median:.4f} ms", flush=True)
        if use_cuda:
            x_cuda = x_cpu.cuda()
            with torch.inference_mode():
                ref_cuda = fp32_cuda(x_cuda)
            for name, model, x in (("cuda_fp32", fp32_cuda, x_cuda),
                                   ("cuda_fp16", fp16_cuda, x_cuda.half())):
                with torch.inference_mode():
                    output = model(x)
                torch.cuda.synchronize()
                median, stdev = timing(model, x, args.warmup, args.repeats)
                rows.append({"batch": batch, "variant": name, "device": "cuda",
                             "median_ms": median, "stdev_ms": stdev,
                             "model_state_mib": model_sizes[name], **accuracy(ref_cuda, output)})
                print(f"batch={batch:3d} {name:16s} {median:.4f} ms", flush=True)

    for row in rows:
        baseline_name = "cpu_fp32" if row["device"] == "cpu" else "cuda_fp32"
        baseline = next(r["median_ms"] for r in rows if r["batch"] == row["batch"] and r["variant"] == baseline_name)
        row["speedup_vs_same_device_fp32"] = baseline / row["median_ms"]

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "torch": torch.__version__, "python": platform.python_version(),
        "platform": platform.platform(), "cpu_threads": args.threads,
        "quantized_engine": engine, "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "cuda_runtime": torch.version.cuda if use_cuda else None,
        "architecture": "Linear(768, 3072) -> GELU -> Linear(3072, 768)",
        "seed": 2026, "warmup": args.warmup, "repeats": args.repeats,
        "timing": "CPU wall clock or GPU CUDA events per call; median of repeats",
        "model_state_mib": "Serialized state_dict size, not process or device memory",
        "comparison": "Speedups are calculated against FP32 on the same device only",
        "note": "Accuracy compares random untrained model outputs; it does not measure task quality",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(rows)} rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
