"""Profile resident compute and CPU-input/CPU-output MLP requests separately."""
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
import random

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic

from benchmark import TinyMLP, select_quantization_engine, state_size_mib
from bench_utils import error_metrics, finish_run, measure, record, start_run

GROUPS = ["profile", "comparison_scope", "batch", "input_features", "hidden_features", "cpu_threads"]


def request(model, x_cpu, variant):
    """Input and output are CPU FP32 tensors; CUDA transfers/casts are included."""
    if variant.startswith("cuda"):
        dtype = torch.float16 if variant.endswith("fp16") else torch.float32
        return model(x_cpu.to(device="cuda", dtype=dtype)).float().cpu()
    return model(x_cpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/profiles"))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 128, 512])
    parser.add_argument("--profiles", nargs="+", choices=["resident", "roundtrip"], default=["resident", "roundtrip"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    if min(args.batch_sizes + [args.trials, args.threads]) < 1 or args.samples < 2 or args.warmup < 0:
        parser.error("sizes, trials and threads must be positive; samples >= 2; warmup >= 0")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.quantized.engine = select_quantization_engine(torch.backends.quantized.supported_engines)
    cuda = torch.cuda.is_available() and not args.cpu_only
    metadata = start_run(args.output, args, GROUPS, "variant",
                         "Resident mode compares variants on the same device. Roundtrip mode compares all devices with CPU FP32 input/output, including blocking CUDA transfers/casts. No serving/network/model-loading time.")
    rows = []
    for trial in range(args.trials):
        torch.manual_seed(args.seed + trial)
        rng = random.Random(args.seed + trial)
        fp32 = TinyMLP().eval()
        models = {"cpu_fp32": fp32, "cpu_int8_dynamic": quantize_dynamic(copy.deepcopy(fp32), {nn.Linear}, dtype=torch.qint8).eval()}
        if cuda:
            models.update(cuda_fp32=copy.deepcopy(fp32).cuda(), cuda_fp16=copy.deepcopy(fp32).cuda().half())
        sizes = {name: state_size_mib(model) for name, model in models.items()}
        for batch in args.batch_sizes:
            x = torch.randn(batch, 768)
            with torch.inference_mode():
                reference_cpu = fp32(x)
                reference_cuda = models["cuda_fp32"](x.cuda()) if cuda else None
            options = list(itertools.product(args.profiles, models))
            rng.shuffle(options)
            for profile, variant in options:
                device = "cuda" if variant.startswith("cuda") else "cpu"
                model = models[variant]
                row = dict(profile=profile, comparison_scope="cpu_io" if profile == "roundtrip" else device,
                           batch=batch, input_features=768, hidden_features=3072, cpu_threads=args.threads,
                           variant=variant, device=device, trial=trial, seed=args.seed + trial,
                           items_per_request=batch, model_state_mib=sizes[variant])
                if profile == "roundtrip":
                    fn = lambda: request(model, x, variant)
                    reference = reference_cpu
                else:
                    resident = x.to(device=device, dtype=torch.float16 if variant == "cuda_fp16" else torch.float32)
                    fn = lambda: model(resident)
                    reference = reference_cuda if device == "cuda" else reference_cpu
                with torch.inference_mode():
                    output = fn()
                    metrics = error_metrics(reference, output)
                    del output
                    row.update(measure(fn, device, args.warmup, args.samples))
                row.update(metrics, status="ok", reason="")
                rows.append(row)
                record(args.output, row)
            print(f"trial={trial} batch={batch}: recorded {len(options)} options", flush=True)
        del models, fp32, model, fn, reference, reference_cpu, reference_cuda
        if cuda:
            torch.cuda.empty_cache()
    summary = finish_run(args.output, metadata, rows)
    print(f"Saved {len(rows)} measurements / {len(summary)} configurations to {args.output}")


if __name__ == "__main__":
    main()
