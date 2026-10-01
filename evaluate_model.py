"""Validate deployment tradeoffs on pinned DistilBERT and SST-2 validation data."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import random

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic

from benchmark import select_quantization_engine, state_size_mib
from bench_utils import error_metrics, finish_run, measure, record, start_run, write_csv, write_json

MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
DATASET = "stanfordnlp/sst2"
DATASET_REVISION = "8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb"
VALIDATION = "data/validation-00000-of-00001.parquet"
GROUPS = ["batch", "max_length", "model_revision", "quality_split"]


def classification_metrics(labels, reference, output):
    metrics = error_metrics(reference, output)
    baseline = reference.argmax(dim=-1)
    predicted = output.argmax(dim=-1)
    good = predicted == labels
    base_good = baseline == labels
    metrics.update(examples=len(labels), correct=int(good.sum()), accuracy=float(good.float().mean()),
                   baseline_correct=int(base_good.sum()), changed_predictions=int((predicted != baseline).sum()),
                   regressions=int((base_good & ~good).sum()), recoveries=int((~base_good & good).sum()),
                   accuracy_drop_pp=max(0., 100. * (int(base_good.sum()) - int(good.sum())) / len(labels)))
    return metrics


def infer(model, encoded, device):
    inputs = {key: value.to(device) for key, value in encoded.items()}
    return model(**inputs).logits.float().cpu()


def main():
    # Optional imports: the core benchmarks and offline unit tests need only torch.
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/model-validation"))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--limit", type=int, default=872)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if min(args.batch_sizes + [args.limit, args.threads, args.trials]) < 1 or not 2 <= args.max_length <= 512 or args.samples < 2 or args.warmup < 0:
        parser.error("positive sizes/threads/trials required; max-length 2..512; samples >=2; warmup >=0")
    if max(args.batch_sizes) > args.limit:
        parser.error("batch size must not exceed validation subset size")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.quantized.engine = select_quantization_engine(torch.backends.quantized.supported_engines)
    cuda = torch.cuda.is_available() and not args.cpu_only
    metadata = start_run(args.output, args, GROUPS, "variant",
                         "DistilBERT SST-2 validation; timing includes CPU text tokenization, input transfers, model forward and CPU FP32 logits. Model loading/network/serving excluded. No training or accuracy tuning.")
    metadata["sources"] = {"model": MODEL, "model_revision": MODEL_REVISION, "dataset": DATASET,
                           "dataset_revision": DATASET_REVISION, "split": "validation", "file": VALIDATION,
                           "model_license": "Apache-2.0", "dataset_license": "unknown in dataset card; data are not redistributed"}
    dataset_path = hf_hub_download(DATASET, VALIDATION, repo_type="dataset", revision=DATASET_REVISION, local_files_only=args.local_files_only)
    metadata["dataset_sha256"] = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    dataset = pq.read_table(dataset_path).to_pylist()[:args.limit]
    labels = torch.tensor([r["label"] for r in dataset])
    texts = [r["sentence"] for r in dataset]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION, local_files_only=args.local_files_only, trust_remote_code=False)
    fp32 = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=MODEL_REVISION,
            local_files_only=args.local_files_only, trust_remote_code=False, use_safetensors=True, attn_implementation="eager").eval()
    models = {"cpu_fp32": fp32, "cpu_int8_dynamic": quantize_dynamic(copy.deepcopy(fp32), {nn.Linear}, dtype=torch.qint8).eval()}
    if cuda:
        models.update(cuda_fp32=copy.deepcopy(fp32).cuda(), cuda_fp16=copy.deepcopy(fp32).cuda().half())
    sizes = {name: state_size_mib(model) for name, model in models.items()}
    metadata.update(examples=len(dataset), attention_implementation="eager", padding="max_length",
                    quantization="Dynamic INT8 Linear only; embeddings and LayerNorm remain float",
                    quality_note="All variants evaluated at each batch size because dynamic activation quantization can depend on batch. Validation accuracy is descriptive, not an independent test-set/generalization guarantee.")
    write_json(args.output / "metadata.json", metadata)
    encoded = tokenizer(texts, padding="max_length", truncation=True, max_length=args.max_length, return_tensors="pt")
    qualities, predictions, metrics_by_case = [], [], {}
    with torch.inference_mode():
        for batch in args.batch_sizes:
            baseline = None
            for variant, model in models.items():
                device = "cuda" if variant.startswith("cuda") else "cpu"
                chunks = []
                for offset in range(0, len(texts), batch):
                    chunks.append(infer(model, {k: v[offset:offset + batch] for k, v in encoded.items()}, device))
                logits = torch.cat(chunks)
                if variant == "cpu_fp32":
                    baseline = logits
                metrics = classification_metrics(labels, baseline, logits)
                metrics_by_case[batch, variant] = metrics
                qualities.append(dict(batch=batch, variant=variant, model_state_mib=sizes[variant], **metrics))
                for index, row in enumerate(dataset):
                    predictions.append(dict(idx=row["idx"], batch=batch, variant=variant, label=row["label"],
                                            prediction=int(logits[index].argmax()), logit_0=float(logits[index, 0]), logit_1=float(logits[index, 1])))
                print(f"quality batch={batch} {variant}: {metrics['correct']}/{len(labels)} correct; {metrics['changed_predictions']} changed vs CPU FP32", flush=True)
                write_csv(args.output / "quality.csv", qualities)
            del baseline
    write_csv(args.output / "predictions.csv", predictions)
    rows = []
    for trial in range(args.trials):
        rng = random.Random(args.seed + trial)
        options = [(b, v) for b in args.batch_sizes for v in models]
        rng.shuffle(options)
        for batch, variant in options:
            model = models[variant]
            device = "cuda" if variant.startswith("cuda") else "cpu"
            # Same deterministic representative request for all variants within a trial;
            # rotate between trials. Full quality evaluation above uses all examples.
            offset = (trial * max(args.batch_sizes)) % (len(texts) - max(args.batch_sizes) + 1)
            request_texts = texts[offset:offset + batch]
            def fn():
                request_inputs = tokenizer(request_texts, padding="max_length", truncation=True,
                                           max_length=args.max_length, return_tensors="pt")
                return infer(model, request_inputs, device)
            row = dict(batch=batch, max_length=args.max_length, model_revision=MODEL_REVISION, quality_split="sst2_validation",
                       variant=variant, trial=trial, items_per_request=batch, model_state_mib=sizes[variant],
                       request_offset=offset, status="ok", reason="", **metrics_by_case[batch, variant])
            row.update(measure(fn, device, args.warmup, args.samples))
            record(args.output, row)
            rows.append(row)
            print(f"timing trial={trial} batch={batch} {variant}: {row['median_ms']:.3f} ms / P95 {row['p95_ms']:.3f} ms", flush=True)
    finish_run(args.output, metadata, rows)


if __name__ == "__main__":
    main()
