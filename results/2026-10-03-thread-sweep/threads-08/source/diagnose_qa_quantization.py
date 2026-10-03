"""Separate weight rounding, dynamic activation error, and layerwise QA drift."""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.ao.quantization import quantize_dynamic, per_channel_dynamic_qconfig
from benchmark import select_quantization_engine
from bench_utils import environment, error_metrics, write_json, write_csv

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"


def quantized_copy(layer, per_channel=False):
    config = {nn.Linear: per_channel_dynamic_qconfig} if per_channel else {nn.Linear}
    return quantize_dynamic(nn.Sequential(copy.deepcopy(layer)).eval(), config, dtype=torch.qint8)[0]


@contextmanager
def rounded_mlp_weights(model, per_channel=False):
    """Diagnostic FP32 execution with INT8-rounded weights; NOT an INT8 speedup."""
    originals = []
    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and ".mlp." in name:
                quant = quantized_copy(module, per_channel)
                originals.append((module, module.weight))
                module.weight = nn.Parameter(quant.weight().dequantize(), requires_grad=False)
        yield len(originals)
    finally:
        for module, weight in originals:
            module.weight = weight


def compare_linear(layer, inputs):
    """Replay the identical activation through real quantized operators and FP32 controls."""
    rows = []
    with torch.inference_mode():
        for scope, x in (("prefill", inputs), ("last_token", inputs[:, -1:, :].contiguous())):
            reference = layer(x)
            for per_channel in (False, True):
                quant = quantized_copy(layer, per_channel)
                rounded = nn.functional.linear(x, quant.weight().dequantize(), quant.bias())
                actual = quant(x)
                tokenwise = torch.cat([quant(x[:, i:i+1, :].contiguous()) for i in range(x.shape[1])], dim=1)
                rows.append(dict(scope=scope, weight_granularity="channel" if per_channel else "tensor",
                                 weight_only_relative_l2=error_metrics(reference, rounded)["relative_l2_error"],
                                 dynamic_relative_l2=error_metrics(reference, actual)["relative_l2_error"],
                                 dynamic_vs_rounded_relative_l2=error_metrics(rounded, actual)["relative_l2_error"],
                                 tokenwise_dynamic_relative_l2=error_metrics(reference, tokenwise)["relative_l2_error"],
                                 input_abs_max=float(x.abs().max()), input_rms=float(x.square().mean().sqrt()),
                                 finite=bool(torch.isfinite(actual).all())))
    return rows


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error("output must not exist; threads must be positive")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.backends.quantized.engine = select_quantization_engine(torch.backends.quantized.supported_engines)
    torch.manual_seed(2026)
    source = Path(__file__).with_name("qa_probe.json")
    probe = json.loads(source.read_text(encoding="utf-8"))
    meta = dict(started_utc=datetime.now(timezone.utc).isoformat(), model=MODEL, revision=REVISION,
                environment=environment(), fixture_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                purpose="One known failing QA probe. Teacher-forced identical inputs isolate drift; layer replay isolates weight-only rounding from native dynamic quantization. No speed or general accuracy claim.")
    archive = args.output / "source"
    archive.mkdir()
    meta["source_sha256"] = {}
    for name in ("diagnose_qa_quantization.py", "qa_probe.json", "benchmark.py", "bench_utils.py"):
        data = Path(__file__).with_name(name).read_bytes()
        (archive / name).write_bytes(data)
        meta["source_sha256"][name] = hashlib.sha256(data).hexdigest()
    write_json(args.output / "metadata.json", meta)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, local_files_only=True,
                trust_remote_code=False, use_safetensors=True, dtype=torch.float32, attn_implementation="eager").eval()
    context = "\n\n".join(f"[{d['id']}] {d['title']}\n{d['text']}" for d in probe["documents"])
    prompt = tokenizer.apply_chat_template([{"role":"system","content":probe["system"]},
             {"role":"user","content":f"资料：\n{context}\n\n问题：{probe['question']}"}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    meta["input_sha256"] = hashlib.sha256(json.dumps(inputs["input_ids"][0].tolist()).encode()).hexdigest()
    meta["input_tokens"] = inputs["input_ids"].shape[1]
    layer_inputs, reference_hidden, candidate_hidden = {}, {}, {}
    hooks = []
    names = [f"model.layers.{index}.mlp.{projection}" for index in (0,13,27) for projection in ("up_proj","down_proj")]
    for name in names:
        def capture(module, args, output, key=name):
            layer_inputs[key] = args[0].detach().clone()
        hooks.append(model.get_submodule(name).register_forward_hook(capture))
    for index, layer in enumerate(model.model.layers):
        def hidden(module, args, output, key=index):
            value = output[0] if isinstance(output, tuple) else output
            reference_hidden[key] = value[:, -1, :].detach().clone()
        hooks.append(layer.register_forward_hook(hidden))
    with torch.inference_mode():
        before_embed = model.get_input_embeddings()(inputs["input_ids"]).clone()
        reference = model(**inputs, use_cache=False, logits_to_keep=1).logits.clone()
    for hook in hooks: hook.remove()
    rows = []
    for name, x in layer_inputs.items():
        print("replay", name, tuple(x.shape), flush=True)
        rows.extend(dict(module=name, **r) for r in compare_linear(model.get_submodule(name), x))
    write_csv(args.output / "linear_replay.csv", rows)
    del layer_inputs
    with torch.inference_mode():
        base_tokens = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True,
                                     pad_token_id=tokenizer.eos_token_id)[0, inputs["input_ids"].shape[1]:].tolist()
    controls = []
    for per_channel in (False, True):
        with rounded_mlp_weights(model, per_channel) as count, torch.inference_mode():
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits
            tokens = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True,
                                     pad_token_id=tokenizer.eos_token_id)[0, inputs["input_ids"].shape[1]:].tolist()
            controls.append(dict(weight_granularity="channel" if per_channel else "tensor", modules=count,
                                 execution="FP32 matmul with dequantized INT8-rounded weights; diagnostic only",
                                 final_logits=error_metrics(reference, logits), output_token_ids=tokens,
                                 answer=tokenizer.decode(tokens, skip_special_tokens=True)))
            print("weight-only control", controls[-1]["answer"], flush=True)
    write_json(args.output / "weight_only_controls.json", controls)
    selected = {name for name, module in model.named_modules() if isinstance(module, nn.Linear) and ".mlp." in name}
    model = quantize_dynamic(model, selected, dtype=torch.qint8, inplace=True).eval()
    meta["converted_modules"] = [name for name, module in model.named_modules() if isinstance(module, torch.ao.nn.quantized.dynamic.Linear)]
    if set(meta["converted_modules"]) != selected:
        raise RuntimeError("Unexpected converted layer set")
    hooks = []
    for index, layer in enumerate(model.model.layers):
        def hidden(module, args, output, key=index):
            value = output[0] if isinstance(output, tuple) else output
            candidate_hidden[key] = value[:, -1, :].detach().clone()
        hooks.append(layer.register_forward_hook(hidden))
    with torch.inference_mode():
        after_embed = model.get_input_embeddings()(inputs["input_ids"]).clone()
        candidate = model(**inputs, use_cache=False, logits_to_keep=1).logits.clone()
    for hook in hooks: hook.remove()
    with torch.inference_mode():
        quant_tokens = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True,
                                      pad_token_id=tokenizer.eos_token_id)[0, inputs["input_ids"].shape[1]:].tolist()
    drift = [dict(layer=i, **error_metrics(reference_hidden[i], candidate_hidden[i])) for i in reference_hidden]
    write_csv(args.output / "layer_drift.csv", drift)
    outcome = dict(embedding_bitwise_unchanged=torch.equal(before_embed, after_embed),
                   final_logits=error_metrics(reference, candidate),
                   baseline_first_token=int(reference[0,-1].argmax()), quantized_first_token=int(candidate[0,-1].argmax()),
                   baseline_answer=tokenizer.decode(base_tokens, skip_special_tokens=True),
                   quantized_answer=tokenizer.decode(quant_tokens, skip_special_tokens=True),
                   baseline_output_ids=base_tokens, quantized_output_ids=quant_tokens)
    write_json(args.output / "outcome.json", outcome)
    meta["completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.output / "metadata.json", meta)
    print(json.dumps(outcome, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
