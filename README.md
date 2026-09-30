# Linear Inference Bench | 线性层量化推理实验

An inference precision and deployment tradeoff study for a Transformer style feedforward block. It measures **latency, serialized model size, and output error** for CPU FP32, CPU dynamic INT8, GPU FP32, and GPU FP16. A small selector uses measured data to choose a variant under explicit quality and size constraints.

本项目研究部署时的精度选择，而不只是汇报单个速度数字。CPU 与 GPU 各自使用 FP32 对照；INT8 与 FP16 的误差使用同设备 FP32 输出计算，避免跨设备加速比造成误导。

## 核心问题与方法

模型结构：`Linear(768, 3072) → GELU → Linear(3072, 768)`，与 Transformer 前馈层形状相近。随机种子 2026，权重未训练，输入为随机张量。测试 batch 为 1、8、32、128。

| 变体 | 实现 | 对照 |
| --- | --- | --- |
| CPU FP32 | 原始 PyTorch `nn.Linear` | CPU 基线 |
| CPU INT8 dynamic | `torch.ao.quantization.quantize_dynamic`，仅量化 Linear | CPU FP32 |
| CUDA FP32 | 同一 FP32 权重的 CUDA 复制 | CUDA 基线 |
| CUDA FP16 | 同一权重与输入转换为 FP16 | CUDA FP32 |

每项预热 50 次，计时 100 次，报告单次 forward 的中位数和标准差。CPU 使用 4 线程及墙钟时间；GPU 使用 CUDA Event。模型大小是序列化 `state_dict` 的字节数，**不是运行时显存**。还计算最大绝对误差、相对 L2 误差与余弦相似度。

## 本机实测摘要

环境：AMD Ryzen 5 5600H、NVIDIA RTX 3050 Laptop GPU 4 GB、驱动 546.30、Windows 11、Python 3.12.6、PyTorch 2.5.1+cu121；动态量化引擎为 `x86`。完整数据见 `results/results.csv`。

| Batch | CPU FP32 | CPU INT8 | CPU 加速 | CUDA FP32 | CUDA FP16 | CUDA 加速 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.718 ms | 0.225 ms | 3.19× | 0.132 ms | 0.132 ms | 1.00× |
| 8 | 1.089 ms | 0.398 ms | 2.74× | 0.155 ms | 0.144 ms | 1.07× |
| 32 | 2.094 ms | 0.652 ms | 3.21× | 0.177 ms | 0.138 ms | 1.28× |
| 128 | 5.876 ms | 2.351 ms | 2.50× | 0.475 ms | 0.188 ms | 2.52× |

| 变体 | `state_dict` 大小 | 输出相对 L2 误差范围 |
| --- | ---: | ---: |
| FP32 | 18.016 MiB | 0 |
| CPU dynamic INT8 | 4.518 MiB | 0.0210–0.0284 |
| CUDA FP16 | 9.009 MiB | 0.00054–0.00061 |

本机 GPU 的 batch=1 配置中，FP16 没有明显速度优势；在 batch=128 时有 2.52 倍加速。因此选择精度应按输入规模与误差容忍度进行，而不是默认使用低精度。CPU INT8 的权重文件约为 FP32 的四分之一，所测 batch 的延迟均更低。

## 可复现运行

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe benchmark.py
.\.venv\Scripts\python.exe recommend.py --max-relative-error 0.03
```

无 GPU 环境：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python benchmark.py --cpu-only
```

`recommend.py` 接受 `--max-relative-error` 和 `--max-model-mib`，分别在 CPU 和 CUDA 的已有测量中选最快变体。它不会把 CPU 与 GPU 的延迟放进同一个加速比，也不推断未测 batch 的性能。

## 仓库结构

```text
benchmark.py        # 模型、量化、计时、误差和环境信息
recommend.py        # 误差/大小约束下的实测精度选择
tests/              # 动态量化类型和输出检查
results/results.csv # 原始逐配置结果
results/metadata.json
.github/workflows/   # CPU 正确性与小规模运行检查
```

## 解读与局限

- 模型权重和输入均随机生成，数值误差只表明输出接近程度，**不等于下游任务准确率**。
- 动态 INT8 的收益受 CPU 指令集、量化后端、线程数和 batch 影响；FP16 的收益受 GPU、功耗和形状影响。
- 测量只覆盖一个小型前馈块；没有计入模型加载、数据传输、预处理或完整服务开销。
- 一次基准运行的速度不是跨设备保证。CSV 保留标准差，便于判断短时波动。
- GPU FP16 计算采用普通 PyTorch eager 模型，没有声称自研量化算子。

## 简历可用表述

> 设计 Transformer 风格前馈层推理精度实验，比较 CPU 动态 INT8 和 GPU FP16 在 4 个 batch 下的延迟、权重大小与输出误差；在 Ryzen 5 5600H 上，CPU INT8 相对 FP32 加速 2.50–3.21 倍，序列化权重由 18.016 MiB 降至 4.518 MiB，并实现基于实测误差与大小约束的选择器。

## 参考

- [PyTorch dynamic quantization API](https://docs.pytorch.org/docs/main/generated/torch.ao.quantization.quantize_dynamic.html)
- [PyTorch quantization overview](https://docs.pytorch.org/docs/stable/quantization)
- [PyTorch CUDA Event timing](https://docs.pytorch.org/docs/stable/generated/torch.cuda.streams.Event.html)

License: MIT. No API keys, private data, or downloaded datasets are used.
