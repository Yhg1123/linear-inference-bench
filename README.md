# Linear Inference Bench | 线性层量化推理实验

An inference precision and deployment tradeoff study for a Transformer style feedforward block. It measures **latency, serialized model size, and output error** for CPU FP32, CPU dynamic INT8, GPU FP32, and GPU FP16. A small selector uses measured data to choose a variant under explicit quality and size constraints.

本项目研究部署时的精度选择，而不只是汇报单个速度数字。CPU 与 GPU 各自使用 FP32 对照；INT8 与 FP16 的误差使用同设备 FP32 输出计算，避免跨设备加速比造成误导。

## 应用化扩展（2026-10-01）

2026-10-03 新增[线程数 × 批量大小复测](results/2026-10-03-thread-sweep/INTERPRETATION.md)：128 个配置、5 个种子、32,000 个单次计时样本，计入 CPU/GPU 传输。数据支持小请求按线程和设备实测选择，也显示增加到 16 个 CPU 线程不一定比 8 个更快；完整范围、P95、误差和源码均保存。

新增包含 CPU/GPU 传输的请求级 profile、P95/多种子约束选择，以及固定版本 DistilBERT 在全部 872 条 SST-2 验证样本上的真实质量评估。

小 MLP 的 batch=1 计入传输后 CPU INT8 更快；完整 DistilBERT 则 GPU FP16 更快。FP16 在本次验证集上与 FP32 的预测完全一致；CPU INT8 在 batch=1/8 分别少答对 **16/7 条**，实际模型权重文件仅缩小到约 **52%**。因此，单层的加速、误差和压缩比不能直接代表完整模型。

[应用方式与选择器](APPLICATIONS.md) · [传输边界实验](results/2026-10-01-workloads/README.md) · [真实模型验证](results/2026-10-01-sst2/README.md)

新增[中文文档问答验收](QA_VALIDATION.md)：同机运行公开 Qwen 模型，并检查实际生成内容。全 Linear INT8、仅 MLP INT8、MLP 逐通道 INT8 均未通过简单价格题，保留原始失败回答；这说明分类任务中的量化收益不能直接当作生成式问答的部署结论。

## 核心问题与方法

2026-10-02：[为什么 Qwen 动态 INT8 会跑题？](results/2026-10-02-qa-quant-ablation/README.md) 新增真实激活回放、28 层误差追踪、两种权重舍入整模型控制及可复现诊断脚本。结果将排查重点从权重舍入缩小到动态激活量化和执行路径，保留完整失败答案与源码快照。

模型结构：`Linear(768, 3072) → GELU → Linear(3072, 768)`，与 Transformer 前馈层形状相近。随机种子 2026，权重未训练，输入为随机张量。测试 batch 为 1、8、32、128。

| 变体 | 实现 | 对照 |
| --- | --- | --- |
| CPU FP32 | 原始 PyTorch `nn.Linear` | CPU 基线 |
| CPU INT8 dynamic | `torch.ao.quantization.quantize_dynamic`，仅量化 Linear | CPU FP32 |
| CUDA FP32 | 同一 FP32 权重的 CUDA 复制 | CUDA 基线 |
| CUDA FP16 | 同一权重与输入转换为 FP16 | CUDA FP32 |

每项预热 50 次，计时 100 次，报告单次 forward 的中位数和标准差。CPU 使用 4 线程及墙钟时间；GPU 使用 CUDA Event。模型大小是序列化 `state_dict` 的字节数，**不是运行时显存**。还计算最大绝对误差、相对 L2 误差与余弦相似度。

## 原始 RTX 3050 实测摘要

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

## RTX 5070 Laptop 复测（2026-10-01）

新增环境：Ryzen 9 8945HX、RTX 5070 Laptop 8 GB、驱动 582.05、Python 3.12.6、PyTorch 2.10.0+cu128。每个配置仍预热 50 次、计时 100 次，独立运行三轮。原始 3050 文件保留不变。

三轮各 16 个配置均运行成功。补充 oneDNN 量化引擎回退，修复当前 Windows PyTorch 构建无法启动的问题。首轮 CPU INT8 加速 3.47–4.67×；CUDA FP16 在 batch 128 为 0.0851 ms、加速 2.27×，但小 batch 没有稳定收益。4 项单元测试及 CPU smoke 通过。

| Batch | CPU FP32 ms | CPU INT8 ms | CPU 加速 | CUDA FP32 ms | CUDA FP16 ms | CUDA 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.6133 | 0.1314 | 4.67× | 0.0875 | 0.0968 | 0.90× |
| 8 | 0.6833 | 0.1972 | 3.47× | 0.0682 | 0.0777 | 0.88× |
| 32 | 1.2353 | 0.3086 | 4.00× | 0.1410 | 0.0810 | 1.74× |
| 128 | 3.6424 | 0.7947 | 4.58× | 0.1934 | 0.0851 | 2.27× |

上表为首轮数据。硬件、驱动和软件版本同时改变，不能把跨机器差异当作纯硬件升级收益。后端排名及短操作延迟有波动，详见 [完整复测报告、三轮范围与复现命令](results/2026-10-01-rtx5070-laptop/README.md)。原 `requirements.txt` 对应旧环境；RTX 50 系列使用 `requirements-cu128.txt`，选择器读取新结果时传 `--results results/2026-10-01-rtx5070-laptop/results.csv`。

## 可复现运行（原 RTX 3050 环境）

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

Code license: MIT. Core microbenchmarks use random tensors. Optional model validation downloads pinned public model/data assets to the Hugging Face cache; weights and source sentences are not committed. Source licenses and revisions are recorded in the model report.
