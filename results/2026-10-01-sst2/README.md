# 真实模型：DistilBERT / SST-2 部署验证

本次环境为 Ryzen 9 8945HX、RTX 5070 Laptop 8 GB、Windows 11、驱动 582.05、Python 3.12.6、PyTorch 2.10.0+cu128，CPU 固定 4 线程、TF32 关闭。配置、源码哈希和依赖版本保存在 [metadata.json](metadata.json)。

与原 `benchmark.py` 的 CUDA Event 计时不同，新实验使用逐请求同步墙钟计时，包含 Python/主机提交开销；因此不能将新旧绝对延迟直接当作同一指标。每轮候选顺序随机化，所有原始计时在 [samples.jsonl](samples.jsonl)，逐轮结果在 [results.csv](results.csv)，汇总在 [summary.csv](summary.csv)。汇总延迟是三轮中位数的中位数，P95 是三轮各自 P95 的最大值，误差/资源取最坏观测值；不是统计置信区间。

短操作仍存在噪声；随机顺序和重复测量降低了固定顺序带来的风险，但未控制系统调度、温度、频率和所有后台负载。样本 P95 不能当作生产 SLA 保证。显存指标是 PyTorch allocator 在已分配输入/模型等张量以上的单次操作峰值增量，包含输出/临时张量，不是整机或模型总显存；CPU 显存指标留空。

## 来源与边界

- 模型：[DistilBERT SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english/tree/714eb0fa89d2f80546fda750413ed43d93601a13)，固定 revision `714eb0fa89d2f80546fda750413ed43d93601a13`；通过 safetensors 加载，不执行远端自定义代码。
- 数据：[Stanford SST-2](https://huggingface.co/datasets/stanfordnlp/sst2/tree/8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb)，固定 revision `8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb`，全部 872 条 validation；文件 SHA-256 在 metadata 中。模型 Apache-2.0；数据卡标注许可 unknown。本仓库不重新分发模型权重或原始句子。
- 使用同一已训练模型的复制；只动态量化 Linear，embedding / LayerNorm 等保留浮点。所有变体固定 eager attention，避免后端差异混入精度比较。
- batch=1/8 分别完整评估一次验证集，因为动态激活量化可能受 batch 组成影响。准确率没有重复三次，也没有把计时样本当作独立质量样本。
- 固定 padding/truncation 到 128 tokens。请求计时包含 CPU 文本分词、输入传输、完整 forward、输出转 CPU FP32 logits；不含下载、加载、HTTP、排队或 softmax/标签后处理。
- 计时每配置 3 轮，预热 5 次、测量 20 次；各轮使用固定且不同的验证集片段，候选顺序随机化。完整验证集用于质量评价，计时只覆盖这些代表性请求，未覆盖真实流量分布。

## 实测结果

| Batch | 变体 | 请求中位数 ms | 最坏轮次 P95 ms | 验证准确率 | 下降百分点 | 改变预测数 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | cpu_fp32 | 40.121 | 41.072 | 91.06% (794/872) | 0.000 | 0 |
| 1 | cpu_int8_dynamic | 11.675 | 12.902 | 89.22% (778/872) | 1.835 | 38 |
| 1 | cuda_fp32 | 3.552 | 5.121 | 91.06% (794/872) | 0.000 | 0 |
| 1 | cuda_fp16 | 3.485 | 11.195 | 91.06% (794/872) | 0.000 | 0 |
| 8 | cpu_fp32 | 277.437 | 302.751 | 91.06% (794/872) | 0.000 | 0 |
| 8 | cpu_int8_dynamic | 82.357 | 95.935 | 90.25% (787/872) | 0.803 | 35 |
| 8 | cuda_fp32 | 14.711 | 16.068 | 91.06% (794/872) | 0.000 | 0 |
| 8 | cuda_fp16 | 5.187 | 9.375 | 91.06% (794/872) | 0.000 | 0 |

质量统计均以同 batch 的 CPU FP32 为基线；准确率和预测变化来自 [quality.csv](quality.csv)。[predictions.csv](predictions.csv) 保存样本 ID、真实标签、预测和两个 logits，可复核差异，但不包含原始文本。

| 变体 | state_dict MiB | 相对 CPU FP32 大小 |
| --- | --- | --- |
| CPU FP32 | 255.451 | 100% |
| CPU INT8（仅 Linear） | 132.288 | 51.79% |
| GPU FP16 | 127.746 | 50.01% |

## 应用结论

1. GPU FP16 在两个 batch 上都与 CPU/GPU FP32 的分类预测完全一致（均 794/872）。batch=8 的请求中位数从 GPU FP32 的 14.711 ms 降为 5.187 ms，约 2.84×。这提供了比随机张量误差更直接的质量证据，但不能保证别的数据集或模型也无损。
2. CPU INT8 将请求中位数降低约 3.4×，同时带来真实质量损失：batch=1 少答对 16 条，准确率下降 1.835 个百分点；batch=8 少答对 7 条，下降 0.803 个百分点。batch=1 有 27 条由对变错、11 条由错变对；batch=8 分别为 21/14。总准确率不能完全表达预测变化。
3. 整体压缩比不是单个 Linear 的四分之一。本次 INT8 state_dict 约为 FP32 的 52%，因为只有 Linear 被量化，其余参数仍为浮点。
4. 小 MLP 的 batch=1 选择偏向 CPU INT8，但完整模型的同 batch GPU FP16 约 3.485 ms、CPU INT8 约 11.675 ms。算子结论不能直接替代完整模型部署验证。
5. 在示例约束“最多下降 1 个百分点、state_dict ≤150 MiB、仅允许 CPU、logits 相对 L2 ≤0.25”下，batch=1 无可行候选，batch=8 选择 INT8。[CPU 策略](policy-cpu-150mib.json)
6. P95 会影响选择：增加观测 P95 ≤10 ms 后，batch=1 选 GPU FP32，batch=8 选 GPU FP16。[P95 策略](policy-p95-10ms.json) batch=1 FP16 的第三轮 P95 为 11.1955 ms，该波动未删除。样本量很小，应以此触发更长的服务压测，不能宣布已经满足生产 SLA。

这是已有 fine-tuned 模型的 validation 结果；未训练、未按本次验证集调参，不是新的独立测试集认证。量化质量结果只适用于本模型 revision、tokenizer、长度、batch 和引擎。

## 复现

```powershell
python -m pip install -r requirements-cu128.txt
python -m pip install -r requirements-model.txt
python evaluate_model.py --output results/my-sst2
python select_profile.py --run results/my-sst2 --output cpu-policy.json --only-options cpu_fp32 cpu_int8_dynamic --max-model-mib 150 --max-accuracy-drop-pp 1 --max-relative-error 0.25
```

模型/数据初次运行会下载到 Hugging Face 缓存。缓存完整后可用 `--local-files-only`；无 GPU 时用 `--cpu-only`。阈值是说明工具用法的示例，需要按应用实际质量要求制定。
