# 线性层部署边界：驻留计算与传输往返

本次环境为 Ryzen 9 8945HX、RTX 5070 Laptop 8 GB、Windows 11、驱动 582.05、Python 3.12.6、PyTorch 2.10.0+cu128，CPU 固定 4 线程、TF32 关闭。配置、源码哈希和依赖版本保存在 [metadata.json](metadata.json)。

与原 `benchmark.py` 的 CUDA Event 计时不同，新实验使用逐请求同步墙钟计时，包含 Python/主机提交开销；因此不能将新旧绝对延迟直接当作同一指标。每轮候选顺序随机化，所有原始计时在 [samples.jsonl](samples.jsonl)，逐轮结果在 [results.csv](results.csv)，汇总在 [summary.csv](summary.csv)。汇总延迟是三轮中位数的中位数，P95 是三轮各自 P95 的最大值，误差/资源取最坏观测值；不是统计置信区间。

短操作仍存在噪声；随机顺序和重复测量降低了固定顺序带来的风险，但未控制系统调度、温度、频率和所有后台负载。样本 P95 不能当作生产 SLA 保证。显存指标是 PyTorch allocator 在已分配输入/模型等张量以上的单次操作峰值增量，包含输出/临时张量，不是整机或模型总显存；CPU 显存指标留空。

## 范围与计时边界

仍使用随机权重的 `768 → 3072 → 768` MLP。3 轮分别使用种子 2026/2027/2028，batch=1/8/32/128/512，每配置预热 10 次、计时 30 次。共 40 个配置、120 条逐轮记录，全部成功。

- `resident`：输入/输出留在模型设备上，不含传输。选择器分别在 CPU、CUDA 内比较。
- `roundtrip`：统一从 CPU FP32 输入到 CPU FP32 输出，GPU 路径包含输入转换、H2D、forward、输出转换和 D2H，因此可以在这个边界内比较 CPU 与 GPU。
- 不包含网络、请求排队、模型加载或训练。`items_per_second` 是 batch/中位数延迟对应的串行速率，不是并发服务吞吐。

## 结果

以下为 CPU 输入/输出往返场景的三轮中位数汇总：

| Batch | CPU FP32 ms | CPU INT8 ms | GPU FP32 ms | GPU FP16 ms |
| --- | --- | --- | --- | --- |
| 1 | 0.2667 | 0.1264 | 0.1800 | 0.1531 |
| 8 | 0.4373 | 0.1894 | 0.1605 | 0.1494 |
| 32 | 1.2004 | 0.3358 | 0.2800 | 0.2030 |
| 128 | 3.4311 | 0.7290 | 0.3778 | 0.2813 |
| 512 | 13.1544 | 2.6963 | 0.9203 | 0.5089 |

1. batch=1 时 CPU INT8 为 0.1264 ms，GPU FP16 为 0.1531 ms；而 GPU FP16 驻留计算为 0.0945 ms。把数据搬运算进去会改变这个小算子的部署选择。
2. batch=8/32/128/512 的往返场景中，GPU FP16 的汇总中位数最低。batch=512 的 GPU FP16 从驻留计算 0.1621 ms 变为往返 0.5089 ms，说明数据边界不能忽略。
3. 误差需要覆盖多个输入/权重种子。batch=512 的 INT8 最坏相对 L2 误差约 0.03243，超过 0.03 的示例阈值，即使它比 CPU FP32 快，选择器仍会拒绝它。
4. 限制权重文件 ≤5 MiB、误差 ≤0.03 后，仅 15 个工作负载组中的 8 个有可行选项；batch=512 和 GPU 驻留组会明确返回无解。[示例策略](policy-5mib.json)

这些数字只描述 MLP。完整模型的实际准确率与速度另见 [DistilBERT 验证](../2026-10-01-sst2/README.md)，不能用本实验的约 3% 输出误差推断模型质量。

## 复现

```powershell
python profile_workloads.py --output results/my-workloads
python select_profile.py --run results/my-workloads --output policy.json --max-relative-error 0.03 --max-model-mib 5
```

仅用 CPU 可传 `--cpu-only`，更多说明见 [APPLICATIONS.md](../../APPLICATIONS.md)。历史 CUDA Event 基准保留不变。
