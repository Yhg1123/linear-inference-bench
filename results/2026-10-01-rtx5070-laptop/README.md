# 2026-10-01 · RTX 5070 Laptop 复现实验

## 环境与方法

- CPU：AMD Ryzen 9 8945HX，16 核 / 32 线程；系统内存约 32 GB。
- GPU：NVIDIA GeForce RTX 5070 Laptop GPU，驱动报告 8151 MiB，计算能力 12.0；驱动 582.05，Windows 11（26200）。
- Python 3.12.6；PyTorch 2.10.0+cu128；CUDA runtime 12.8；cuDNN 9.10.2。
- 测试开始时接通电源，Windows 平衡电源方案；未修改系统电源设置或锁定 GPU 时钟。开始/结束的 GPU 状态见 [environment.json](environment.json)，不是持续监控或恒定功耗保证。
- 沿用原实验的输入规模、随机种子 2026、预热 50 次、每配置计时 100 次。三个独立进程依次运行，两个仓库的基准不并发。
- 第一轮是本目录 [results.csv](results.csv)，后两轮在 [repeat-02/results.csv](repeat-02/results.csv)、[repeat-03/results.csv](repeat-03/results.csv)。表格标明首轮或三轮汇总，不挑选最好的一轮。
- [summary.csv](summary.csv) 的 `median_of_medians_ms` 是三轮各自中位数的中位数；不是把 300 个样本合并后的中位数。原脚本不导出单次计时样本。

原 RTX 3050 数据仍在 [../results.csv](../results.csv)，对应 PyTorch 2.5.1+cu121。这里的硬件、驱动、PyTorch、CUDA 版本同时变化，跨机器差异不能解释为纯硬件提升。PyTorch 从 2.7 开始提供 Blackwell / CUDA 12.8 支持，安装版本参考 [官方发行说明](https://pytorch.org/blog/pytorch-2-7/) 和 [官方历史版本页](https://pytorch.org/get-started/previous-versions/)。

## 兼容性修复

原脚本在本机启动失败：`torch.backends.quantized.supported_engines == ['onednn']`，而选择器只检查 `x86/fbgemm/qnnpack`。原始失败日志保存在 [initial-attempt/benchmark.log](initial-attempt/benchmark.log)。新增 `onednn` 回退并保留原有优先级；模型、输入、误差定义和计时逻辑不变。当前运行实际使用 `onednn`，原机器使用 `x86`。新增回归测试覆盖单一 oneDNN、原有优先级及无可用引擎场景；CPU CI 增加 PyTorch 2.5.1 / 2.10.0 矩阵。

## 首轮结果

| Batch | CPU FP32 ms | CPU INT8 ms | CPU 加速 | CUDA FP32 ms | CUDA FP16 ms | CUDA 加速 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.6133 | 0.1314 | 4.67× | 0.0875 | 0.0968 | 0.90× |
| 8 | 0.6833 | 0.1972 | 3.47× | 0.0682 | 0.0777 | 0.88× |
| 32 | 1.2353 | 0.3086 | 4.00× | 0.1410 | 0.0810 | 1.74× |
| 128 | 3.6424 | 0.7947 | 4.58× | 0.1934 | 0.0851 | 2.27× |

| 变体 | state_dict 大小 MiB | 首轮输出相对 L2 误差范围 |
| --- | --- | --- |
| cpu_fp32 | 18.017 | 0.000000–0.000000 |
| cpu_int8_dynamic | 4.518 | 0.021041–0.028381 |
| cuda_fp32 | 18.017 | 0.000000–0.000000 |
| cuda_fp16 | 9.01 | 0.000544–0.000817 |

## 三轮变化

| Batch | 变体 | 三轮延迟范围 ms | 对同轮同设备 FP32 加速范围 |
| --- | --- | --- | --- |
| 1 | cpu_int8_dynamic | 0.1236–0.1343 | 2.09–4.67× |
| 1 | cuda_fp16 | 0.0968–0.1123 | 0.88–1.01× |
| 8 | cpu_int8_dynamic | 0.1972–0.2110 | 2.23–3.47× |
| 8 | cuda_fp16 | 0.0777–0.2269 | 0.34–0.88× |
| 32 | cpu_int8_dynamic | 0.3020–0.3150 | 3.70–4.00× |
| 32 | cuda_fp16 | 0.0810–0.0849 | 1.66–1.74× |
| 128 | cpu_int8_dynamic | 0.7374–0.7947 | 4.47–4.58× |
| 128 | cuda_fp16 | 0.0851–0.0873 | 1.98–2.27× |

CPU INT8 在所有 batch、所有轮次均更快；CUDA FP16 在 batch 32、128 均更快，但 batch 1 没有稳定收益，batch 8 三轮均慢于 FP32。第三轮 batch 8 的 FP16 为 0.2269 ms，该波动原样保留。短操作受主机提交、WDDM 调度和动态频率影响，不从一次测量推断稳定性能。

## 与原机器并列查看

以下都是首轮绝对延迟；CPU 和 GPU 分别比较，CPU 量化引擎从 x86 变为 oneDNN，软件和硬件变化共同影响结果。

| Batch | 5600H INT8 ms | 8945HX INT8 ms | 3050 FP16 ms | 5070 FP16 ms |
| --- | --- | --- | --- | --- |
| 1 | 0.2254 | 0.1314 | 0.1321 | 0.0968 |
| 8 | 0.3982 | 0.1972 | 0.1444 | 0.0777 |
| 32 | 0.6521 | 0.3086 | 0.1382 | 0.0810 |
| 128 | 2.3511 | 0.7947 | 0.1884 | 0.0851 |

## 验证

- 4 项单元测试通过，包括真实动态量化类型/输出检查与引擎选择回归测试。
- CPU-only smoke 的 2 个配置运行成功。
- 三轮各 16 行，配置网格完整，无重复；时间/误差有限且非负，模型大小大于零；加速比仅相对同设备同 batch 的 FP32 复核。
- INT8 相对 L2 误差低于 0.05，CUDA FP16 低于 0.002。原权重未训练，误差不代表任务准确率。
- 日志保留 PyTorch 的可选 NumPy 缺失提示与旧量化 API 弃用提示；本实验未使用 NumPy，动态量化实际运行与校验均通过。为保留实验定义，未迁移量化 API。

## 复现命令

在仓库根目录运行，使用 Python 3.12.6。`requirements.txt` 保留原机器的版本；本次使用独立的 `requirements-cu128.txt`。如果需要复刻全部 Python 依赖，使用本目录 [requirements-lock.txt](requirements-lock.txt)。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cu128.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
$output = 'results/local-rtx5070'
.\.venv\Scripts\python.exe benchmark.py --output $output
.\.venv\Scripts\python.exe benchmark.py --output "$output/repeat-02"
.\.venv\Scripts\python.exe benchmark.py --output "$output/repeat-03"
.\.venv\Scripts\python.exe recommend.py --results "$output/results.csv" --max-relative-error 0.03
```

命令和 UTC 时间戳见 [execution.json](execution.json)；硬件、软件和运行源码 SHA-256 见 [environment.json](environment.json)；每轮 `metadata.json`、`benchmark.log`、`validation.log` 保留实际设置、输出和检查结果。首轮选择器输出在 [recommendations.csv](recommendations.csv)。历史 `results/results.csv` 保持不变，选择器默认仍读取历史数据，查看本次结果必须显式传入 `--results`。
