# 从精度实验到部署选择

这个仓库可以回答：**给定设备、batch、数据传输边界、模型大小、误差和实际准确率要求，哪些部署方式可行？** 它包含三个层次：历史单算子基准、带传输边界的 MLP profile、公开预训练模型的实际任务验证。

进一步补充了[中文生成式问答的应用验收](QA_VALIDATION.md)。在同一台机器上，Qwen2.5-1.5B 的简单动态 INT8 方案没有通过价格题验收；这说明 DistilBERT 分类实验中可接受的量化方案，不能直接推广到任意问答模型。

## 1. 先测数据边界

```powershell
python profile_workloads.py --batch-sizes 1 8 32 128 512 --trials 3 --output results/my-workloads
python select_profile.py --run results/my-workloads --output policy.json --max-relative-error 0.03 --max-model-mib 5
```

`resident` 只在同设备内比较；`roundtrip` 统一从 CPU FP32 输入到 CPU FP32 输出，允许在这个边界内比较 CPU/GPU。默认记录 3 个种子、随机候选顺序、原始请求计时、最坏误差/P95/资源和无解原因。[本机报告](results/2026-10-01-workloads/README.md)

## 2. 验证实际模型质量

激活已安装 CPU torch 或 CUDA torch 的 Python 环境，然后运行：

```powershell
python -m pip install -r requirements-model.txt
python evaluate_model.py --output results/my-sst2
python select_profile.py --run results/my-sst2 --output cpu-policy.json --only-options cpu_fp32 cpu_int8_dynamic --max-model-mib 150 --max-accuracy-drop-pp 1 --max-relative-error 0.25
```

模型和数据 revision 固定，下载保存在 Hugging Face 缓存。全量 872 条验证样本的结果在本机已跑通。参数 `--limit` 可做小样本调试，但小样本准确率不能冒充全量结果；`--local-files-only` 用已有缓存；`--cpu-only` 禁用 CUDA。[真实模型报告](results/2026-10-01-sst2/README.md)

## 3. 解释与消费策略

JSON 中每个精确工作负载都有 `selected` 或 `no_match`；还保留可行候选及拒绝原因。可选约束包括：`--max-relative-error`、`--max-p95-ms`、`--max-model-mib`、`--max-extra-mib`、`--max-spread-ratio`、`--max-accuracy-drop-pp` 和 `--only-options`。

质量降幅单位是**百分点**，不是相对百分比。未测实际准确率的随机 MLP 数据不能用于满足准确率约束；未测 CPU 显存不能用于满足显存约束。NaN、负数、失败或缺失轮次都不会成为候选。

策略是带版本和数据摘要哈希的实测建议，不会自动搬移业务模型或对未测设备/形状外推。模型文件大小不同于运行时 RAM/VRAM；随机层的输出误差不同于任务准确率；小样本 P95 不保证服务 SLA。阈值示例用于展示决策边界，并非通用部署标准。
