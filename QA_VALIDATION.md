# 从分类模型到中文问答的量化验收

与配套的 attention-kernel-lab 一起搭建了本地文档问答应用：HTTP 请求 → BM25 检索 → Qwen2.5-1.5B-Instruct → KV cache 生成 → 流式文本返回。模型 revision 固定为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，无需 AI API。应用与原始记录集中保存在一个仓库，避免两份实现发生偏离。

## 本次量化验收的结论

CPU FP32 能正确回答“专业版每月 99 元、200GB”。下面三种动态量化配置在同一价格题、短长两种资料中均跑题：

| 配置 | 保持 FP32 的部分 | 结果 |
|---|---|---|
| 全 Linear 动态 INT8 | embedding、norm 等非 Linear | 出现编程求助或无关选择题 |
| 仅 MLP Linear 动态 INT8 | 注意力投影、输出头、embedding、norm | 出现翻译题、团队概念等无关回答 |
| 仅 MLP、逐输出通道 INT8 权重 | 同上一行 | 仍出现代码及无关内容 |

环境为 PyTorch 2.10.0+cu128，Windows CPU quantization backend 为 oneDNN。量化设置属于 `torch.ao.quantization.quantize_dynamic`，不是专门为该模型校准的 GPTQ、AWQ 等方案。

**这些方案没有通过应用质量检查，不能把它们的计时缩短作为可部署的加速收益。** 这里仅有一道预检问题，不足以估计通用准确率，但足以拒绝在该演示中启用这些已出现明显错误的配置。后续正式多题测试使用通过基础预检的 CPU FP32 和三个 GPU BF16 配置；仍保留无答案问题的错误，详见应用报告。

本结果没有推翻本仓库先前的 DistilBERT/SST-2 数据：CPU INT8 在 batch=1 和 batch=8 时的准确率损失，仍分别是约 1.83 和 0.80 个百分点。两种任务和模型不同，应分别验收。生成式模型一次输出包含连续多步预测，不能只拿随机 Linear 的误差或另一模型的分类准确率作部署依据。

## 代码、原始答案与复现

2026-10-02 新增[数值诊断与整模型消融](results/2026-10-02-qa-quant-ablation/README.md)：同一价格题中，两种仅舍入权重的 FP32 控制都保持正确，实际 dynamic INT8 跑题。真实激活回放显示动态量化路径的偏差明显高于权重舍入，并随 prefill/逐 token 调用范围变化。这为后续改进提供了方向，但尚未产出通过质量验收的 INT8 问答部署方案。

- [可运行应用和测试方法](https://github.com/Yhg1123/attention-kernel-lab/blob/main/QA_TESTING.md)
- [量化及 FP16 失败的原始预检记录](https://github.com/Yhg1123/attention-kernel-lab/tree/main/results/2026-10-01-qa-preflight)
- [完整 HTTP 应用测试报告](https://github.com/Yhg1123/attention-kernel-lab/blob/main/results/2026-10-01-qa-http-worker/README.md)

在配套仓库安装设备对应的 PyTorch 与 `requirements-qa.txt`，再运行：

```powershell
python qa_benchmark.py --output results/my-int8-qa-check --trials 1 --question-ids price --max-new-tokens 64 --fixed-tokens 32 --variants cpu_eager_fp32 cpu_eager_int8 cpu_eager_int8_mlp cpu_eager_int8_mlp_per_channel --local-files-only
```

第一次尚未下载模型时去掉 `--local-files-only`。自然回答用于核对答案；固定输出控制只用于速度分析，不计入正确性。若要验证自己的业务，应替换文档和验收问题，且需要检查完整回答内容与无答案场景。
