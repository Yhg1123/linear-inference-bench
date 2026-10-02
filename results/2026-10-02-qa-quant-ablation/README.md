# Qwen 问答跑题：权重舍入与动态 INT8 的对照

同一台 Windows / Ryzen 9 8945HX，PyTorch 2.10.0+cu128、oneDNN、CPU 4 线程。模型和 revision 与前一天问答实验一致。输入为原价格题的 **403 个 token**，SHA-256 与历史 HTTP 请求完全一致；没有加入标准答案。只量化 28 层 MLP 的 84 个 Linear，注意力、输出头、embedding 保持 FP32。

## 整模型结果

| 计算方式 | 首次完整前向 logits 相对 L2 误差 | 贪心生成结果 |
|---|---:|---|
| 原始 FP32 | 对照 | 99 元、200GB |
| 逐张量 INT8 舍入权重，反量化后用 FP32 计算 | 9.416% | 与对照生成 token 完全相同 |
| 逐通道 INT8 舍入权重，反量化后用 FP32 计算 | 1.725% | 与对照生成 token 完全相同 |
| 原生逐张量动态 INT8 | 70.919% | 跑题，开始回答翻译选择题 |

权重舍入控制是诊断用途，**依然执行 FP32 矩阵乘法，没有证明 INT8 加速**。每次控制结束都会恢复原参数，再执行下一组。前后 embedding 输出逐位一致；实际转换的模块名称完整记录在 metadata 中。

## 相同真实激活的独立 Linear 回放

从 FP32 前向捕获第 0、13、27 层 up/down projection 的真实输入，每个输入分别走原始 FP32、反量化权重 FP32、原生 dynamic INT8。另将同一 prefill 输入切成逐 token 的独立 dynamic INT8 调用，观察量化作用范围的影响。这里的 `last_token` 是原 FP32 prefill 的最后一个位置，**不等同于真实自回归 decode 轨迹**。

以下为逐通道权重、完整 prefill 的输出相对 L2 误差：

| down projection | 只舍入权重 | 原生动态 INT8 | 每 token 单独调用动态 INT8 |
|---|---:|---:|---:|
| 第 0 层 | 0.853% | 36.183% | 11.590% |
| 第 13 层 | 1.120% | 26.067% | 7.217% |
| 第 27 层 | 0.934% | 24.321% | 8.169% |

这支持“动态激活量化及对应执行路径是重要误差来源”的解释，而非把全部问题归因于权重舍入。逐 token 调用明显减小这些回放误差，但仍高于只舍入权重；它未经过整模型准确率或速度验收，不能直接推荐用于部署。不同相对误差使用不同参照，不能相加来分解总误差。

第一个 decoder 层的最后位置隐藏状态已经产生 28.51% 相对 L2 偏差，最后一层为 69.30%；中间不单调。最终 logits 偏差出现在相同输入的前向计算中，因此不是先生成了不同文字才造成的表面差异。

**应用价值：** 后续优化应优先验证适合该模型的激活处理或权重专用量化方案，并先通过真实问答验收。继续只调权重逐通道设置，不能解决这里观察到的主要误差。本实验没有证明 oneDNN 存在缺陷，也没有比较其他后端、量化框架或模型。

## 原始证据与复现

- `linear_replay.csv`：24 行回放；`layer_drift.csv`：28 层漂移。
- `weight_only_controls.json`：两个整模型权重控制的完整答案及 token。
- `outcome.json`：FP32/动态 INT8 的完整输出、token 和前向误差。
- `metadata.json` 与 `source/`：固定模型版本、环境、输入哈希、84 个转换模块、实际测量源码快照。
- `outcome.json` 中 `*_first_token` 指 `use_cache=False` 前向的 argmax；`*_output_ids[0]` 才是实际 `generate(use_cache=True)` 的首 token。动态 INT8 中二者不同，不能混用。

安装本仓库 PyTorch 依赖及 `requirements-model.txt`，先缓存指定模型版本，再运行：

```powershell
python diagnose_qa_quantization.py --output results/my-qa-quant-ablation --threads 4
python -m unittest discover -s tests -v
```

输出目录必须不存在。原始探索运行保留在 [qa-quant-diagnosis](../2026-10-02-qa-quant-diagnosis/README.md)。这是**一个已知失败问题的定向诊断**，不是通用正确率评测；未测延迟或部署压缩收益。
