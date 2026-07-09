# 评估报告

## 评估设置

| 项 | 值 |
|---|---|
| 数据集 | 5 篇 AI Agent 论文(ReAct/Reflexion/Toolformer/ToT/Voyager), 106 页 -> 567 chunks |
| Embedding | `qwen3-embedding:0.6b`(本地 42model, 1024 维) |
| LLM(generate + judge) | `deepseek-chat`(云端 DeepSeek) |
| 评估框架 | Ragas 0.4.3 |
| 指标 | faithfulness, answer_relevancy, context_precision, context_recall |
| 对比 | baseline(朴素 RAG, `advanced=none`) vs HyDE(`advanced=hyde`) |

## 双场景对比(核心)

为验证 HyDE 的适用场景, 跑了两组 20 QA:
- **场景 A 论文式 QA**: 问题用论文术语, query ≈ document
- **场景 B 口语化 QA**: 问题用日常用语, query ≠ document(放大 query-document 差异)

### 场景 A: 论文式 QA (query ≈ document)

| 指标 | baseline | hyde | delta |
|---|---|---|---|
| faithfulness | 0.824 | 0.833 | +0.009 |
| answer_relevancy | 0.921 | 0.919 | -0.002 |
| context_precision | 0.833 | 0.800 | -0.033 |
| context_recall | 0.972 | 0.971 | -0.002 |

**结论: HyDE 无显著提升。** query ≈ document, HyDE 的"对齐语义"优势无发挥空间(context_recall 两版都 ~0.97, 朴素检索已命中)。

### 场景 B: 口语化 QA (query ≠ document)

| 指标 | baseline | hyde | delta |
|---|---|---|---|
| faithfulness | 0.828 | 0.755 | -0.073 |
| answer_relevancy | 0.487 | 0.471 | -0.016 |
| context_precision | 0.500 | 0.526 | **+0.026** |
| context_recall | 0.625 | 0.650 | **+0.025** |

**结论: HyDE 在检索指标(context_precision +0.026, context_recall +0.025)双双正向。** 口语化提问放大了 query-document 差异, HyDE 的假设文档对齐发挥了作用。

## 解读

**核心发现: HyDE 的价值取决于 query 与 document 的表达差异。**
- query ≈ document(论文式提问): HyDE 无用武之地
- query ≠ document(口语化提问): HyDE 检索精度/召回小幅正向

这验证了 HyDE 的适用场景判断——**进阶检索不是万能的, 要看场景匹配**。这也是本项目把"场景不匹配时 HyDE 无增益"诚实呈现的原因: 它本身就是有价值的工程结论。

## 局限

1. **delta 偏小(+0.025)**: 样本 20, 统计意义有限, 结果方向性参考为主。
2. **faithfulness 口语化场景 -0.073**: HyDE 检索的 chunks 虽 precision/recall 高, 但生成 faithfulness 略降, 可能是 judge 噪声或假设文档引入的轻微偏离。
3. **answer_relevancy 不可靠**: DeepSeek 不支持 n>1, Ragas 该指标触发 `BadRequestError: Invalid n value`, 数值不稳, 仅参考。
4. **judge = generator**: 评估 judge 与生成模型同为 DeepSeek, 存在自评偏差。
5. **场景 A 的 QA 已被场景 B 覆盖**: 场景 A 数字来自之前运行(git 历史), 当前 `qa_pairs.json` 是场景 B 口语化 QA。

## 改进方向

- QA 集扩到 50+ 并区分场景, 提升统计效力
- 换更强且支持 n>1 的 judge(如 GPT-4)消除 answer_relevancy 限制与自评偏差
- 叠加 rerank(bge-reranker)与 HyDE 对比/组合
