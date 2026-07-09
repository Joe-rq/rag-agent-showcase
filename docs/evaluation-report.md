# 评估报告

## 评估设置

| 项 | 值 |
|---|---|
| 数据集 | 5 篇 AI Agent 论文(ReAct/Reflexion/Toolformer/ToT/Voyager), 106 页 -> 567 chunks |
| Embedding | `qwen3-embedding:0.6b`(本地 42model, 1024 维) |
| LLM(generate + judge) | `deepseek-chat`(云端 DeepSeek) |
| 评估框架 | Ragas 0.4.3 |
| 指标 | faithfulness, answer_relevancy, context_precision, context_recall |
| QA 集 | 20 个(LLM 合成) |
| 对比 | baseline(朴素 RAG, `advanced=none`) vs 改进(HyDE, `advanced=hyde`) |

## 已知局限
1. **judge = generator**: 评估 judge 与生成模型同为 DeepSeek, 存在自评偏差(模型倾向给自己生成的答案打高分)。更严谨的做法是用更强且不同的模型(如 GPT-4)做 judge。
2. **QA 自动合成**: 20 个 QA 由 LLM 从 chunks 合成, 问题和参考答案同源, 导致 query ≈ document(见下文解读), 压缩了 HyDE 的发挥空间。
3. **QA 规模 20**: 统计意义有限, 指标波动大, 结果方向性参考为主。

## 结果

| 指标 | baseline(朴素) | hyde(改进) | delta |
|---|---|---|---|
| faithfulness | 0.824 | 0.833 | +0.009 |
| answer_relevancy | 0.921 | 0.919 | -0.002 |
| context_precision | 0.833 | 0.800 | -0.033 |
| context_recall | 0.972 | 0.971 | -0.002 |

## 解读

### 核心结论

**在本数据集与 QA 集上, HyDE 相比朴素 RAG 无显著提升**(4 指标 delta 均在 ±0.03 内, context_precision 甚至略降 -0.033)。

这与 5 QA 小规模预览的结论(HyDE 全面提升, context_precision +0.15)相反, 说明 5 QA 是小样本偶然, 20 QA 更可信。

### 为什么 HyDE 在本场景没提升?

1. **QA 合成致 query ≈ document**: QA 由 LLM 从 chunks 合成, 问题和参考答案都直接来自同一 chunk, query 与 document 的表达差异本就很小。HyDE 的核心价值是"把口语化/简短 query 对齐到 document 语义空间", 但本场景 query 已经很接近 document, HyDE 无用武之地(context_recall 两版都 0.97, 说明朴素检索已能命中)。

2. **HyDE 假设文档可能引入噪声**: HyDE 让 LLM 生成假设答案再检索, 若假设答案偏离真实文档(DeepSeek 对论文细节的假设可能不准), 反而检索到错误 chunk, 拉低 context_precision。

3. **judge = generator 偏差**: 评估 judge 与生成模型同为 DeepSeek, 分数普遍偏高(0.82-0.97), 区分度低, 小幅差异可能被偏差淹没。

### HyDE 什么时候才有用?

HyDE 的优势在 **query 与 document 表达差异大** 的场景:
- 口语化提问 vs 学术文档
- 短问句 vs 长论述
- 跨语言/跨领域检索

本场景(论文细节问答, 问题≈文档)不属于此, 故 HyDE 无增益。这说明"进阶检索不是万能的, 要看场景匹配"--这本身就是有价值的工程判断。

### 下一步改进方向

- **QA 集改用真实用户提问风格**(口语化/简短), 放大 query-document 差异, 再验 HyDE
- **换更强且不同的 judge 模型**(如 GPT-4), 提升评估区分度
- **叠加 rerank**(bge-reranker)而非 HyDE, 或 HyDE + rerank 组合
- **扩大 QA 规模到 50+**, 提升统计效力
