# RAG Agent Showcase

基于 LangChain 的检索增强生成(RAG)智能体 -- 全链路自实现的独立工程案例: 数据接入、chunking、向量检索、HyDE 进阶检索、Ragas 四指标评估、FastAPI 服务化。

## 关键设计决策

| 维度 | 选择 | 理由 |
|---|---|---|
| 模型后端 | Embedding 本地 42model + LLM 云端 DeepSeek | embedding 本地零计费, LLM 云端快稳, 两端 OpenAI 兼容可切换 |
| 数据集 | 5 篇 AI Agent 论文固定冻结(ReAct/Reflexion/Toolformer/ToT/Voyager) | 可复现、可对比 |
| Chunking | recursive(800/120), 保留 fixed/markdown 变体对比 | 多级分隔符保语义, 评估驱动选型 |
| 检索 | HyDE 假设文档嵌入 | 对齐 query-document 语义, 提升检索精度 |
| 评估 | Ragas 四指标 + 20 QA + baseline 对比 | 专业化, 量化 HyDE 增益 |
| 服务 | FastAPI(弃用已废弃的 LangServe) | 跟进技术趋势, 路由透明 |

## 架构

```
[论文 PDF] -> pymupdf 解析 -> chunking -> [qwen3-embedding] -> FAISS 索引
                                                            |
问 question -> [HyDE: LLM 生成假设文档] -> 检索 top-k -> 拼上下文 -> [DeepSeek] -> answer
```

## 技术栈

- **LangChain 0.3** -- LCEL 链式编排
- **FAISS** -- 向量检索
- **Ragas** -- RAG 评估
- **42model** -- 本地 embedding 后端(OpenAI 兼容, 离线零计费)
- **DeepSeek** -- 云端 LLM(OpenAI 兼容)
- **FastAPI** -- 服务化
- **uv** -- 依赖管理

## 快速开始

前置: 已装 [42model](https://42model.com/)(`42model abilities` 显示有 `qwen3-embedding:0.6b`),且有云端 LLM API key(默认 DeepSeek)。

```bash
# 1. 启动 42model 本地 embedding 后端(后台常驻)
42model serve --port 11520 -y &

# 2. 安装依赖
uv sync

# 3. 配置(填 LLM_API_KEY, 默认 DeepSeek)
cp .env.example .env

# 4. 构建索引(首次会下载 5 篇论文)
uv run python -m rag_agent.indexing

# 5. 冒烟测试(端到端跑通)
uv run python verify.py

# 6. 评估
uv run python -m rag_agent.eval

# 7. 启动服务
uv run uvicorn app.main:app --port 8000
```

## 项目结构

```
rag-agent-showcase/
├── src/rag_agent/
│   ├── config.py      # 配置加载(yaml + env)
│   ├── models.py      # 模型工厂(LLM / embedder)
│   ├── indexing.py    # 数据加载 + chunking + FAISS
│   ├── retrieval.py   # 检索 + HyDE
│   ├── chain.py       # RAG LCEL 链
│   └── eval.py        # Ragas 评估
├── app/main.py        # FastAPI 服务
├── data/              # 论文 PDF + FAISS 索引(gitignore)
├── evals/             # QA 集 + 评估结果
├── docs/
│   ├── adr/           # 选型决策记录
│   └── evaluation-report.md
├── config.yaml        # 全配置
└── pyproject.toml
```

## 评估结果

20 QA 规模, baseline(朴素 RAG) vs HyDE, Ragas 四指标:

| 指标 | baseline | hyde | delta |
|---|---|---|---|
| faithfulness | 0.824 | 0.833 | +0.009 |
| answer_relevancy | 0.921 | 0.919 | -0.002 |
| context_precision | 0.833 | 0.800 | -0.033 |
| context_recall | 0.972 | 0.971 | -0.002 |

**结论: 本场景下 HyDE 无显著提升。** 原因: QA 从 chunks 合成, query ≈ document, HyDE 的"对齐 query-document 语义"优势无发挥空间(context_recall 两版均 0.97)。这反而是个诚实的工程结论--进阶检索不是万能的, 要看场景匹配。详见 [docs/evaluation-report.md](docs/evaluation-report.md)。

## 已知局限

1. **judge = generator**: 评估 judge 与生成模型同为 DeepSeek, 存在自评偏差。更严谨的做法是用更强且不同的模型(如 GPT-4)做 judge。
2. **QA 自动合成**: 20 个 QA 由 LLM 合成, 质量受 LLM 限制, 可能偏简单/偏难, 影响 context_precision/recall 区分度。
3. **LLM 依赖云端 key**: DeepSeek 需 API key, 非"完全本地"。embedding 已本地化(42model)。
