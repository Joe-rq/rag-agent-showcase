# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目性质

RAG 智能体工程案例: LangChain LCEL 链 + FAISS + Ragas 评估 + FastAPI 服务。全链路自实现(chunking / 检索 / 评估), 模型后端走 OpenAI 兼容接口可切换。简历用途。

## 常用命令

```bash
# 依赖(强制 uv, 禁止 pip)
uv sync

# 启动本地 embedding 后端(默认 42model; 多后端见下)
42model serve --port 11520 -y &

# 建索引(下载 5 篇论文 -> chunking -> FAISS)
uv run python -m rag_agent.indexing

# 端到端冒烟
uv run python verify.py

# 评估(baseline 朴素 RAG vs HyDE, Ragas 四指标)
uv run python -m rag_agent.eval --num 20

# 测试(全 mock, 离线, ~1.3s)
uv run pytest tests/ -q
uv run pytest tests/test_models.py::TestGetLLM -q   # 单测试类

# 启动服务(http://localhost:8000/playground)
uv run uvicorn app.main:app --port 8000
```

## 架构

**模型后端 provider 抽象**(关键): `config.yaml` 的 `llm.provider` / `embedding.provider` 三选(`42model` 本地 / `ollama` 本地 / `cloud` 云端), 全部走 OpenAI 兼容接口, 由 `models.py` 的 `get_llm` / `get_embedder` 按 provider 创建。切换后端只改 provider 字段; api_key 从 `.env` 读(`LLM_API_KEY` for cloud LLM, `EMBEDDING_CLOUD_API_KEY` for cloud embedding; 本地后端用占位串)。

**RAG 数据流**:
- `indexing.py`: arxiv id 构造 URL 下载 PDF(arxiv 4.x 移除了 `download_pdf`) -> pymupdf 解析(带页码) -> chunking(fixed / recursive / markdown) -> `get_embedder` -> FAISS 持久化
- `retrieval.py`: `make_retriever(advanced=none|hyde|rerank)`; HyDE 先让 LLM 生成假设文档再用其 embedding 检索
- `chain.py`: LCEL 链 `retriever | docs2str | prompt | llm | strip_think`
- `eval.py`: 合成 QA -> 跑 baseline/hyde -> Ragas 四指标 -> 对比表
- `app/main.py`: FastAPI(`/chat` `/health` `/playground`), 弃用已废弃的 LangServe

**配置**: `config.yaml`(结构化, 入库) + `.env`(api_key, gitignore)。`config.py` 读 yaml, `models.py` 读 env。支持 `RAG_CONFIG_PATH` 环境变量覆盖 config 路径(pip 安装后 `parents[2]` 不再指向仓库根)。

## 关键约定(非显然, 改动时务必注意)

- **本地后端 embedding 参数**: 42model / ollama 的 OpenAI 兼容代理**不接受 token-id 输入**且**单次 input 上限 256**, 故 `get_embedder` 对本地 provider 强制 `check_embedding_ctx_length=False` + `chunk_size=128`; 云端用默认长度检查。改 `models.py` 别破坏这套。
- **`<think>` 标签**: minicpm5 / deepseek-reasoner 等模型输出带 `<think>...</think>`, `chain.strip_think` 剥离(含未闭合的, 正则用 `\Z` 兜底)。RAG 链末尾必须挂 strip_think, 否则思考内容污染答案与评估。
- **ragas 0.4.3 硬 import 已移除模块**: ragas 0.4.3 顶层 `from langchain_community.chat_models.vertexai import ChatVertexAI` 在新版 langchain-community 已移除。`eval.py` 顶部用 `sys.modules` stub 掉 VertexAI(本项目用 ChatOpenAI, 不用 VertexAI)。升级 ragas / langchain-community 时注意这条会失效。
- **`chunk_overlap=0` 是合法值**: `indexing.build_index` 用 `is not None` 而非 `or` 判断, 否则 `chunk_overlap=0` 被 falsy 吞掉。
- **测试全 mock**: `tests/` 不调任何外部服务(DeepSeek / 42model / 网络)。用 monkeypatch 替换 `get_llm` / `get_embedder` / `ChatOpenAI` / `OpenAIEmbeddings` / urllib。新增测试保持这套。
- **FAISS docstore 引用**: retriever 返回的是 docstore 内对象引用, 改 metadata 要返回 `Document` 拷贝(见 `retrieval.py` HyDE 分支), 否则污染底层索引。

## 评估结论(已跑, 诚实记录)

20 QA 实测: HyDE 在本场景(论文细节问答, QA 合成致 query ≈ document)**无显著提升**(context_precision -0.033)。这是场景不匹配的结论, 非代码 bug。详见 `docs/evaluation-report.md`。改检索策略前先读它。

## Git

- commit message 用中文(项目规范)
