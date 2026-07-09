# ADR-0003: 弃用 LangServe, 用裸 FastAPI

## 上下文
RAG 服务化需选 HTTP 框架。LangServe 已被 LangChain 官方废弃, 推荐迁移到 LangGraph Platform 或直接用 FastAPI。

## 决策
用裸 FastAPI + Pydantic 模型, 手写 `/chat` `/health` `/playground` 路由。

## 理由
- LangServe 已废弃, 继续依赖有维护与安全风险
- 本项目路由简单, FastAPI 足够, 不需要 LangServe 的 Runnable 自动序列化包装
- 直接 FastAPI 更透明、可控, 也体现"跟得上技术趋势"的判断

## 后果
- **正面**: 无废弃依赖, 路由清晰, 易测试
- **负面**: 失去 LangServe 的 Runnable 自动暴露(本项目不需要, 反而减少黑盒)
