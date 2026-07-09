"""模型工厂: 按 provider 创建 LLM / embedder。

支持三种后端(均走 OpenAI 兼容接口):
  - 42model: 本地 42model serve (默认 embedding)
  - ollama:  本地 ollama (需 ollama serve + 已 pull 模型)
  - cloud:   云端 OpenAI 兼容 (LLM 默认 DeepSeek; embedding 可选 SiliconFlow 等)

切换只改 config.yaml 的 provider 字段; api_key 从 .env 读。

注:
  - cloud provider 的 api_key 缺失时此处不 fail-fast, 留待 openai SDK 在 invoke
    时抛鉴权错。改进点: 可在构造时提前校验并给出更明确提示。
  - config 的 embedding.dimensions 仅用于显示/校验, 不强制截断——本地后端
    (42model/ollama)不支持 OpenAI 的 dimensions 参数, 维度由模型原生决定。
"""
from __future__ import annotations

import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import cfg

# 各 provider 的 api_key 获取方式: 本地后端用固定占位(不校验), 云端从环境变量读
_LLM_API_KEY = {
    "cloud": lambda: os.environ.get("LLM_API_KEY", ""),
    "ollama": lambda: "ollama",
    "42model": lambda: "42model",
}
_EMB_API_KEY = {
    "42model": lambda: "42model",
    "ollama": lambda: "ollama",
    "cloud": lambda: os.environ.get("EMBEDDING_CLOUD_API_KEY", ""),
}

# 本地后端(OpenAI 兼容代理): 不接受 token-id 输入且单次 input 有上限
_LOCAL_PROVIDERS = {"42model", "ollama"}


def get_llm(**overrides):
    """按 cfg['llm']['provider'] 创建 LLM。overrides 覆盖默认配置。"""
    c = cfg["llm"]
    provider = c["provider"]
    pc = c["providers"][provider]
    kwargs = dict(
        base_url=pc["base_url"],
        model=pc["model"],
        api_key=_LLM_API_KEY[provider](),
        temperature=c.get("temperature", 0.0),
        max_tokens=c.get("max_tokens", 512),
        timeout=60,
        max_retries=2,
    )
    kwargs.update(overrides)
    return ChatOpenAI(**kwargs)


def get_embedder(**overrides):
    """按 cfg['embedding']['provider'] 创建 embedder。

    本地后端(42model/ollama): check_embedding_ctx_length=False + chunk_size=128
      (不接受 token-id 输入且单次 input 有上限);
    云端: 用默认长度检查(超长由 API 处理), 避免盲目关闭截断。
    """
    provider = cfg["embedding"]["provider"]
    pc = cfg["embedding"]["providers"][provider]
    kwargs = dict(
        base_url=pc["base_url"],
        model=pc["model"],
        api_key=_EMB_API_KEY[provider](),
        timeout=60,
        max_retries=2,
    )
    if provider in _LOCAL_PROVIDERS:
        kwargs["check_embedding_ctx_length"] = False
        kwargs["chunk_size"] = 128
    kwargs.update(overrides)
    return OpenAIEmbeddings(**kwargs)
