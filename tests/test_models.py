"""models.py 完整测试: LLM/embedder 工厂 + API key 来源。

全部纯本地运行, 不调任何外部服务(DeepSeek / 42model / 网络)。

mock 策略:
  - API key 来源 (_LLM_API_KEY / _EMB_API_KEY): 直接调 lambda,
    用 monkeypatch.setenv / delenv 控制环境变量, 保证确定性。
  - get_llm / get_embedder 的 kwargs: monkeypatch 把 ChatOpenAI / OpenAIEmbeddings
    换成 kwargs 捕获器, 避开 langchain-openai 构造时对空 api_key 的校验
    (cloud provider 未设 LLM_API_KEY 时 api_key='' 会抛 Missing credentials)。
  - 真实构造冒烟测试: 仅对 key 非空的确定路径做真实实例化。
"""
from __future__ import annotations

import pytest

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from rag_agent.models import _EMB_API_KEY, _LLM_API_KEY, get_embedder, get_llm


# =====================================================================
# 辅助: 捕获传给构造器的 kwargs
# =====================================================================

def _capture(monkeypatch, target: str) -> dict:
    """把 rag_agent.models.<target> 替换为 kwargs 捕获器, 返回捕获 dict。

    target: 'ChatOpenAI' 或 'OpenAIEmbeddings'。
    捕获器返回自身 dict 作为"实例"(足够做断言, 不触网络)。
    """
    captured: dict = {}

    def _fake(**kwargs):  # noqa: ANN202
        captured.update(kwargs)
        return captured

    monkeypatch.setattr(f"rag_agent.models.{target}", _fake)
    return captured


# =====================================================================
# API key 来源: _LLM_API_KEY 的 lambda
# =====================================================================


class TestLLMApiKeySource:
    """_LLM_API_KEY 各 provider 的 key 获取逻辑。"""

    def test_cloud_reads_env_var(self, monkeypatch):
        """cloud provider 的 key 从 LLM_API_KEY 读。"""
        monkeypatch.setenv("LLM_API_KEY", "sk-test-123")
        assert _LLM_API_KEY["cloud"]() == "sk-test-123"

    def test_cloud_defaults_to_empty(self, monkeypatch):
        """LLM_API_KEY 未设置时返回空串(不抛异常)。"""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        assert _LLM_API_KEY["cloud"]() == ""

    def test_ollama_is_fixed_placeholder(self):
        """ollama 用固定占位 'ollama', 不读环境变量。"""
        assert _LLM_API_KEY["ollama"]() == "ollama"

    def test_42model_is_fixed_placeholder(self):
        """42model 用固定占位 '42model'。"""
        assert _LLM_API_KEY["42model"]() == "42model"


# =====================================================================
# API key 来源: _EMB_API_KEY 的 lambda
# =====================================================================


class TestEmbedderApiKeySource:
    """_EMB_API_KEY 各 provider 的 key 获取逻辑。"""

    def test_cloud_reads_env_var(self, monkeypatch):
        """cloud provider 的 key 从 EMBEDDING_CLOUD_API_KEY 读。"""
        monkeypatch.setenv("EMBEDDING_CLOUD_API_KEY", "sk-emb-456")
        assert _EMB_API_KEY["cloud"]() == "sk-emb-456"

    def test_cloud_defaults_to_empty(self, monkeypatch):
        """EMBEDDING_CLOUD_API_KEY 未设置时返回空串。"""
        monkeypatch.delenv("EMBEDDING_CLOUD_API_KEY", raising=False)
        assert _EMB_API_KEY["cloud"]() == ""

    @pytest.mark.parametrize("provider,expected", [("42model", "42model"), ("ollama", "ollama")])
    def test_local_providers_fixed(self, provider, expected):
        """本地后端(42model/ollama)用固定占位 key, 不读环境变量。"""
        assert _EMB_API_KEY[provider]() == expected


# =====================================================================
# get_llm 工厂
# =====================================================================


class TestGetLLM:
    """get_llm: 按 cfg['llm']['provider'] 构造 ChatOpenAI。"""

    def test_returns_chatonenai_instance(self, monkeypatch):
        """真实构造冒烟测试: 默认 cloud provider, 设 env 后返回 ChatOpenAI 实例。"""
        monkeypatch.setenv("LLM_API_KEY", "fake-key")
        llm = get_llm()
        assert isinstance(llm, ChatOpenAI)

    def test_cloud_provider_kwargs(self, monkeypatch):
        """cloud provider: kwargs 取自 cfg + LLM_API_KEY 环境变量。"""
        monkeypatch.setenv("LLM_API_KEY", "sk-cloud-xyz")
        captured = _capture(monkeypatch, "ChatOpenAI")
        get_llm()
        assert captured["base_url"] == "https://api.deepseek.com/v1"
        assert captured["model"] == "deepseek-chat"
        assert captured["api_key"] == "sk-cloud-xyz"
        assert captured["temperature"] == 0.0
        assert captured["max_tokens"] == 512

    def test_overrides_take_precedence(self, monkeypatch):
        """overrides 覆盖默认配置, 未覆盖项保持默认。"""
        captured = _capture(monkeypatch, "ChatOpenAI")
        get_llm(temperature=0.7, max_tokens=1024, model="deepseek-reasoner")
        assert captured["temperature"] == 0.7
        assert captured["max_tokens"] == 1024
        assert captured["model"] == "deepseek-reasoner"
        # 未覆盖的保持默认
        assert captured["base_url"] == "https://api.deepseek.com/v1"

    def test_defaults_when_cfg_missing_optional_keys(self, monkeypatch):
        """cfg 缺 temperature/max_tokens 时用默认值 0.0 / 512。"""
        fake_cfg = {
            "llm": {
                "provider": "cloud",
                "providers": {"cloud": {"base_url": "http://x/v1", "model": "m"}},
            }
        }
        monkeypatch.setattr("rag_agent.models.cfg", fake_cfg)
        captured = _capture(monkeypatch, "ChatOpenAI")
        get_llm()
        assert captured["temperature"] == 0.0
        assert captured["max_tokens"] == 512

    def test_local_provider_uses_fixed_key(self, monkeypatch):
        """ollama provider: api_key='ollama'(不读 env), 其余取自 cfg。"""
        fake_cfg = {
            "llm": {
                "provider": "ollama",
                "temperature": 0.1,
                "max_tokens": 100,
                "providers": {
                    "ollama": {"base_url": "http://localhost:11434/v1", "model": "qwen2.5:7b"}
                },
            }
        }
        monkeypatch.setattr("rag_agent.models.cfg", fake_cfg)
        captured = _capture(monkeypatch, "ChatOpenAI")
        get_llm()
        assert captured["api_key"] == "ollama"
        assert captured["base_url"] == "http://localhost:11434/v1"
        assert captured["model"] == "qwen2.5:7b"
        assert captured["temperature"] == 0.1
        assert captured["max_tokens"] == 100

    def test_unknown_provider_raises_keyerror(self, monkeypatch):
        """provider 不在 providers 块中 -> KeyError。"""
        fake_cfg = {
            "llm": {
                "provider": "azure",
                "providers": {"cloud": {"base_url": "x", "model": "y"}},
            }
        }
        monkeypatch.setattr("rag_agent.models.cfg", fake_cfg)
        with pytest.raises(KeyError):
            get_llm()


# =====================================================================
# get_embedder 工厂
# =====================================================================


class TestGetEmbedder:
    """get_embedder: 按 cfg['embedding']['provider'] 构造 OpenAIEmbeddings。"""

    def test_returns_openaiembeddings_instance(self):
        """真实构造冒烟测试: 默认 42model provider, key='42model' 非空, 可直接构造。"""
        emb = get_embedder()
        assert isinstance(emb, OpenAIEmbeddings)

    def test_default_provider_kwargs(self, monkeypatch):
        """默认 42model provider: kwargs 含正确的 base_url/model/key + 固定参数。"""
        captured = _capture(monkeypatch, "OpenAIEmbeddings")
        get_embedder()
        assert captured["base_url"] == "http://localhost:11520/v1"
        assert captured["model"] == "qwen3-embedding:0.6b-q8_0"
        assert captured["api_key"] == "42model"
        # 本地后端固定参数
        assert captured["check_embedding_ctx_length"] is False
        assert captured["chunk_size"] == 128

    def test_overrides_take_precedence(self, monkeypatch):
        """overrides 覆盖默认配置, 未覆盖项保持默认。"""
        captured = _capture(monkeypatch, "OpenAIEmbeddings")
        get_embedder(model="bge-m3", chunk_size=256)
        assert captured["model"] == "bge-m3"
        assert captured["chunk_size"] == 256
        # 未覆盖保持默认
        assert captured["check_embedding_ctx_length"] is False
        assert captured["api_key"] == "42model"

    def test_cloud_provider_reads_env(self, monkeypatch):
        """cloud provider: api_key 从 EMBEDDING_CLOUD_API_KEY 读。"""
        fake_cfg = {
            "embedding": {
                "provider": "cloud",
                "providers": {
                    "cloud": {
                        "base_url": "https://api.siliconflow.cn/v1",
                        "model": "BAAI/bge-m3",
                    }
                },
            }
        }
        monkeypatch.setattr("rag_agent.models.cfg", fake_cfg)
        monkeypatch.setenv("EMBEDDING_CLOUD_API_KEY", "sk-emb-cloud")
        captured = _capture(monkeypatch, "OpenAIEmbeddings")
        get_embedder()
        assert captured["api_key"] == "sk-emb-cloud"
        assert captured["base_url"] == "https://api.siliconflow.cn/v1"
        assert captured["model"] == "BAAI/bge-m3"

    def test_local_provider_kwargs(self, monkeypatch):
        """ollama provider: api_key='ollama', 其余取自 cfg。"""
        fake_cfg = {
            "embedding": {
                "provider": "ollama",
                "providers": {
                    "ollama": {"base_url": "http://localhost:11434/v1", "model": "bge-m3"}
                },
            }
        }
        monkeypatch.setattr("rag_agent.models.cfg", fake_cfg)
        captured = _capture(monkeypatch, "OpenAIEmbeddings")
        get_embedder()
        assert captured["api_key"] == "ollama"
        assert captured["base_url"] == "http://localhost:11434/v1"
        assert captured["model"] == "bge-m3"
        # 固定参数不随 provider 变
        assert captured["check_embedding_ctx_length"] is False
        assert captured["chunk_size"] == 128

    def test_unknown_provider_raises_keyerror(self, monkeypatch):
        """provider 不在 providers 块中 -> KeyError。"""
        fake_cfg = {
            "embedding": {
                "provider": "huggingface",
                "providers": {"42model": {"base_url": "x", "model": "y"}},
            }
        }
        monkeypatch.setattr("rag_agent.models.cfg", fake_cfg)
        with pytest.raises(KeyError):
            get_embedder()
