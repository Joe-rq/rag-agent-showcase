"""检索器 (retrieval.py) 的单元测试。

不调任何外部服务 (DeepSeek / 42model / 网络 / FAISS):
  - get_llm 用 monkeypatch 替换为返回固定 AIMessage 的 RunnableLambda
  - FAISS index 用 FakeIndex (只实现 as_retriever) 替换

用法: uv run pytest tests/test_retrieval.py -q
"""
from __future__ import annotations

import urllib.request

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from rag_agent import retrieval
from rag_agent.config import cfg
from rag_agent.retrieval import HYDE_PROMPT, _hyde_generate, make_retriever


# ---------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------
class FakeRetriever:
    """模拟 index.as_retriever() 的返回值: 只需 .invoke(query) -> List[Document]。"""

    def __init__(self, docs=None):
        self.docs = docs or []
        self.invoked_with: list = []

    def invoke(self, query):
        # 每次返回新拷贝, 避免 hyde 路径改 metadata 时跨调用互相污染
        self.invoked_with.append(query)
        return [
            Document(page_content=d.page_content, metadata=dict(d.metadata))
            for d in self.docs
        ]


class FakeIndex:
    """模拟 FAISS index: 只需 .as_retriever(search_kwargs=...)。"""

    def __init__(self, retriever=None):
        self.retriever = retriever or FakeRetriever()
        self.as_retriever_kwargs = None

    def as_retriever(self, **kwargs):
        self.as_retriever_kwargs = kwargs
        return self.retriever


def _fake_llm_factory(text="假设文档内容"):
    """返回 get_llm 的替身工厂。

    替身忽略 prompt 输入, 恒定返回 content=text 的 AIMessage;
    并把最后一次被调用时的 kwargs 记到 .last_kwargs 供断言。
    """

    def fake_get_llm(**kwargs):
        fake_get_llm.last_kwargs = kwargs
        return RunnableLambda(lambda _prompt: AIMessage(content=text))

    fake_get_llm.last_kwargs = None
    return fake_get_llm


# ---------------------------------------------------------------------
# HYDE_PROMPT (纯对象, 无需 mock)
# ---------------------------------------------------------------------
class TestHydePrompt:
    def test_format_contains_question_and_template_markers(self):
        msgs = HYDE_PROMPT.format_messages(question="什么是 RAG?")
        assert len(msgs) == 1
        content = msgs[0].content
        assert "什么是 RAG?" in content
        assert "假设文档" in content

    def test_format_accepts_various_questions(self):
        for q in ("普通问题", "", "含特殊字符 <>&"):
            content = HYDE_PROMPT.format_messages(question=q)[0].content
            assert q in content


# ---------------------------------------------------------------------
# _hyde_generate (mock get_llm)
# ---------------------------------------------------------------------
class TestHydeGenerate:
    def test_returns_parsed_string(self, monkeypatch):
        monkeypatch.setattr(retrieval, "get_llm", _fake_llm_factory("假设答案文本"))
        assert _hyde_generate("什么是 RAG?") == "假设答案文本"

    def test_llm_called_with_temperature_zero_and_capped_tokens(self, monkeypatch):
        fake = _fake_llm_factory()
        monkeypatch.setattr(retrieval, "get_llm", fake)
        _hyde_generate("任意问题")
        assert fake.last_kwargs["temperature"] == 0.0
        assert fake.last_kwargs["max_tokens"] == 200

    def test_does_not_trigger_network(self, monkeypatch):
        """get_llm 被替换后, _hyde_generate 不应发起任何 urllib 请求。"""

        def _fail(*a, **kw):
            raise AssertionError("不应发起网络请求")

        monkeypatch.setattr(urllib.request, "urlopen", _fail)
        monkeypatch.setattr(retrieval, "get_llm", _fake_llm_factory())
        assert _hyde_generate("q") == "假设文档内容"


# ---------------------------------------------------------------------
# make_retriever
# ---------------------------------------------------------------------
class TestMakeRetriever:
    # --- 路由: none / rerank / 未知 -> base ---
    def test_none_returns_base_retriever(self):
        idx = FakeIndex()
        r = make_retriever(idx, top_k=3, advanced="none")
        assert r is idx.retriever
        assert idx.as_retriever_kwargs == {"search_kwargs": {"k": 3}}

    def test_rerank_falls_back_to_base(self):
        idx = FakeIndex()
        r = make_retriever(idx, top_k=5, advanced="rerank")
        assert r is idx.retriever

    def test_unknown_advanced_falls_through_to_base(self):
        idx = FakeIndex()
        r = make_retriever(idx, top_k=2, advanced="bogus")
        assert r is idx.retriever

    def test_base_path_never_calls_get_llm(self, monkeypatch):
        """none 路径返回的 retriever 不应触碰 get_llm。"""

        def _fail(**kw):
            raise AssertionError("none 路径不应调用 get_llm")

        monkeypatch.setattr(retrieval, "get_llm", _fail)
        base = FakeRetriever(docs=[Document(page_content="d")])
        idx = FakeIndex(retriever=base)
        r = make_retriever(idx, advanced="none")
        out = r.invoke("q")  # 若误走 hyde 会触发 _fail
        assert len(out) == 1

    # --- 默认值走 config ---
    def test_default_top_k_from_config(self):
        idx = FakeIndex()
        make_retriever(idx, advanced="none")
        assert idx.as_retriever_kwargs["search_kwargs"]["k"] == cfg["retrieval"]["top_k"]

    def test_default_advanced_routes_to_hyde(self):
        """advanced 缺省时取 cfg['retrieval']['advanced'](当前为 hyde)。"""
        assert cfg["retrieval"]["advanced"] == "hyde"
        idx = FakeIndex()
        r = make_retriever(idx)  # 全默认
        assert isinstance(r, RunnableLambda)

    def test_explicit_top_k_overrides_config(self):
        idx = FakeIndex()
        make_retriever(idx, top_k=7, advanced="none")
        assert idx.as_retriever_kwargs["search_kwargs"]["k"] == 7

    def test_search_kwargs_shape(self):
        idx = FakeIndex()
        make_retriever(idx, top_k=9, advanced="none")
        kw = idx.as_retriever_kwargs
        assert set(kw.keys()) == {"search_kwargs"}
        assert isinstance(kw["search_kwargs"], dict)
        assert kw["search_kwargs"]["k"] == 9

    # --- hyde 路径 ---
    def test_hyde_returns_runnable_lambda(self):
        idx = FakeIndex()
        r = make_retriever(idx, top_k=3, advanced="hyde")
        assert isinstance(r, RunnableLambda)
        assert isinstance(r, Runnable)

    def test_hyde_invoke_calls_llm_then_base_with_hyde_doc(self, monkeypatch):
        monkeypatch.setattr(retrieval, "get_llm", _fake_llm_factory("HYPOTHETICAL"))
        base = FakeRetriever(
            docs=[Document(page_content="d1"), Document(page_content="d2")]
        )
        idx = FakeIndex(retriever=base)
        r = make_retriever(idx, top_k=2, advanced="hyde")
        out = r.invoke("什么是 RAG?")
        # base 拿到的是 hyde 生成文本, 不是原问题
        assert base.invoked_with == ["HYPOTHETICAL"]
        assert len(out) == 2

    def test_hyde_tags_docs_with_retrieval_metadata(self, monkeypatch):
        monkeypatch.setattr(retrieval, "get_llm", _fake_llm_factory("H"))
        base = FakeRetriever(docs=[Document(page_content="d1", metadata={"src": "a"})])
        idx = FakeIndex(retriever=base)
        r = make_retriever(idx, advanced="hyde")
        out = r.invoke("q")
        assert all(d.metadata.get("retrieval") == "hyde" for d in out)
        assert out[0].metadata["src"] == "a"  # 原 metadata 被保留

    def test_hyde_does_not_mutate_source_docs(self, monkeypatch):
        monkeypatch.setattr(retrieval, "get_llm", _fake_llm_factory("H"))
        src = [Document(page_content="d1", metadata={})]
        base = FakeRetriever(docs=src)
        idx = FakeIndex(retriever=base)
        r = make_retriever(idx, advanced="hyde")
        r.invoke("q")
        assert src[0].metadata == {}  # 原始 doc 未被打标

    # --- 边界 ---
    def test_hyde_empty_question(self, monkeypatch):
        monkeypatch.setattr(retrieval, "get_llm", _fake_llm_factory("H"))
        base = FakeRetriever(docs=[Document(page_content="d")])
        idx = FakeIndex(retriever=base)
        r = make_retriever(idx, advanced="hyde")
        out = r.invoke("")
        assert len(out) == 1

    def test_hyde_empty_result(self, monkeypatch):
        monkeypatch.setattr(retrieval, "get_llm", _fake_llm_factory("H"))
        base = FakeRetriever(docs=[])
        idx = FakeIndex(retriever=base)
        r = make_retriever(idx, advanced="hyde")
        assert r.invoke("q") == []

    # --- 错误传播 ---
    def test_hyde_propagates_llm_error(self, monkeypatch):
        def _raising_llm(**kw):
            raise RuntimeError("llm down")

        monkeypatch.setattr(retrieval, "get_llm", _raising_llm)
        idx = FakeIndex()
        r = make_retriever(idx, advanced="hyde")
        with pytest.raises(RuntimeError, match="llm down"):
            r.invoke("q")
