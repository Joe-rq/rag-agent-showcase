"""RAG 链(chain.py)的测试。

覆盖: 纯函数(strip_think / _docs2str)直接测; build_rag_chain 用 monkeypatch
隔离外部依赖(make_retriever / get_llm 返回 RunnableLambda), 不触达 DeepSeek /
42model / 网络 / FAISS。

用法: uv run pytest tests/test_chain.py -q
"""
import re

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

import rag_agent.chain as chain_mod
from rag_agent.chain import (
    RAG_PROMPT,
    THINK_RE,
    _docs2str,
    build_rag_chain,
    strip_think,
)


# --------------------------------------------------------------------------- #
# strip_think (纯函数)
# --------------------------------------------------------------------------- #
def test_strip_think_removes_single_block():
    assert strip_think("a<think>hidden</think>b") == "ab"


def test_strip_think_removes_multiple_blocks():
    assert strip_think("<think>x</think> keep <think>y</think> end") == "keep  end"


def test_strip_think_handles_multiline_with_dotall():
    """THINK_RE 带 re.DOTALL, 跨行的 think 块也应整体剥离。"""
    text = "<think>\nline1\nline2\n</think>answer"
    assert strip_think(text) == "answer"


def test_strip_think_no_tags_unchanged():
    assert strip_think("plain answer") == "plain answer"


def test_strip_think_only_block_returns_empty():
    assert strip_think("<think>only reasoning</think>") == ""


def test_strip_think_empty_string():
    assert strip_think("") == ""


def test_strip_think_strips_outer_whitespace():
    assert strip_think("   <think>x</think>   hi   ") == "hi"


def test_strip_think_strips_unclosed_tag_to_end():
    """缺闭合标签(被 max_tokens 截断)时, <think> 到串尾一并剥离, 不泄漏思考内容。"""
    assert strip_think("<think>no close") == ""


def test_strip_think_strips_unclosed_tag_preserves_prefix():
    """截断的 think 块前的正常文本应保留。"""
    assert strip_think("keep this<think>truncated reasoning") == "keep this"


# --------------------------------------------------------------------------- #
# _docs2str (纯函数)
# --------------------------------------------------------------------------- #
def test_docs2str_empty_list():
    assert _docs2str([]) == ""


def test_docs2str_single_doc():
    d = Document(page_content="hello world", metadata={"source": "a.pdf", "page": 0})
    assert _docs2str([d]) == "[a.pdf:p0] hello world"


def test_docs2str_multiple_docs_joined_with_separator():
    d1 = Document(page_content="c1", metadata={"source": "a.pdf", "page": 0})
    d2 = Document(page_content="c2", metadata={"source": "b.pdf", "page": 3})
    assert _docs2str([d1, d2]) == "[a.pdf:p0] c1\n\n---\n\n[b.pdf:p3] c2"


def test_docs2str_missing_metadata_defaults_to_question_mark():
    d = Document(page_content="c", metadata={})
    assert _docs2str([d]) == "[?:p?] c"


def test_docs2str_partial_metadata():
    d = Document(page_content="c", metadata={"source": "only.pdf"})
    assert _docs2str([d]) == "[only.pdf:p?] c"


# --------------------------------------------------------------------------- #
# 模块常量
# --------------------------------------------------------------------------- #
def test_rag_prompt_has_context_and_question_vars():
    assert set(RAG_PROMPT.input_variables) == {"context", "question"}


def test_think_re_compiled_with_dotall():
    assert THINK_RE.flags & re.DOTALL


# --------------------------------------------------------------------------- #
# build_rag_chain (隔离 make_retriever / get_llm)
# --------------------------------------------------------------------------- #
def _fake_llm(content: str) -> RunnableLambda:
    """返回 RunnableLambda: 忽略输入, 产出给定 content 的 AIMessage。"""
    return RunnableLambda(lambda _prompt: AIMessage(content=content))


def _fake_retriever(docs) -> RunnableLambda:
    return RunnableLambda(lambda _q: docs)


def test_build_rag_chain_returns_runnable(monkeypatch):
    monkeypatch.setattr(
        chain_mod, "make_retriever", lambda index, advanced=None: _fake_retriever([])
    )
    monkeypatch.setattr(chain_mod, "get_llm", lambda: _fake_llm("ok"))
    rag = build_rag_chain("idx")
    assert isinstance(rag, Runnable)


def test_build_rag_chain_passes_index_and_advanced(monkeypatch):
    calls = []

    def fake_make_retriever(index, top_k=None, advanced=None):
        calls.append((index, advanced))
        return _fake_retriever([])

    monkeypatch.setattr(chain_mod, "make_retriever", fake_make_retriever)
    monkeypatch.setattr(chain_mod, "get_llm", lambda: _fake_llm("ok"))

    build_rag_chain("my_index", advanced="hyde")
    assert calls == [("my_index", "hyde")]


def test_build_rag_chain_default_advanced_is_none(monkeypatch):
    calls = []

    def fake_make_retriever(index, top_k=None, advanced=None):
        calls.append(advanced)
        return _fake_retriever([])

    monkeypatch.setattr(chain_mod, "make_retriever", fake_make_retriever)
    monkeypatch.setattr(chain_mod, "get_llm", lambda: _fake_llm("ok"))

    build_rag_chain("idx")
    assert calls == [None]


def test_build_rag_chain_calls_get_llm_once_at_build(monkeypatch):
    llm_calls = []

    def fake_get_llm():
        llm_calls.append(1)
        return _fake_llm("ok")

    monkeypatch.setattr(
        chain_mod, "make_retriever", lambda index, advanced=None: _fake_retriever([])
    )
    monkeypatch.setattr(chain_mod, "get_llm", fake_get_llm)

    build_rag_chain("idx")
    assert len(llm_calls) == 1


def test_chain_end_to_end_strips_think(monkeypatch):
    docs = [Document(page_content="RAG is X", metadata={"source": "a.pdf", "page": 1})]
    monkeypatch.setattr(
        chain_mod, "make_retriever", lambda index, advanced=None: _fake_retriever(docs)
    )
    monkeypatch.setattr(
        chain_mod, "get_llm", lambda: _fake_llm("<think>reasoning</think>final answer")
    )

    rag = build_rag_chain("idx")
    out = rag.invoke("what is RAG?")
    assert out == "final answer"
    assert "<think>" not in out


def test_chain_context_pipeline_uses_docs2str(monkeypatch):
    """校验 _docs2str 的格式化结果确实流入了 prompt 的 context。"""
    docs = [Document(page_content="content here", metadata={"source": "s.pdf", "page": 2})]
    captured = {}

    def fake_llm_fn(prompt_value):
        captured["prompt"] = str(prompt_value)
        return AIMessage(content="ans")

    monkeypatch.setattr(
        chain_mod, "make_retriever", lambda index, advanced=None: _fake_retriever(docs)
    )
    monkeypatch.setattr(chain_mod, "get_llm", lambda: RunnableLambda(fake_llm_fn))

    rag = build_rag_chain("idx")
    rag.invoke("question?")
    assert "[s.pdf:p2] content here" in captured["prompt"]


def test_chain_passes_question_to_retriever(monkeypatch):
    """retriever 收到的应当是原始 question 字符串。"""
    seen = []
    monkeypatch.setattr(
        chain_mod,
        "make_retriever",
        lambda index, advanced=None: RunnableLambda(lambda q: seen.append(q) or []),
    )
    monkeypatch.setattr(chain_mod, "get_llm", lambda: _fake_llm("ok"))

    rag = build_rag_chain("idx")
    rag.invoke("the question")
    assert seen == ["the question"]


def test_chain_propagates_retriever_invoke_error(monkeypatch):
    def boom(_q):
        raise RuntimeError("retriever down")

    monkeypatch.setattr(
        chain_mod, "make_retriever", lambda index, advanced=None: RunnableLambda(boom)
    )
    monkeypatch.setattr(chain_mod, "get_llm", lambda: _fake_llm("ok"))

    rag = build_rag_chain("idx")
    with pytest.raises(RuntimeError, match="retriever down"):
        rag.invoke("q")


def test_chain_propagates_llm_invoke_error(monkeypatch):
    def boom(_prompt):
        raise RuntimeError("llm down")

    monkeypatch.setattr(
        chain_mod, "make_retriever", lambda index, advanced=None: _fake_retriever([])
    )
    monkeypatch.setattr(chain_mod, "get_llm", lambda: RunnableLambda(boom))

    rag = build_rag_chain("idx")
    with pytest.raises(RuntimeError, match="llm down"):
        rag.invoke("q")


def test_build_rag_chain_propagates_make_retriever_error(monkeypatch):
    def fake_make_retriever(index, advanced=None):
        raise ValueError("bad index")

    monkeypatch.setattr(chain_mod, "make_retriever", fake_make_retriever)
    monkeypatch.setattr(chain_mod, "get_llm", lambda: _fake_llm("ok"))

    with pytest.raises(ValueError, match="bad index"):
        build_rag_chain("idx")


def test_build_rag_chain_propagates_get_llm_error(monkeypatch):
    monkeypatch.setattr(
        chain_mod, "make_retriever", lambda index, advanced=None: _fake_retriever([])
    )

    def fake_get_llm():
        raise OSError("missing api key")

    monkeypatch.setattr(chain_mod, "get_llm", fake_get_llm)

    with pytest.raises(OSError, match="missing api key"):
        build_rag_chain("idx")
