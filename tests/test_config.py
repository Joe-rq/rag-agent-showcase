"""配置加载 + 配置驱动的 provider 工厂 / 纯函数 / 索引与检索的测试。

覆盖范围:
  - config.py:  load_config / cfg 结构 / PROJECT_ROOT / CONFIG_PATH
  - models.py:  get_llm / get_embedder (构造器 mock, 不调任何 API)
  - chain.py:   strip_think / _docs2str (纯函数)
  - indexing.py: make_splitter (纯函数) / download_papers (mock urllib) /
                 build_index / load_index (mock FAISS + 工厂)
  - retrieval.py: make_retriever 各分支 (fake index)

硬性约束: 不触达外部服务 (DeepSeek / 42model / 网络)。所有外部依赖用
monkeypatch 替换为 fake / Mock。

用法: uv run pytest tests/test_config.py -q
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_agent.chain import _docs2str, strip_think
from rag_agent.config import CONFIG_PATH, PROJECT_ROOT, cfg, load_config
from rag_agent.indexing import build_index, download_papers, load_index, make_splitter
from rag_agent.models import _EMB_API_KEY, _LLM_API_KEY, get_embedder, get_llm
from rag_agent.retrieval import make_retriever


# --------------------------------------------------------------------- #
# Fake 类: 捕获 get_llm / get_embedder 传给底层客户端的 kwargs, 且不构造
# 真实 OpenAI 客户端 (避免依赖 api_key / 触发网络)。
# --------------------------------------------------------------------- #
class FakeChatOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeOpenAIEmbeddings:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_llm_ctor(monkeypatch):
    """把 models 模块里的 ChatOpenAI 换成捕获 kwargs 的 fake。"""
    monkeypatch.setattr("rag_agent.models.ChatOpenAI", FakeChatOpenAI)
    return FakeChatOpenAI


@pytest.fixture
def fake_emb_ctor(monkeypatch):
    """把 models 模块里的 OpenAIEmbeddings 换成捕获 kwargs 的 fake。"""
    monkeypatch.setattr("rag_agent.models.OpenAIEmbeddings", FakeOpenAIEmbeddings)
    return FakeOpenAIEmbeddings


# ===================================================================== #
# config.py: 结构 + 缓存 + 路径
# ===================================================================== #
def test_config_has_providers():
    """config.yaml 的 llm/embedding 都有 provider + providers 块。"""
    for section in ("llm", "embedding"):
        assert "provider" in cfg[section]
        assert cfg[section]["provider"] in cfg[section]["providers"]


def test_default_providers():
    """默认 LLM=cloud, embedding=42model。"""
    assert cfg["llm"]["provider"] == "cloud"
    assert cfg["embedding"]["provider"] == "42model"


def test_provider_api_key_sources():
    """每个 provider 都有对应的 api_key 获取方式。"""
    for p in cfg["llm"]["providers"]:
        assert p in _LLM_API_KEY
    for p in cfg["embedding"]["providers"]:
        assert p in _EMB_API_KEY


def test_embedding_dimensions_consistent():
    """embedding 维度配置存在且为正整数。"""
    d = cfg["embedding"]["dimensions"]
    assert isinstance(d, int) and d > 0


def test_load_config_returns_dict():
    """load_config 返回 dict 且与模块级 cfg 内容一致。"""
    c = load_config()
    assert isinstance(c, dict)
    assert c is cfg


def test_load_config_is_cached():
    """lru_cache: 两次调用返回同一对象。"""
    assert load_config() is load_config()


def test_cfg_top_level_sections():
    """cfg 含全部顶层配置段。"""
    for key in ("llm", "embedding", "dataset", "chunking", "retrieval", "eval"):
        assert key in cfg, f"缺少顶层段: {key}"


def test_project_root_resolves_to_repo_root():
    """PROJECT_ROOT = src/rag_agent/config.py 的 parents[2], 即仓库根。"""
    assert PROJECT_ROOT.is_dir()
    assert (PROJECT_ROOT / "src" / "rag_agent" / "config.py").is_file()


def test_config_path_is_existing_yaml():
    """CONFIG_PATH 指向真实存在的 config.yaml。"""
    assert CONFIG_PATH.is_file()
    assert CONFIG_PATH.name == "config.yaml"
    assert CONFIG_PATH == PROJECT_ROOT / "config.yaml"


@pytest.mark.parametrize("section", ["llm", "embedding"])
def test_providers_have_required_fields(section):
    """每个 provider 子配置都带 base_url + model。"""
    for name, pc in cfg[section]["providers"].items():
        assert "base_url" in pc, f"{section}.{name} 缺 base_url"
        assert "model" in pc, f"{section}.{name} 缺 model"
        assert isinstance(pc["base_url"], str) and pc["base_url"]
        assert isinstance(pc["model"], str) and pc["model"]


def test_llm_config_scalar_fields():
    assert isinstance(cfg["llm"].get("temperature"), (int, float))
    assert isinstance(cfg["llm"].get("max_tokens"), int) and cfg["llm"]["max_tokens"] > 0


def test_chunking_default_fields():
    ch = cfg["chunking"]["default"]
    assert ch["strategy"] in ("fixed", "recursive", "markdown")
    assert isinstance(ch["chunk_size"], int) and ch["chunk_size"] > 0
    assert isinstance(ch["chunk_overlap"], int) and 0 <= ch["chunk_overlap"] < ch["chunk_size"]


def test_chunking_variants_structure():
    variants = cfg["chunking"]["variants"]
    assert isinstance(variants, list) and variants
    for v in variants:
        assert v["strategy"] in ("fixed", "recursive", "markdown")
        assert isinstance(v["chunk_size"], int) and v["chunk_size"] > 0
        assert 0 <= v["chunk_overlap"] < v["chunk_size"]


def test_dataset_papers_structure():
    papers = cfg["dataset"]["papers"]
    assert isinstance(papers, list) and len(papers) >= 1
    for p in papers:
        assert "id" in p and "name" in p
        assert isinstance(p["id"], str) and p["id"]
        assert isinstance(p["name"], str) and p["name"]


def test_retrieval_config():
    assert isinstance(cfg["retrieval"]["top_k"], int) and cfg["retrieval"]["top_k"] > 0
    assert cfg["retrieval"]["advanced"] in ("none", "hyde", "rerank")


def test_eval_config():
    ev = cfg["eval"]
    assert isinstance(ev["num_questions"], int) and ev["num_questions"] > 0
    assert isinstance(ev["metrics"], list) and ev["metrics"]


# ===================================================================== #
# models.py: get_llm / get_embedder (构造器 mock, 不调 API)
# ===================================================================== #
def test_get_llm_passes_default_provider_kwargs(fake_llm_ctor):
    """get_llm() 按 cfg 默认 provider (cloud) 传 base_url/model/api_key/温度/长度。"""
    llm = get_llm()
    pc = cfg["llm"]["providers"][cfg["llm"]["provider"]]
    assert llm.kwargs["base_url"] == pc["base_url"]
    assert llm.kwargs["model"] == pc["model"]
    assert llm.kwargs["temperature"] == cfg["llm"]["temperature"]
    assert llm.kwargs["max_tokens"] == cfg["llm"]["max_tokens"]
    assert "api_key" in llm.kwargs


def test_get_llm_applies_overrides(fake_llm_ctor):
    """overrides 覆盖默认 kwargs。"""
    llm = get_llm(temperature=0.9, model="custom-model", max_tokens=123)
    assert llm.kwargs["temperature"] == 0.9
    assert llm.kwargs["model"] == "custom-model"
    assert llm.kwargs["max_tokens"] == 123


@pytest.mark.parametrize("provider", ["cloud", "ollama", "42model"])
def test_get_llm_each_provider_kwargs(fake_llm_ctor, monkeypatch, provider):
    """切换 cfg['llm']['provider'] 后 get_llm 用对应后端的 base_url/model。"""
    monkeypatch.setitem(cfg["llm"], "provider", provider)
    llm = get_llm()
    pc = cfg["llm"]["providers"][provider]
    assert llm.kwargs["base_url"] == pc["base_url"]
    assert llm.kwargs["model"] == pc["model"]


def test_get_llm_cloud_api_key_from_env(fake_llm_ctor, monkeypatch):
    """cloud provider 的 api_key 从 LLM_API_KEY 读。"""
    monkeypatch.setitem(cfg["llm"], "provider", "cloud")
    monkeypatch.setenv("LLM_API_KEY", "secret-llm-key")
    assert get_llm().kwargs["api_key"] == "secret-llm-key"


def test_get_llm_cloud_api_key_empty_when_env_missing(fake_llm_ctor, monkeypatch):
    """env 缺失时 cloud api_key 回退为空串, 不抛异常。"""
    monkeypatch.setitem(cfg["llm"], "provider", "cloud")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert get_llm().kwargs["api_key"] == ""


def test_get_embedder_passes_default_kwargs(fake_emb_ctor):
    """get_embedder() 按默认 provider 传 base_url/model/api_key。"""
    emb = get_embedder()
    pc = cfg["embedding"]["providers"][cfg["embedding"]["provider"]]
    assert emb.kwargs["base_url"] == pc["base_url"]
    assert emb.kwargs["model"] == pc["model"]
    assert "api_key" in emb.kwargs


def test_get_embedder_sets_local_backend_flags(fake_emb_ctor):
    """本地后端兼容参数固定注入: check_embedding_ctx_length=False, chunk_size=128。"""
    emb = get_embedder()
    assert emb.kwargs["check_embedding_ctx_length"] is False
    assert emb.kwargs["chunk_size"] == 128


def test_get_embedder_applies_overrides(fake_emb_ctor):
    emb = get_embedder(chunk_size=256, model="emb-override")
    assert emb.kwargs["chunk_size"] == 256
    assert emb.kwargs["model"] == "emb-override"


@pytest.mark.parametrize("provider", ["42model", "ollama", "cloud"])
def test_get_embedder_each_provider_kwargs(fake_emb_ctor, monkeypatch, provider):
    """切换 embedding provider 后用对应后端的 base_url/model。"""
    monkeypatch.setitem(cfg["embedding"], "provider", provider)
    emb = get_embedder()
    pc = cfg["embedding"]["providers"][provider]
    assert emb.kwargs["base_url"] == pc["base_url"]
    assert emb.kwargs["model"] == pc["model"]


def test_get_embedder_cloud_api_key_from_env(fake_emb_ctor, monkeypatch):
    """cloud embedding 的 api_key 从 EMBEDDING_CLOUD_API_KEY 读。"""
    monkeypatch.setitem(cfg["embedding"], "provider", "cloud")
    monkeypatch.setenv("EMBEDDING_CLOUD_API_KEY", "secret-emb-key")
    assert get_embedder().kwargs["api_key"] == "secret-emb-key"


def test_llm_api_key_local_placeholders():
    """本地 LLM 后端用固定占位 api_key, 不读环境变量。"""
    assert _LLM_API_KEY["ollama"]() == "ollama"
    assert _LLM_API_KEY["42model"]() == "42model"
    assert _EMB_API_KEY["42model"]() == "42model"
    assert _EMB_API_KEY["ollama"]() == "ollama"


def test_llm_api_key_cloud_reads_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k1")
    assert _LLM_API_KEY["cloud"]() == "k1"
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert _LLM_API_KEY["cloud"]() == ""


# ===================================================================== #
# chain.py: 纯函数 strip_think / _docs2str
# ===================================================================== #
@pytest.mark.parametrize(
    "text,expected",
    [
        ("", ""),
        ("no think here", "no think here"),
        ("<think>hidden</think>", ""),
        ("<think>x</think>visible", "visible"),
        ("a<think>x</think>b", "ab"),
        ("a<think>x</think>b<think>y</think>c", "abc"),
        ("  <think>x</think>  ", ""),
        ("line1\n<think>\ninner line\n</think>\nline2", "line1\n\nline2"),
    ],
)
def test_strip_think(text, expected):
    assert strip_think(text) == expected


def test_strip_think_removes_all_tags():
    """任意残留的 <think> 开标签都不应留下。"""
    out = strip_think("ok<think>AAA</think>ok<think>BBB</think>ok")
    assert "<think>" not in out
    assert "</think>" not in out
    assert out == "okokok"


@pytest.mark.parametrize(
    "meta,expected_prefix",
    [
        ({"source": "ReAct", "page": 3}, "[ReAct:p3]"),
        ({"page": 1}, "[?:p1]"),
        ({"source": "X"}, "[X:p?]"),
        ({}, "[?:p?]"),
    ],
)
def test_docs2str_formats_metadata(meta, expected_prefix):
    doc = Document(page_content="content body", metadata=meta)
    out = _docs2str([doc])
    assert out.startswith(expected_prefix)
    assert out.endswith("content body")


def test_docs2str_empty_list():
    assert _docs2str([]) == ""


def test_docs2str_joins_multiple_with_separator():
    docs = [
        Document(page_content="a", metadata={"source": "A", "page": 1}),
        Document(page_content="b", metadata={"source": "B", "page": 2}),
    ]
    out = _docs2str(docs)
    assert out == "[A:p1] a\n\n---\n\n[B:p2] b"


# ===================================================================== #
# indexing.py: make_splitter (纯函数)
# ===================================================================== #
def test_make_splitter_fixed_is_character_splitter():
    s = make_splitter("fixed", 800, 0)
    assert type(s) is CharacterTextSplitter
    assert s._chunk_size == 800
    assert s._chunk_overlap == 0


def test_make_splitter_recursive_is_recursive_splitter():
    s = make_splitter("recursive", 800, 120)
    assert isinstance(s, RecursiveCharacterTextSplitter)
    assert s._chunk_size == 800
    assert s._chunk_overlap == 120


def test_make_splitter_markdown_is_recursive_with_heading_separators():
    s = make_splitter("markdown", 500, 50)
    assert isinstance(s, RecursiveCharacterTextSplitter)
    seps = getattr(s, "_separators", None) or s.get_separators()
    assert "\n## " in seps and "\n### " in seps


def test_make_splitter_unknown_strategy_falls_back_to_recursive():
    s = make_splitter("bogus", 300, 10)
    assert isinstance(s, RecursiveCharacterTextSplitter)
    assert s._chunk_size == 300
    assert s._chunk_overlap == 10


# ===================================================================== #
# indexing.py: download_papers (mock urllib.request, 不联网)
# ===================================================================== #
class _FakeResp:
    """模拟 urlopen 返回的上下文管理器。"""

    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._data


@pytest.fixture
def isolated_papers_dir(tmp_path, monkeypatch):
    """把 PAPERS_DIR / INDEX_DIR 重定向到临时目录, 不污染仓库 data/。"""
    papers = tmp_path / "papers"
    index = tmp_path / "index"
    monkeypatch.setattr("rag_agent.indexing.PAPERS_DIR", papers)
    monkeypatch.setattr("rag_agent.indexing.INDEX_DIR", index)
    return papers, index


def test_download_papers_skips_existing(isolated_papers_dir, monkeypatch):
    """已存在的 PDF 直接跳过, 不发起网络请求。"""
    papers, _ = isolated_papers_dir
    papers.mkdir(parents=True)
    existing = papers / "ReAct_2210.03629.pdf"
    existing.write_bytes(b"ALREADY")

    urlopen = MagicMock()
    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("urllib.request.Request", lambda *a, **k: ("REQ", a[0] if a else k))

    paths = download_papers([{"id": "2210.03629", "name": "ReAct"}])
    assert paths == [existing]
    urlopen.assert_not_called()


def test_download_papers_downloads_missing(isolated_papers_dir, monkeypatch):
    """缺失的 PDF 走 arxiv URL 下载并落盘。"""
    papers, _ = isolated_papers_dir
    captured = {}

    def fake_request(url, headers=None):
        captured["url"] = url
        captured["headers"] = headers
        return ("REQ", url)

    urlopen = MagicMock(return_value=_FakeResp(b"%PDF-FAKE"))
    monkeypatch.setattr("urllib.request.Request", fake_request)
    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    paths = download_papers([{"id": "2303.11366", "name": "Reflexion"}])

    target = papers / "Reflexion_2303.11366.pdf"
    assert paths == [target]
    assert target.read_bytes() == b"%PDF-FAKE"
    assert captured["url"] == "https://arxiv.org/pdf/2303.11366"
    urlopen.assert_called_once()
    # urlopen 第二个关键字参数为 timeout=60
    assert urlopen.call_args.kwargs.get("timeout") == 60


def test_download_papers_mixed_existing_and_missing(isolated_papers_dir, monkeypatch):
    """混合场景: 存在的跳过, 缺失的下载。"""
    papers, _ = isolated_papers_dir
    papers.mkdir(parents=True)
    (papers / "ReAct_2210.03629.pdf").write_bytes(b"OLD")

    monkeypatch.setattr("urllib.request.Request", lambda url, headers=None: ("REQ", url))
    urlopen = MagicMock(return_value=_FakeResp(b"%PDF-NEW"))
    monkeypatch.setattr("urllib.request.urlopen", urlopen)

    papers_cfg = [
        {"id": "2210.03629", "name": "ReAct"},       # 已存在
        {"id": "2302.04761", "name": "Toolformer"},  # 需下载
    ]
    paths = download_papers(papers_cfg)
    assert len(paths) == 2
    urlopen.assert_called_once()
    assert (papers / "Toolformer_2302.04761.pdf").read_bytes() == b"%PDF-NEW"


def test_download_papers_creates_papers_dir(isolated_papers_dir, monkeypatch):
    """PAPERS_DIR 不存在时会被自动创建。"""
    papers, _ = isolated_papers_dir
    assert not papers.exists()
    monkeypatch.setattr("urllib.request.Request", lambda url, headers=None: ("REQ", url))
    monkeypatch.setattr("urllib.request.urlopen", MagicMock(return_value=_FakeResp(b"%PDF")))
    download_papers([{"id": "1", "name": "N"}])
    assert papers.is_dir()


# ===================================================================== #
# indexing.py: build_index / load_index (mock FAISS + 工厂, 不嵌入/不联网)
# ===================================================================== #
def _patch_indexing_pipeline(monkeypatch, tmp_path):
    """把 build_index 的外部依赖全部换成 Mock, 仅保留默认值解析逻辑。"""
    paths = [tmp_path / "ReAct_2210.03629.pdf"]
    docs = [Document(page_content="page content", metadata={"source": "ReAct", "page": 1})]

    mk_download = MagicMock(return_value=paths)
    mk_parse = MagicMock(return_value=docs)
    mk_splitter = MagicMock()
    mk_splitter.return_value.split_documents.return_value = [
        Document(page_content="chunk", metadata={"source": "ReAct", "page": 1})
    ]
    mk_embedder = MagicMock(name="embedder")
    mk_get_embedder = MagicMock(return_value=mk_embedder)

    fake_faiss = MagicMock(name="FAISS")
    fake_index = MagicMock(name="index")
    fake_faiss.from_documents.return_value = fake_index

    monkeypatch.setattr("rag_agent.indexing.download_papers", mk_download)
    monkeypatch.setattr("rag_agent.indexing.parse_pdf", mk_parse)
    monkeypatch.setattr("rag_agent.indexing.make_splitter", mk_splitter)
    monkeypatch.setattr("rag_agent.indexing.get_embedder", mk_get_embedder)
    monkeypatch.setattr("rag_agent.indexing.FAISS", fake_faiss)
    return mk_download, mk_parse, mk_splitter, mk_embedder, fake_faiss, fake_index


def test_build_index_uses_cfg_defaults(monkeypatch, tmp_path, isolated_papers_dir):
    """不传参时 build_index 用 cfg['chunking']['default'] 调 make_splitter。"""
    _, _, mk_splitter, mk_embedder, fake_faiss, fake_index = _patch_indexing_pipeline(
        monkeypatch, tmp_path
    )
    _, index_dir = isolated_papers_dir

    result = build_index(tag="unit")

    ch = cfg["chunking"]["default"]
    mk_splitter.assert_called_once_with(ch["strategy"], ch["chunk_size"], ch["chunk_overlap"])
    # embedder 传给 FAISS.from_documents
    fake_faiss.from_documents.assert_called_once()
    assert fake_faiss.from_documents.call_args.args[1] is mk_embedder
    # 索引按 tag 持久化到 INDEX_DIR/tag
    fake_index.save_local.assert_called_once_with(str(index_dir / "unit"))
    assert result is fake_index


def test_build_index_accepts_overrides(monkeypatch, tmp_path, isolated_papers_dir):
    """显式参数覆盖 cfg 默认值并透传给 make_splitter。"""
    _, _, mk_splitter, _, _, _ = _patch_indexing_pipeline(monkeypatch, tmp_path)

    build_index(strategy="fixed", chunk_size=200, chunk_overlap=20, tag="custom")

    mk_splitter.assert_called_once_with("fixed", 200, 20)


def test_load_index_passes_path_and_unsafe_deserialize(monkeypatch, isolated_papers_dir):
    """load_index 用 INDEX_DIR/tag 路径 + allow_dangerous_deserialization=True。"""
    mk_embedder = MagicMock(name="embedder")
    monkeypatch.setattr("rag_agent.indexing.get_embedder", MagicMock(return_value=mk_embedder))
    fake_faiss = MagicMock(name="FAISS")
    fake_index = MagicMock(name="index")
    fake_faiss.load_local.return_value = fake_index
    monkeypatch.setattr("rag_agent.indexing.FAISS", fake_faiss)
    _, index_dir = isolated_papers_dir

    result = load_index("mytag")

    fake_faiss.load_local.assert_called_once_with(
        str(index_dir / "mytag"), mk_embedder, allow_dangerous_deserialization=True
    )
    assert result is fake_index


# ===================================================================== #
# retrieval.py: make_retriever 各分支 (fake index, 不嵌入)
# ===================================================================== #
def _fake_index(docs):
    """构造 fake vectorstore: as_retriever() 返回可记录 invoke 的 base retriever。"""
    index = MagicMock(name="index")
    base = MagicMock(name="base_retriever")
    base.invoke.return_value = list(docs)
    index.as_retriever.return_value = base
    return index, base


def test_make_retriever_none_returns_base():
    index, base = _fake_index([Document(page_content="a")])
    r = make_retriever(index, advanced="none")
    assert r is base
    index.as_retriever.assert_called_once_with(search_kwargs={"k": cfg["retrieval"]["top_k"]})


def test_make_retriever_rerank_falls_back_to_base():
    """rerank 尚未实现, 回退基础检索。"""
    index, base = _fake_index([Document(page_content="a")])
    assert make_retriever(index, advanced="rerank") is base


def test_make_retriever_unknown_strategy_returns_base():
    index, base = _fake_index([Document(page_content="a")])
    assert make_retriever(index, advanced="bogus") is base


def test_make_retriever_top_k_override():
    index, base = _fake_index([Document(page_content="a")])
    make_retriever(index, top_k=7, advanced="none")
    index.as_retriever.assert_called_once_with(search_kwargs={"k": 7})


def test_make_retriever_hyde_returns_runnable(monkeypatch):
    """hyde 分支返回 RunnableLambda, 不在构造时调 LLM。"""
    monkeypatch.setattr("rag_agent.retrieval._hyde_generate", lambda q: "HYDE DOC")
    index, _ = _fake_index([Document(page_content="a")])
    r = make_retriever(index, advanced="hyde")
    assert isinstance(r, RunnableLambda)


def test_make_retriever_hyde_invokes_base_and_tags_docs(monkeypatch):
    """hyde retriever 用假设文档检索, 并在结果 metadata 标注 retrieval=hyde。"""
    monkeypatch.setattr("rag_agent.retrieval._hyde_generate", lambda q: "HYDE DOC")
    docs = [Document(page_content="c1", metadata={}), Document(page_content="c2", metadata={})]
    index, base = _fake_index(docs)

    retriever = make_retriever(index, advanced="hyde")
    out = retriever.invoke("what is react?")

    base.invoke.assert_called_once_with("HYDE DOC")
    assert [d.metadata.get("retrieval") for d in out] == ["hyde", "hyde"]


def test_make_retriever_hyde_does_not_mutate_source_docs(monkeypatch):
    """hyde 分支返回拷贝, 不应污染底层 docstore 共享对象的 metadata。"""
    monkeypatch.setattr("rag_agent.retrieval._hyde_generate", lambda q: "HYDE DOC")
    docs = [Document(page_content="c1", metadata={"source": "ReAct", "page": 1})]
    index, base = _fake_index(docs)

    retriever = make_retriever(index, advanced="hyde")
    out = retriever.invoke("q")

    # 输出是带标记的拷贝
    assert out[0].metadata["retrieval"] == "hyde"
    # 但 base 返回的原始 doc(模拟 docstore 共享对象)不应被原地改动
    assert "retrieval" not in docs[0].metadata
    # 且返回的应是新对象, 而非同一引用
    assert out[0] is not docs[0]


def test_make_retriever_default_uses_cfg_advanced(monkeypatch):
    """不传 advanced 时取 cfg['retrieval']['advanced'] (默认 hyde)。"""
    monkeypatch.setattr("rag_agent.retrieval._hyde_generate", lambda q: "HYDE")
    index, _ = _fake_index([Document(page_content="a")])
    r = make_retriever(index)
    # cfg 默认 advanced=hyde -> RunnableLambda
    assert isinstance(r, RunnableLambda)
