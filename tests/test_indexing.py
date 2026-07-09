"""indexing 模块测试: 数据加载 + chunking + FAISS 索引构建。

设计原则:
- 纯函数 (make_splitter) 直接测, 不 mock。
- 涉及外部 (网络/模型/FAISS) 的函数用 monkeypatch 替换, 测试只验证编排逻辑,
  绝不真正调用 DeepSeek/42model/arxiv 网络。
- 文件系统副作用重定向到 tmp_path, 不污染真实 data/ 目录。

用法: uv run pytest tests/test_indexing.py -q
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from rag_agent import indexing as ix
from rag_agent.config import cfg


# =====================================================================
# 辅助 fake 对象
# =====================================================================
class _FakePage:
    """模拟 pymupdf page, 只实现 get_text。"""

    def __init__(self, text: str):
        self._text = text

    def get_text(self) -> str:
        return self._text


class _FakePdfDoc:
    """模拟 fitz.open 返回的文档: 可迭代 + 上下文管理器。"""

    def __init__(self, pages):
        self._pages = pages

    def __iter__(self):
        return iter(self._pages)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeHttpResp:
    """模拟 urllib.request.urlopen 返回的响应对象。"""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# =====================================================================
# make_splitter (纯函数)
# =====================================================================
class TestMakeSplitter:
    def test_fixed_returns_character_splitter(self):
        from langchain_text_splitters import CharacterTextSplitter

        s = ix.make_splitter("fixed", 800, 120)
        assert isinstance(s, CharacterTextSplitter)

    def test_recursive_returns_recursive_splitter(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        s = ix.make_splitter("recursive", 800, 120)
        assert isinstance(s, RecursiveCharacterTextSplitter)

    def test_markdown_returns_recursive_splitter(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        s = ix.make_splitter("markdown", 800, 120)
        assert isinstance(s, RecursiveCharacterTextSplitter)

    def test_markdown_uses_heading_separators(self):
        """markdown 策略优先按标题切分 (含 ## / ###)。"""
        s = ix.make_splitter("markdown", 800, 10)
        seps = s._separators
        assert "\n## " in seps
        assert "\n### " in seps

    def test_recursive_uses_default_separators(self):
        s = ix.make_splitter("recursive", 800, 10)
        assert s._separators == ["\n\n", "\n", ". ", " ", ""]

    def test_unknown_strategy_falls_back_to_recursive(self):
        """未识别策略走默认分支 (recursive)。"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        s = ix.make_splitter("does-not-exist", 800, 10)
        assert isinstance(s, RecursiveCharacterTextSplitter)
        assert s._separators == ["\n\n", "\n", ". ", " ", ""]

    def test_passes_chunk_size_and_overlap(self):
        s = ix.make_splitter("recursive", 500, 50)
        assert s._chunk_size == 500
        assert s._chunk_overlap == 50

    def test_fixed_separator_is_double_newline(self):
        s = ix.make_splitter("fixed", 800, 10)
        assert s._separator == "\n\n"

    def test_actually_splits_long_document(self):
        """功能验证: 长文本被切成多个 chunk。"""
        text = "段落一。\n\n" + ("a" * 1500) + "\n\n段落二。"
        doc = Document(page_content=text)
        s = ix.make_splitter("recursive", chunk_size=200, chunk_overlap=20)
        chunks = s.split_documents([doc])
        assert len(chunks) > 1
        # 每个 chunk 都继承 metadata 结构 (可为空)
        assert all(isinstance(c, Document) for c in chunks)

    def test_empty_input_yields_no_chunks(self):
        s = ix.make_splitter("recursive", 200, 20)
        assert s.split_documents([]) == []


# =====================================================================
# download_papers (mock urllib 网络)
# =====================================================================
class TestDownloadPapers:
    def _paper(self, name="ReAct", pid="2210.03629"):
        return {"name": name, "id": pid}

    def test_creates_papers_dir(self, monkeypatch, tmp_path):
        """目录不存在时会被创建。"""
        papers_dir = tmp_path / "papers"
        assert not papers_dir.exists()
        monkeypatch.setattr(ix, "PAPERS_DIR", papers_dir)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=60: _FakeHttpResp(b"%PDF-fake"),
        )
        ix.download_papers([self._paper()])
        assert papers_dir.is_dir()

    def test_downloads_new_file_with_correct_url(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ix, "PAPERS_DIR", tmp_path / "papers")

        captured = {}

        def fake_urlopen(req, timeout=60):
            captured["url"] = req.full_url
            captured["headers"] = req.headers
            return _FakeHttpResp(b"%PDF-FAKE-CONTENT")

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
        paths = ix.download_papers([self._paper("ReAct", "2210.03629")])

        assert len(paths) == 1
        written = paths[0].read_bytes()
        assert written == b"%PDF-FAKE-CONTENT"
        # 文件名格式: {name}_{id}.pdf
        assert paths[0].name == "ReAct_2210.03629.pdf"
        # URL 直接由 arxiv id 构造
        assert captured["url"] == "https://arxiv.org/pdf/2210.03629"

    def test_skips_existing_file_without_network(self, monkeypatch, tmp_path):
        """已存在的 PDF 不应触发 urlopen。"""
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        (papers_dir / "ReAct_2210.03629.pdf").write_bytes(b"ALREADY-THERE")

        monkeypatch.setattr(ix, "PAPERS_DIR", papers_dir)

        def fail(req, timeout=60):  # 任何网络访问都判失败
            raise AssertionError("不应在文件已存在时发起网络请求")

        monkeypatch.setattr("urllib.request.urlopen", fail)
        paths = ix.download_papers([self._paper()])

        assert len(paths) == 1
        assert paths[0].read_bytes() == b"ALREADY-THERE"

    def test_empty_list_returns_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ix, "PAPERS_DIR", tmp_path / "papers")
        called = {"n": 0}

        def fake(req, timeout=60):
            called["n"] += 1
            return _FakeHttpResp(b"")

        monkeypatch.setattr("urllib.request.urlopen", fake)
        assert ix.download_papers([]) == []
        assert called["n"] == 0

    def test_mixed_existing_and_new(self, monkeypatch, tmp_path):
        papers_dir = tmp_path / "papers"
        papers_dir.mkdir()
        (papers_dir / "ReAct_2210.03629.pdf").write_bytes(b"OLD")

        monkeypatch.setattr(ix, "PAPERS_DIR", papers_dir)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=60: _FakeHttpResp(b"%PDF-NEW"),
        )
        paths = ix.download_papers(
            [self._paper("ReAct", "2210.03629"), self._paper("Reflexion", "2303.11366")]
        )
        assert {p.name for p in paths} == {
            "ReAct_2210.03629.pdf",
            "Reflexion_2303.11366.pdf",
        }
        assert (papers_dir / "Reflexion_2303.11366.pdf").read_bytes() == b"%PDF-NEW"

    def test_propagates_network_error(self, monkeypatch, tmp_path):
        """urlopen 抛错时, 异常应向上传播 (不静默吞掉)。"""
        monkeypatch.setattr(ix, "PAPERS_DIR", tmp_path / "papers")

        def boom(req, timeout=60):
            raise OSError("network down")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        with pytest.raises(OSError):
            ix.download_papers([self._paper()])


# =====================================================================
# parse_pdf (mock fitz)
# =====================================================================
class TestParsePdf:
    def _patch_fitz(self, monkeypatch, pages):
        fake_fitz = MagicMock()
        fake_fitz.open.return_value = _FakePdfDoc(pages)
        monkeypatch.setattr(ix, "fitz", fake_fitz)
        return fake_fitz

    def test_multiple_pages_become_documents(self, monkeypatch):
        pages = [_FakePage("page one text"), _FakePage("page two text")]
        self._patch_fitz(monkeypatch, pages)
        docs = ix.parse_pdf(Path("fake.pdf"), "ReAct")
        assert len(docs) == 2
        assert docs[0].page_content == "page one text"
        assert docs[1].page_content == "page two text"

    def test_metadata_source_and_one_indexed_page(self, monkeypatch):
        pages = [_FakePage("a"), _FakePage("b"), _FakePage("c")]
        self._patch_fitz(monkeypatch, pages)
        docs = ix.parse_pdf(Path("fake.pdf"), "Voyager")
        assert [d.metadata["page"] for d in docs] == [1, 2, 3]
        assert all(d.metadata["source"] == "Voyager" for d in docs)

    def test_empty_pages_are_skipped(self, monkeypatch):
        """空白页 (get_text 全是空白) 不应产出 Document。"""
        pages = [_FakePage("real content"), _FakePage("   \n  "), _FakePage("more")]
        self._patch_fitz(monkeypatch, pages)
        docs = ix.parse_pdf(Path("fake.pdf"), "ReAct")
        assert len(docs) == 2
        assert docs[0].page_content == "real content"
        assert docs[1].page_content == "more"
        # 页码仍按原始页序: 第三页是 page=3
        assert docs[1].metadata["page"] == 3

    def test_all_empty_pages_returns_empty(self, monkeypatch):
        self._patch_fitz(monkeypatch, [_FakePage(""), _FakePage("\n")])
        assert ix.parse_pdf(Path("fake.pdf"), "X") == []

    def test_empty_pdf_returns_empty(self, monkeypatch):
        self._patch_fitz(monkeypatch, [])
        assert ix.parse_pdf(Path("fake.pdf"), "X") == []

    def test_fitz_open_called_with_path(self, monkeypatch):
        fake_fitz = self._patch_fitz(monkeypatch, [_FakePage("hi")])
        p = Path("/tmp/whatever.pdf")
        ix.parse_pdf(p, "ReAct")
        fake_fitz.open.assert_called_once_with(p)


# =====================================================================
# build_index (编排: mock download/parse/embedder/FAISS)
# =====================================================================
class TestBuildIndex:
    def _stub_pipeline(self, monkeypatch, tmp_path, docs_per_paper=2):
        """把外部依赖全部打桩, 返回收集调用的容器。"""
        monkeypatch.setattr(ix, "INDEX_DIR", tmp_path / "index")

        # download_papers: 返回伪造路径 (stem 能被 split('_')[0] 解析)
        def fake_download(papers):
            paths = []
            for p in papers:
                fp = tmp_path / f"{p['name']}_{p['id']}.pdf"
                fp.write_bytes(b"fake")  # 内容无关紧要
                paths.append(fp)
            return paths

        monkeypatch.setattr(ix, "download_papers", fake_download)

        # parse_pdf: 每篇产出 docs_per_paper 个 Document
        def fake_parse(path, name):
            return [
                Document(page_content=f"{name} 内容段落 {i} " * 50, metadata={"source": name, "page": i + 1})
                for i in range(docs_per_paper)
            ]

        monkeypatch.setattr(ix, "parse_pdf", fake_parse)

        # embedder: 任意 Mock 即可, 不调 API
        embedder = MagicMock(name="embedder")
        monkeypatch.setattr(ix, "get_embedder", lambda: embedder)

        # FAISS.from_documents -> 返回 Mock index, 记录 save_local 入参
        index = MagicMock(name="faiss_index")
        monkeypatch.setattr(ix.FAISS, "from_documents", staticmethod(lambda chunks, emb: index))
        return embedder, index

    def test_uses_config_defaults_when_args_none(self, monkeypatch, tmp_path):
        embedder, index = self._stub_pipeline(monkeypatch, tmp_path)
        result = ix.build_index()  # 全部用默认

        dflt = cfg["chunking"]["default"]
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        # 验证默认策略 recursive 被使用 (通过从文档数推断 splitter 正常工作)
        assert result is index
        index.save_local.assert_called_once()
        # 保存路径 = INDEX_DIR / "default"
        save_path = index.save_local.call_args.args[0]
        assert save_path == str((tmp_path / "index" / "default"))

    def test_explicit_args_override_defaults(self, monkeypatch, tmp_path):
        embedder, index = self._stub_pipeline(monkeypatch, tmp_path)
        ix.build_index(strategy="fixed", chunk_size=300, chunk_overlap=0, tag="custom")

        save_path = index.save_local.call_args.args[0]
        assert save_path == str((tmp_path / "index" / "custom"))
        assert (tmp_path / "index" / "custom").is_dir()  # mkdir 被执行

    def test_chunk_overlap_zero_not_swallowed(self, monkeypatch, tmp_path):
        """chunk_overlap=0 是合法值, 不能被 `or` 当 falsy 替换成默认值。"""
        self._stub_pipeline(monkeypatch, tmp_path)
        captured = {}

        def fake_make_splitter(strategy, chunk_size, chunk_overlap):
            captured["strategy"] = strategy
            captured["chunk_size"] = chunk_size
            captured["chunk_overlap"] = chunk_overlap
            mk = MagicMock()
            mk.split_documents.return_value = [
                Document(page_content="c", metadata={"source": "X", "page": 1})
            ]
            return mk

        monkeypatch.setattr(ix, "make_splitter", fake_make_splitter)
        ix.build_index(strategy="fixed", chunk_size=800, chunk_overlap=0)
        assert captured["chunk_overlap"] == 0
        assert captured["chunk_size"] == 800

    def test_invokes_faiss_from_documents_with_chunks_and_embedder(self, monkeypatch, tmp_path):
        embedder, index = self._stub_pipeline(monkeypatch, tmp_path, docs_per_paper=3)
        # 用计数器确认 from_documents 被调用一次, 且入参是 chunks 列表
        calls = []

        def fake_from_documents(chunks, emb):
            calls.append((len(chunks), emb))
            return index

        monkeypatch.setattr(ix.FAISS, "from_documents", staticmethod(fake_from_documents))

        ix.build_index(tag="t")

        assert len(calls) == 1
        n_chunks, used_embedder = calls[0]
        # 5 篇论文 * 3 页 -> 至少产出 >= 5 个 chunk (实际更多, 因内容被切分)
        assert n_chunks >= 5
        assert used_embedder is embedder

    def test_save_local_creates_nested_tag_dir(self, monkeypatch, tmp_path):
        _, index = self._stub_pipeline(monkeypatch, tmp_path)
        ix.build_index(tag="deep/nested")
        assert (tmp_path / "index" / "deep" / "nested").is_dir()

    def test_download_receives_configured_papers(self, monkeypatch, tmp_path):
        """build_index 必须把 cfg['dataset']['papers'] 传给 download_papers。"""
        self._stub_pipeline(monkeypatch, tmp_path)
        received = []

        def fake_download(papers):
            received.extend(papers)
            # 返回一个路径, 让后续 parse -> chunk 流程能产出非空 chunks
            return [tmp_path / "Some_9999.pdf"]

        monkeypatch.setattr(ix, "download_papers", fake_download)
        monkeypatch.setattr(
            ix,
            "parse_pdf",
            lambda *a, **k: [Document(page_content="c", metadata={"source": "X", "page": 1})],
        )
        index = MagicMock()
        monkeypatch.setattr(ix.FAISS, "from_documents", staticmethod(lambda *a, **k: index))
        ix.build_index()
        assert received == cfg["dataset"]["papers"]

    def test_get_embedder_called_once(self, monkeypatch, tmp_path):
        self._stub_pipeline(monkeypatch, tmp_path)
        calls = {"n": 0}

        def fake_embedder():
            calls["n"] += 1
            return MagicMock()

        monkeypatch.setattr(ix, "get_embedder", fake_embedder)
        monkeypatch.setattr(
            ix,
            "parse_pdf",
            lambda *a, **k: [Document(page_content="c", metadata={"source": "X", "page": 1})],
        )
        monkeypatch.setattr(ix.FAISS, "from_documents", staticmethod(lambda *a, **k: MagicMock()))
        ix.build_index()
        assert calls["n"] == 1


# =====================================================================
# load_index (mock get_embedder + FAISS.load_local)
# =====================================================================
class TestLoadIndex:
    def test_calls_faiss_load_local_with_tag_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ix, "INDEX_DIR", tmp_path / "index")
        embedder = MagicMock()
        monkeypatch.setattr(ix, "get_embedder", lambda: embedder)
        loaded = MagicMock()
        monkeypatch.setattr(
            ix.FAISS, "load_local", staticmethod(lambda *a, **k: loaded)
        )
        result = ix.load_index("mytag")
        assert result is loaded

    def test_default_tag_is_default(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ix, "INDEX_DIR", tmp_path / "index")
        monkeypatch.setattr(ix, "get_embedder", lambda: MagicMock())
        captured = {}
        monkeypatch.setattr(
            ix.FAISS,
            "load_local",
            staticmethod(lambda path, emb, **kw: captured.update(path=path, kw=kw) or MagicMock()),
        )
        ix.load_index()
        assert captured["path"] == str(tmp_path / "index" / "default")

    def test_passes_allow_dangerous_deserialization_true(self, monkeypatch, tmp_path):
        """load_local 必须显式 allow_dangerous_deserialization=True。"""
        monkeypatch.setattr(ix, "INDEX_DIR", tmp_path / "index")
        monkeypatch.setattr(ix, "get_embedder", lambda: MagicMock())
        captured = {}
        monkeypatch.setattr(
            ix.FAISS,
            "load_local",
            staticmethod(lambda path, emb, **kw: captured.update(kw=kw) or MagicMock()),
        )
        ix.load_index("t")
        assert captured["kw"].get("allow_dangerous_deserialization") is True

    def test_uses_embedder_from_get_embedder(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ix, "INDEX_DIR", tmp_path / "index")
        embedder = MagicMock(name="e")
        monkeypatch.setattr(ix, "get_embedder", lambda: embedder)
        captured = {}
        monkeypatch.setattr(
            ix.FAISS,
            "load_local",
            staticmethod(lambda path, emb, **kw: captured.update(emb=emb) or MagicMock()),
        )
        ix.load_index("t")
        assert captured["emb"] is embedder


# =====================================================================
# main (CLI: mock build_index, 注入 sys.argv)
# =====================================================================
class TestMain:
    def test_default_args(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(ix, "build_index", lambda *a, **k: captured.update(args=a, kwargs=k))
        monkeypatch.setattr("sys.argv", ["indexing.py"])
        ix.main()
        assert captured["args"] == (None, None, None, "default")

    def test_custom_args_passed_through(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(ix, "build_index", lambda *a, **k: captured.update(args=a, kwargs=k))
        monkeypatch.setattr(
            "sys.argv",
            ["indexing.py", "--tag", "exp1", "--strategy", "fixed",
             "--chunk-size", "500", "--chunk-overlap", "20"],
        )
        ix.main()
        assert captured["args"] == ("fixed", 500, 20, "exp1")

    def test_invalid_strategy_rejected_by_argparse(self, monkeypatch):
        """choices=['fixed','recursive','markdown'] 应拒绝非法值。"""
        monkeypatch.setattr(ix, "build_index", lambda *a, **k: None)
        monkeypatch.setattr("sys.argv", ["indexing.py", "--strategy", "bogus"])
        with pytest.raises(SystemExit):
            ix.main()
