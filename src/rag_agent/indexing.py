"""数据加载 + chunking + FAISS 索引构建。

流程: arxiv 拉固定论文 PDF -> pymupdf 解析(带页码 metadata) -> 切分 -> 嵌入 -> FAISS 持久化。

chunking 支持三种策略(fixed / recursive / markdown), 评估时可对比。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import fitz  # pymupdf
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .config import PROJECT_ROOT, cfg
from .models import get_embedder

PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
INDEX_DIR = PROJECT_ROOT / "data" / "index"


# ---------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------
def download_papers(papers: List[dict]) -> List[Path]:
    """下载论文 PDF 到 data/papers, 已存在则跳过。

    直接用 arxiv id 构造 PDF URL (https://arxiv.org/pdf/<id>) + urllib 下载,
    不依赖 arxiv 库的 Result API (arxiv 4.x 已移除 download_pdf / pdf_url)。
    """
    import urllib.request

    PAPERS_DIR.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for p in papers:
        pdf_path = PAPERS_DIR / f"{p['name']}_{p['id']}.pdf"
        if pdf_path.exists():
            paths.append(pdf_path)
            continue
        pdf_url = f"https://arxiv.org/pdf/{p['id']}"
        req = urllib.request.Request(
            pdf_url, headers={"User-Agent": "rag-agent-showcase/0.1"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        # 校验响应确实是 PDF: arxiv 限流/出错时可能返回 200 + HTML
        if not data.startswith(b"%PDF"):
            raise RuntimeError(
                f"下载 {p['name']} 失败: 响应非 PDF (起始字节 {data[:8]!r})"
            )
        # 原子落盘: 先写 .tmp 再 replace, 避免半截文件被当作有效缓存
        tmp_path = pdf_path.with_name(pdf_path.name + ".tmp")
        try:
            with open(tmp_path, "wb") as f:
                f.write(data)
            tmp_path.replace(pdf_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        print(f"[indexing] 下载 {p['name']} -> {pdf_path.name}")
        paths.append(pdf_path)
    return paths


def parse_pdf(pdf_path: Path, paper_name: str) -> List[Document]:
    """用 pymupdf 解析 PDF, 每页一个 Document, 带来源/页码 metadata。"""
    docs: List[Document] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": paper_name, "page": i + 1},
                    )
                )
    return docs


# ---------------------------------------------------------------------
# chunking 策略
# ---------------------------------------------------------------------
def make_splitter(strategy: str, chunk_size: int, chunk_overlap: int):
    """根据策略返回不同的 text splitter。

    - fixed:     按段落定长切, 最朴素基线
    - recursive: 递归字符切分, 默认生产配置
    - markdown:  递归切分但优先按标题结构, 适合结构化文档
    """
    if strategy == "fixed":
        return CharacterTextSplitter(
            separator="\n\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    if strategy == "markdown":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )
    # recursive (默认)
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ---------------------------------------------------------------------
# 索引构建 / 加载
# ---------------------------------------------------------------------
def build_index(
    strategy: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    tag: str = "default",
) -> FAISS:
    """完整流程: 下载 -> 解析 -> 切分 -> 建索引 -> 持久化。"""
    ch = cfg["chunking"]["default"]
    # 用 `is not None` 而非 `or`: 否则合法的 chunk_overlap=0 会被当 falsy 吞掉
    strategy = strategy if strategy is not None else ch["strategy"]
    chunk_size = chunk_size if chunk_size is not None else ch["chunk_size"]
    chunk_overlap = chunk_overlap if chunk_overlap is not None else ch["chunk_overlap"]

    paths = download_papers(cfg["dataset"]["papers"])
    all_docs: List[Document] = []
    for path in paths:
        # 按最后一个下划线切: 只去掉末尾的 arxiv id, 保留含下划线的论文名
        name = path.stem.rsplit("_", 1)[0]
        all_docs.extend(parse_pdf(path, name))

    splitter = make_splitter(strategy, chunk_size, chunk_overlap)
    chunks = splitter.split_documents(all_docs)
    assert chunks, "切分后无任何 chunk, 请检查 PDF 下载/解析是否成功"

    embedder = get_embedder()
    index = FAISS.from_documents(chunks, embedder)

    index_path = INDEX_DIR / tag
    index_path.mkdir(parents=True, exist_ok=True)
    index.save_local(str(index_path))
    print(
        f"[indexing] tag={tag} strategy={strategy} "
        f"{len(all_docs)} 页 -> {len(chunks)} chunks -> {index_path}"
    )
    return index


def load_index(tag: str = "default") -> FAISS:
    embedder = get_embedder()
    return FAISS.load_local(
        str(INDEX_DIR / tag), embedder, allow_dangerous_deserialization=True
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="构建 FAISS 索引")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--strategy", default=None, choices=["fixed", "recursive", "markdown"])
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--chunk-overlap", type=int, default=None)
    args = ap.parse_args()
    build_index(args.strategy, args.chunk_size, args.chunk_overlap, args.tag)


if __name__ == "__main__":
    main()
