"""RAG 链: retriever | docs2str | prompt | llm | strip-think。

strip_think 用于剥离 minicpm5 的 <think>...</think> 思考标签, 避免污染答案与评估。
"""
from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from .models import get_llm
from .retrieval import make_retriever

THINK_RE = re.compile(r"<think>.*?(?:</think>|\Z)", re.DOTALL)


def strip_think(text: str) -> str:
    """剥离 minicpm5 的 <think>...</think> 思考标签。

    未闭合(被 max_tokens 截断)的 <think> 一并剥离到串尾, 避免思考内容泄漏到答案。
    """
    return THINK_RE.sub("", text).strip()


RAG_PROMPT = ChatPromptTemplate.from_template(
    "你是一个文档问答助手。仅根据下面检索到的文档回答问题。"
    "如果文档中没有答案, 请说明“根据已有文档无法回答”。"
    "引用来源时用 [来源:p页码] 格式(如 [ReAct:p3]), 与下方文档标注保持一致。\n\n"
    "检索文档:\n{context}\n\n"
    "问题: {question}\n\n"
    "回答:"
)


def _docs2str(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        parts.append(f"[{src}:p{page}] {d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(index, advanced: str | None = None) -> Runnable:
    """构建 RAG LCEL 链。输入 question(str), 输出 answer(str)。"""
    retriever = make_retriever(index, advanced=advanced)
    llm = get_llm()
    return (
        {
            "context": retriever | RunnableLambda(_docs2str),
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
        | RunnableLambda(strip_think)
    )
