"""检索器: 基础 FAISS 检索 + HyDE (假设文档嵌入) 进阶检索。

HyDE 思路: 先用 LLM 对问题生成一段"假设性答案文档", 再用该文档的 embedding 去检索,
从而把"问题语义"对齐到"文档语义"空间, 缓解 query-document 表达差异。
"""
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from .config import cfg
from .models import get_llm

HYDE_PROMPT = ChatPromptTemplate.from_template(
    "请根据以下问题, 写一段能回答该问题的假设性文档(150字以内)。"
    "不要直接回答问题, 只写一段看起来像包含答案的相关文本。\n\n"
    "问题: {question}\n\n假设文档:"
)


def _hyde_generate(question: str) -> str:
    """用 LLM 生成假设文档。"""
    llm = get_llm(temperature=0.0, max_tokens=200)
    chain = HYDE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"question": question})


def make_retriever(index, top_k: int | None = None, advanced: str | None = None) -> Runnable:
    """返回 Runnable: question(str) -> List[Document]。

    advanced:
      - none:  朴素检索(基线)
      - hyde:  假设文档嵌入检索
      - rerank: 预留(后续接 bge-reranker)
    """
    top_k = top_k or cfg["retrieval"]["top_k"]
    advanced = advanced or cfg["retrieval"]["advanced"]
    base = index.as_retriever(search_kwargs={"k": top_k})

    if advanced in ("none", "rerank"):
        # rerank 暂回退基础检索, 留待后续实现
        return base

    if advanced == "hyde":

        def retrieve_with_hyde(question: str) -> List[Document]:
            hyde_doc = _hyde_generate(question)
            docs = base.invoke(hyde_doc)
            # 返回拷贝并标注检索方式: FAISS retriever 返回的是 docstore 内对象引用,
            # 原地改 metadata 会污染底层索引(尤其在长驻服务里累积污染)。
            return [
                Document(
                    page_content=d.page_content,
                    metadata={**d.metadata, "retrieval": "hyde"},
                )
                for d in docs
            ]

        return RunnableLambda(retrieve_with_hyde)

    return base
