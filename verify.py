"""端到端冒烟测试: 验证后端连通 + RAG 链跑通。

用法: uv run python verify.py
"""
from rag_agent.chain import build_rag_chain
from rag_agent.config import cfg
from rag_agent.indexing import build_index
from rag_agent.models import get_embedder, get_llm


def _provider_info(section: str) -> str:
    p = cfg[section]["provider"]
    pc = cfg[section]["providers"][p]
    return f"{pc['model']} @ {pc['base_url']} (provider={p})"


def main():
    print("== 1. 配置 ==")
    print(f"  LLM: {_provider_info('llm')}")
    print(f"  Emb: {_provider_info('embedding')} ({cfg['embedding']['dimensions']}d)")

    print("\n== 2. 模型连通 ==")
    llm = get_llm(max_tokens=30)
    resp = llm.invoke("只回复'你好'两个字,不要其他内容")
    print(f"  LLM: {resp.content[:80]}")
    emb = get_embedder()
    vec = emb.embed_query("test")
    print(f"  Emb dim: {len(vec)} (期望 {cfg['embedding']['dimensions']})")

    print("\n== 3. 建索引(首次会下载论文) ==")
    index = build_index(tag="default")

    print("\n== 4. RAG 问答 ==")
    chain = build_rag_chain(index)
    q = "ReAct 方法是如何结合推理和行动的?"
    ans = chain.invoke(q)
    print(f"  Q: {q}")
    print(f"  A: {ans[:300]}")


if __name__ == "__main__":
    main()
