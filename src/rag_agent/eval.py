"""Ragas 评估: 生成 QA 对 -> 跑 baseline(朴素) vs 改进(HyDE) -> 四指标对比。

judge 模型与 generator 同为 minicpm5:1b, 存在自评偏差(README 已说明)。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import types
from pathlib import Path

# ---- stub ragas 0.4.3 硬 import 的已移除 community 模块(我们用 ChatOpenAI, 不用 VertexAI) ----
_vmod = types.ModuleType("langchain_community.chat_models.vertexai")
_vmod.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vmod)
import langchain_community.llms as _llms_pkg  # noqa: E402

if not hasattr(_llms_pkg, "VertexAI"):
    _llms_pkg.VertexAI = type("VertexAI", (), {})

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnableLambda  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.dataset_schema import EvaluationDataset  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from .chain import RAG_PROMPT, _docs2str, strip_think  # noqa: E402
from .config import PROJECT_ROOT, cfg  # noqa: E402
from .indexing import load_index  # noqa: E402
from .models import get_embedder, get_llm  # noqa: E402
from .retrieval import make_retriever  # noqa: E402

EVALS_DIR = PROJECT_ROOT / "evals"
QA_PATH = EVALS_DIR / "qa_pairs.json"
RESULTS_DIR = PROJECT_ROOT / cfg["eval"]["output_dir"]
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

QA_GEN_PROMPT = ChatPromptTemplate.from_template(
    "根据以下文档内容, 生成一个口语化、简短的提问(像普通用户随口问的, 不要照搬论文术语), 并给出参考答案(答案必须来自文档)。\n"
    "问题要短、口语化, 和文档的学术表达差异大(例如用日常用语而非论文原话), 这样能测试检索对'非文档式提问'的适应力。\n"
    "严格按格式输出, 不要多余内容:\n"
    "问题: <口语化短问题>\n"
    "答案: <来自文档的参考答案>\n\n"
    "文档:\n{context}"
)


# ---------------------------------------------------------------------
# QA 生成
# ---------------------------------------------------------------------
def generate_qa(num_questions: int) -> list[dict]:
    """从索引 chunks 随机采样, 用 LLM 合成 QA 对。"""
    index = load_index("default")
    docs = list(index.docstore._dict.values())
    random.seed(42)
    llm = get_llm(max_tokens=300)
    chain = QA_GEN_PROMPT | llm | StrOutputParser() | RunnableLambda(strip_think)

    qa: list[dict] = []
    attempts = 0
    while len(qa) < num_questions and attempts < num_questions * 4:
        doc = random.choice(docs)
        out = chain.invoke({"context": doc.page_content})
        if "问题:" in out and "答案:" in out:
            q = out.split("问题:")[1].split("答案:")[0].strip()
            a = out.split("答案:")[1].strip()
            if q and a and len(q) > 5:
                qa.append(
                    {
                        "question": q,
                        "ground_truth": a,
                        "source": doc.metadata.get("source", "?"),
                    }
                )
        attempts += 1
    return qa


# ---------------------------------------------------------------------
# 单题问答(一次检索, contexts 与 answer 用同一批 docs)
# ---------------------------------------------------------------------
def answer_question(question: str, retriever, llm):
    docs = retriever.invoke(question)
    prompt_val = RAG_PROMPT.format(context=_docs2str(docs), question=question)
    raw = llm.invoke(prompt_val).content
    return docs, strip_think(raw)


def run_batch(qa_pairs: list[dict], advanced: str) -> list[dict]:
    """对 QA 集跑 RAG, 返回 Ragas 所需记录。"""
    retriever = make_retriever(load_index("default"), advanced=advanced)
    llm = get_llm()
    records = []
    for i, qa in enumerate(qa_pairs, 1):
        docs, answer = answer_question(qa["question"], retriever, llm)
        records.append(
            {
                "user_input": qa["question"],
                "response": answer,
                "retrieved_contexts": [d.page_content for d in docs],
                "reference": qa["ground_truth"],
            }
        )
        print(f"  [{advanced}] {i}/{len(qa_pairs)} done")
    return records


def eval_batch(records: list[dict]) -> dict:
    """跑 Ragas 四指标, 返回 {metric: mean}。"""
    evaluator_llm = LangchainLLMWrapper(get_llm())
    evaluator_emb = LangchainEmbeddingsWrapper(get_embedder())
    dataset = EvaluationDataset.from_list(records)
    result = evaluate(dataset, metrics=METRICS, llm=evaluator_llm, embeddings=evaluator_emb)
    df = result.to_pandas()
    return {m: float(df[m].mean()) for m in METRIC_NAMES if m in df.columns}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Ragas 评估: baseline vs HyDE")
    ap.add_argument("--num", type=int, default=cfg["eval"]["num_questions"])
    ap.add_argument("--gen-only", action="store_true", help="只生成 QA 不评估")
    args = ap.parse_args()

    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. QA
    if QA_PATH.exists():
        qa_pairs = json.loads(QA_PATH.read_text(encoding="utf-8"))
        print(f"[eval] 加载已有 QA: {len(qa_pairs)} 条")
    else:
        print(f"[eval] 生成 {args.num} 个 QA 对...")
        qa_pairs = generate_qa(args.num)
        QA_PATH.write_text(json.dumps(qa_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] QA 保存到 {QA_PATH} ({len(qa_pairs)} 条)")

    if args.gen_only:
        for qa in qa_pairs[:5]:
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['ground_truth'][:80]}")
        return

    # 2. baseline(none) vs 改进(hyde)
    summary = {}
    for advanced, tag in [("none", "baseline"), ("hyde", "hyde")]:
        print(f"\n[eval] === 跑 {tag} (advanced={advanced}) ===")
        for attempt in range(1, 4):
            try:
                records = run_batch(qa_pairs, advanced)
                (RESULTS_DIR / f"{tag}_records.json").write_text(
                    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"[eval] 评估 {tag} 指标...")
                scores = eval_batch(records)
                summary[tag] = scores
                print(f"[eval] {tag}: {scores}")
                break
            except Exception as e:
                # 重试渡过临时网络错误(如 DeepSeek APIConnectionError); 3 次仍败才跳过
                print(f"[eval] {tag} 第 {attempt}/3 次失败: {type(e).__name__}: {e}")
                if attempt == 3:
                    print(f"[eval] {tag} 重试耗尽, 跳过该 tag")

    # 3. 对比表(缺某一项时给出明确提示而非抛 KeyError)
    base = summary.get("baseline", {})
    hyde = summary.get("hyde", {})
    if not base and not hyde:
        print("[eval] baseline 与 hyde 均失败, 无可对比结果")
        return
    print("\n" + "=" * 60)
    print(f"{'指标':<22}{'baseline(朴素)':<18}{'hyde(改进)':<18}{'delta':<10}")
    print("-" * 60)
    for m in METRIC_NAMES:
        b = base.get(m, 0)
        h = hyde.get(m, 0)
        print(f"{m:<22}{b:<18.3f}{h:<18.3f}{h - b:+.3f}")
    print("=" * 60)

    # 保存汇总
    with open(RESULTS_DIR / "summary.csv", "w", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "baseline", "hyde", "delta"])
        for m in METRIC_NAMES:
            b = base.get(m, 0)
            h = hyde.get(m, 0)
            w.writerow([m, f"{b:.3f}", f"{h:.3f}", f"{h - b:+.3f}"])
    print(f"\n[eval] 汇总保存到 {RESULTS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
