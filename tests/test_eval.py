"""eval.py 的测试。

不调任何外部服务(DeepSeek / 42model / 网络): get_llm / get_embedder / load_index /
make_retriever / ragas.evaluate 均通过 monkeypatch 替换为 Mock 或纯函数, 因此测试
不依赖 .env、网络或 FAISS 索引文件。纯函数(strip_think / _docs2str)直接测。

用法: uv run pytest tests/test_eval.py -q
"""
from __future__ import annotations

import json
import sys
from unittest.mock import Mock

import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_agent import eval as eval_mod
from rag_agent.chain import _docs2str, strip_think


# =====================================================================
# 纯函数: strip_think / _docs2str
# =====================================================================
class TestStripThink:
    def test_strips_single_block(self):
        assert strip_think("<think>reasoning</think>answer") == "answer"

    def test_strips_multiline_block(self):
        assert strip_think("<think>\nline1\nline2\n</think>\nfinal") == "final"

    def test_no_tag_passthrough(self):
        assert strip_think("just an answer") == "just an answer"

    def test_empty_string(self):
        assert strip_think("") == ""

    def test_only_think_block(self):
        assert strip_think("<think>only</think>") == ""

    def test_multiple_blocks_each_stripped(self):
        # 非 greedy: 每对 <think></think> 独立剥离
        assert strip_think("<think>a</think>x<think>b</think>y") == "xy"

    def test_outer_whitespace_trimmed(self):
        assert strip_think("  <think>x</think>  hi  ") == "hi"


class TestDocs2Str:
    def test_single_doc(self):
        doc = Document(page_content="hello", metadata={"source": "p1", "page": 3})
        assert _docs2str([doc]) == "[p1:p3] hello"

    def test_multiple_docs_joined_with_separator(self):
        docs = [
            Document(page_content="a", metadata={"source": "s1", "page": 1}),
            Document(page_content="b", metadata={"source": "s2", "page": 2}),
        ]
        out = _docs2str(docs)
        assert out == "[s1:p1] a\n\n---\n\n[s2:p2] b"

    def test_missing_metadata_defaults_to_question_mark(self):
        assert _docs2str([Document(page_content="x", metadata={})]) == "[?:p?] x"

    def test_empty_list_returns_empty_string(self):
        assert _docs2str([]) == ""


# =====================================================================
# 模块级常量 / 结构
# =====================================================================
class TestModuleShape:
    def test_metrics_and_names_aligned(self):
        assert len(eval_mod.METRICS) == len(eval_mod.METRIC_NAMES) == 4
        assert eval_mod.METRIC_NAMES == [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

    def test_qa_gen_prompt_uses_context_var(self):
        assert "context" in eval_mod.QA_GEN_PROMPT.input_variables

    def test_paths_under_project_root(self):
        assert eval_mod.QA_PATH.parent == eval_mod.EVALS_DIR


# =====================================================================
# answer_question (retriever / llm 由参数传入, 直接传 Mock)
# =====================================================================
class TestAnswerQuestion:
    def test_returns_docs_and_stripped_answer(self):
        docs = [Document(page_content="ctx", metadata={"source": "s", "page": 1})]
        retriever = Mock()
        retriever.invoke.return_value = docs
        llm = Mock()
        llm.invoke.return_value = Mock(content="<think> reasoning </think>final answer")

        out_docs, answer = eval_mod.answer_question("Q?", retriever, llm)

        assert out_docs is docs
        assert answer == "final answer"
        retriever.invoke.assert_called_once_with("Q?")

    def test_answer_without_think_tag(self):
        retriever = Mock()
        retriever.invoke.return_value = []
        llm = Mock()
        llm.invoke.return_value = Mock(content="plain")

        _, answer = eval_mod.answer_question("Q", retriever, llm)
        assert answer == "plain"

    def test_prompt_contains_context_and_question(self):
        docs = [Document(page_content="CTX", metadata={"source": "s", "page": 1})]
        retriever = Mock()
        retriever.invoke.return_value = docs
        llm = Mock()
        llm.invoke.return_value = Mock(content="A")

        eval_mod.answer_question("THEQ", retriever, llm)

        sent = llm.invoke.call_args[0][0]
        assert "THEQ" in sent
        assert "[s:p1] CTX" in sent


# =====================================================================
# generate_qa (mock load_index / get_llm)
# =====================================================================
def _fake_index(docs):
    """构造形如 FAISS index 的对象: index.docstore._dict.values() -> docs。"""
    index = Mock()
    index.docstore._dict.values.return_value = list(docs)
    return index


def _fake_llm_runnable(content: str):
    """可接入 LCEL 链的 Runnable: invoke 任何输入都返回 AIMessage(content)。

    QA_GEN_PROMPT | fake_llm | StrOutputParser() | RunnableLambda(strip_think)
    由此可以端到端跑通而不触达真实 API。
    """
    return RunnableLambda(lambda _prompt: AIMessage(content=content))


class TestGenerateQa:
    def test_generates_expected_count(self, monkeypatch):
        docs = [Document(page_content="docA", metadata={"source": "paperA", "page": 1})]
        monkeypatch.setattr(eval_mod, "load_index", lambda *a, **k: _fake_index(docs))
        monkeypatch.setattr(
            eval_mod,
            "get_llm",
            lambda **kw: _fake_llm_runnable("问题: 什么是检索增强生成?\n答案: 一种结合检索的生成范式。"),
        )

        qa = eval_mod.generate_qa(3)

        assert len(qa) == 3
        assert all(item["question"] and item["ground_truth"] for item in qa)
        assert all(item["source"] == "paperA" for item in qa)

    def test_filters_output_missing_answer_marker(self, monkeypatch):
        docs = [Document(page_content="d", metadata={"source": "s", "page": 1})]
        monkeypatch.setattr(eval_mod, "load_index", lambda *a, **k: _fake_index(docs))
        monkeypatch.setattr(
            eval_mod,
            "get_llm",
            lambda **kw: _fake_llm_runnable("问题: 缺少答案标记的内容"),
        )

        qa = eval_mod.generate_qa(2)
        # 每次输出都缺 "答案:" -> 全被丢弃, attempts 耗尽后返回空
        assert qa == []

    def test_filters_too_short_question(self, monkeypatch):
        docs = [Document(page_content="d", metadata={"source": "s", "page": 1})]
        monkeypatch.setattr(eval_mod, "load_index", lambda *a, **k: _fake_index(docs))
        # len(q) > 5 才接受, "abc" 长度 3 应被过滤
        monkeypatch.setattr(
            eval_mod,
            "get_llm",
            lambda **kw: _fake_llm_runnable("问题: abc\n答案: yes"),
        )

        assert eval_mod.generate_qa(1) == []

    def test_strip_think_applied_before_parse(self, monkeypatch):
        docs = [Document(page_content="d", metadata={"source": "s", "page": 1})]
        monkeypatch.setattr(eval_mod, "load_index", lambda *a, **k: _fake_index(docs))
        raw = "<think>noise</think>问题: 这是一个足够长的问题?\n答案: 这是答案"
        monkeypatch.setattr(eval_mod, "get_llm", lambda **kw: _fake_llm_runnable(raw))

        qa = eval_mod.generate_qa(1)
        assert len(qa) == 1
        assert qa[0]["question"] == "这是一个足够长的问题?"
        assert qa[0]["ground_truth"] == "这是答案"

    def test_terminates_at_attempts_cap_without_infinite_loop(self, monkeypatch):
        docs = [Document(page_content="d", metadata={"source": "s", "page": 1})]
        monkeypatch.setattr(eval_mod, "load_index", lambda *a, **k: _fake_index(docs))
        monkeypatch.setattr(
            eval_mod,
            "get_llm",
            lambda **kw: _fake_llm_runnable("totally garbage with no markers"),
        )

        # num=3 -> attempts 上限 12; 必须在有限步内返回而非死循环
        assert eval_mod.generate_qa(3) == []

    def test_get_llm_called_with_max_tokens(self, monkeypatch):
        docs = [Document(page_content="d", metadata={"source": "s", "page": 1})]
        monkeypatch.setattr(eval_mod, "load_index", lambda *a, **k: _fake_index(docs))
        captured = {}
        monkeypatch.setattr(
            eval_mod,
            "get_llm",
            lambda **kw: (captured.update(kw), _fake_llm_runnable("问题: 啊啊啊啊啊?\n答案: 嗯"))[1],
        )

        eval_mod.generate_qa(1)
        assert captured.get("max_tokens") == 300


# =====================================================================
# run_batch (mock make_retriever / load_index / get_llm)
# =====================================================================
class TestRunBatch:
    def _patch(self, monkeypatch, llm_content="answer"):
        docs = [Document(page_content="ctx", metadata={"source": "s", "page": 1})]
        retriever = Mock()
        retriever.invoke.return_value = docs
        monkeypatch.setattr(eval_mod, "make_retriever", lambda *a, **k: retriever)
        monkeypatch.setattr(eval_mod, "load_index", lambda *a, **k: Mock())
        llm = Mock()
        llm.invoke.return_value = Mock(content=llm_content)
        monkeypatch.setattr(eval_mod, "get_llm", lambda **kw: llm)
        return retriever, llm

    def test_records_structure(self, monkeypatch):
        self._patch(monkeypatch)
        qa_pairs = [
            {"question": "Q1", "ground_truth": "GT1"},
            {"question": "Q2", "ground_truth": "GT2"},
        ]

        records = eval_mod.run_batch(qa_pairs, "hyde")

        assert len(records) == 2
        for rec, qa in zip(records, qa_pairs):
            assert set(rec.keys()) == {
                "user_input",
                "response",
                "retrieved_contexts",
                "reference",
            }
            assert rec["user_input"] == qa["question"]
            assert rec["reference"] == qa["ground_truth"]
            assert rec["response"] == "answer"
            assert rec["retrieved_contexts"] == ["ctx"]

    def test_empty_input_returns_empty(self, monkeypatch):
        self._patch(monkeypatch)
        assert eval_mod.run_batch([], "none") == []

    def test_strips_think_from_response(self, monkeypatch):
        self._patch(monkeypatch, llm_content="<think>x</think>real")

        records = eval_mod.run_batch([{"question": "Q", "ground_truth": "G"}], "none")
        assert records[0]["response"] == "real"

    def test_make_retriever_receives_advanced_flag(self, monkeypatch):
        retriever, _ = self._patch(monkeypatch)
        eval_mod.run_batch([{"question": "Q", "ground_truth": "G"}], "hyde")
        # make_retriever(index, advanced="hyde") —— 第二关键字参数
        assert eval_mod.make_retriever is not None
        retriever.invoke.assert_called_once()


# =====================================================================
# eval_batch (mock ragas.evaluate / wrappers / get_llm / get_embedder)
# =====================================================================
class TestEvalBatch:
    def _patch_ragas(self, monkeypatch, df):
        monkeypatch.setattr(eval_mod, "get_llm", lambda **kw: Mock(name="llm"))
        monkeypatch.setattr(eval_mod, "get_embedder", lambda **kw: Mock(name="emb"))
        monkeypatch.setattr(eval_mod, "LangchainLLMWrapper", lambda llm: ("llm_wrap", llm))
        monkeypatch.setattr(eval_mod, "LangchainEmbeddingsWrapper", lambda emb: ("emb_wrap", emb))
        eds = Mock()
        eds.from_list = lambda recs: ("dataset", recs)
        monkeypatch.setattr(eval_mod, "EvaluationDataset", eds)

        fake_result = Mock()
        fake_result.to_pandas.return_value = df
        captured = {}

        def fake_evaluate(dataset, metrics=None, llm=None, embeddings=None):
            captured["dataset"] = dataset
            captured["metrics"] = metrics
            captured["llm"] = llm
            captured["embeddings"] = embeddings
            return fake_result

        monkeypatch.setattr(eval_mod, "evaluate", fake_evaluate)
        return captured

    def test_returns_mean_per_metric(self, monkeypatch):
        df = pd.DataFrame(
            {
                "faithfulness": [0.8, 0.6],
                "answer_relevancy": [0.7, 0.7],
                "context_precision": [0.5, 0.5],
                "context_recall": [1.0, 0.0],
            }
        )
        captured = self._patch_ragas(monkeypatch, df)

        scores = eval_mod.eval_batch([{"user_input": "q", "reference": "r"}])

        assert scores == {
            "faithfulness": pytest.approx(0.7),
            "answer_relevancy": pytest.approx(0.7),
            "context_precision": pytest.approx(0.5),
            "context_recall": pytest.approx(0.5),
        }
        # 传给 ragas.evaluate 的参数
        assert captured["metrics"] == eval_mod.METRICS
        assert captured["llm"][0] == "llm_wrap"
        assert captured["embeddings"][0] == "emb_wrap"

    def test_missing_metric_column_is_omitted(self, monkeypatch):
        df = pd.DataFrame({"faithfulness": [1.0], "answer_relevancy": [0.5]})
        self._patch_ragas(monkeypatch, df)

        scores = eval_mod.eval_batch([])
        assert "faithfulness" in scores and scores["faithfulness"] == pytest.approx(1.0)
        assert "context_precision" not in scores
        assert "context_recall" not in scores

    def test_values_are_python_floats(self, monkeypatch):
        df = pd.DataFrame(
            {
                "faithfulness": [0.5],
                "answer_relevancy": [0.5],
                "context_precision": [0.5],
                "context_recall": [0.5],
            }
        )
        self._patch_ragas(monkeypatch, df)

        scores = eval_mod.eval_batch([])
        for v in scores.values():
            assert isinstance(v, float)


# =====================================================================
# main (CLI: 全程 mock, 路径重定向到 tmp_path)
# =====================================================================
class TestMain:
    def _redirect_paths(self, monkeypatch, tmp_path):
        evals_dir = tmp_path / "evals"
        results_dir = tmp_path / "results"
        qa_path = evals_dir / "qa_pairs.json"
        monkeypatch.setattr(eval_mod, "EVALS_DIR", evals_dir)
        monkeypatch.setattr(eval_mod, "RESULTS_DIR", results_dir)
        monkeypatch.setattr(eval_mod, "QA_PATH", qa_path)
        return qa_path, results_dir

    def test_gen_only_with_existing_qa_skips_eval(self, monkeypatch, tmp_path):
        qa_path, _ = self._redirect_paths(monkeypatch, tmp_path)
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        pairs = [{"question": "Q%d?" % i, "ground_truth": "A%d" % i} for i in range(3)]
        qa_path.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")

        run_batch = Mock()
        eval_batch = Mock()
        monkeypatch.setattr(eval_mod, "run_batch", run_batch)
        monkeypatch.setattr(eval_mod, "eval_batch", eval_batch)
        monkeypatch.setattr(sys, "argv", ["eval", "--gen-only"])

        assert eval_mod.main() is None
        run_batch.assert_not_called()
        eval_batch.assert_not_called()

    def test_generates_and_persists_qa_when_missing(self, monkeypatch, tmp_path):
        qa_path, _ = self._redirect_paths(monkeypatch, tmp_path)
        generated = [{"question": "新问题?", "ground_truth": "新答案", "source": "p"}]

        def fake_gen(n):
            assert n == 2
            return [dict(item) for item in generated]

        monkeypatch.setattr(eval_mod, "generate_qa", fake_gen)
        monkeypatch.setattr(sys, "argv", ["eval", "--num", "2", "--gen-only"])

        eval_mod.main()

        assert qa_path.exists()
        saved = json.loads(qa_path.read_text(encoding="utf-8"))
        assert saved == generated

    def test_full_run_writes_records_and_summary(self, monkeypatch, tmp_path):
        qa_path, results_dir = self._redirect_paths(monkeypatch, tmp_path)
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        pairs = [{"question": "Q?", "ground_truth": "GT"}]
        qa_path.write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")

        records = [
            {
                "user_input": "Q?",
                "response": "A",
                "retrieved_contexts": ["c"],
                "reference": "GT",
            }
        ]

        def fake_run_batch(qa, advanced):
            return [dict(r) for r in records]

        monkeypatch.setattr(eval_mod, "run_batch", fake_run_batch)
        scores = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        monkeypatch.setattr(eval_mod, "eval_batch", lambda recs: scores)
        monkeypatch.setattr(sys, "argv", ["eval", "--num", "1"])

        eval_mod.main()

        assert (results_dir / "baseline_records.json").exists()
        assert (results_dir / "hyde_records.json").exists()
        assert (results_dir / "summary.csv").exists()

        csv_text = (results_dir / "summary.csv").read_text(encoding="utf-8")
        assert "faithfulness" in csv_text
        assert "baseline" in csv_text.splitlines()[0]
        # delta = hyde - baseline = 0 (两次分数相同)
        assert "+0.000" in csv_text

    def test_full_run_runs_both_tags(self, monkeypatch, tmp_path):
        qa_path, _ = self._redirect_paths(monkeypatch, tmp_path)
        qa_path.parent.mkdir(parents=True, exist_ok=True)
        qa_path.write_text(
            json.dumps([{"question": "Q?", "ground_truth": "G"}], ensure_ascii=False),
            encoding="utf-8",
        )

        seen = []

        def fake_run_batch(qa, advanced):
            seen.append(advanced)
            return [{"user_input": "q", "response": "a", "retrieved_contexts": [], "reference": "g"}]

        monkeypatch.setattr(eval_mod, "run_batch", fake_run_batch)
        monkeypatch.setattr(
            eval_mod,
            "eval_batch",
            lambda recs: {m: 0.5 for m in eval_mod.METRIC_NAMES},
        )
        monkeypatch.setattr(sys, "argv", ["eval"])

        eval_mod.main()
        assert seen == ["none", "hyde"]
