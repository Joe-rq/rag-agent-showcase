"""FastAPI app (app/main.py) 的端点测试。

用法: uv run pytest tests/test_app.py -q

设计要点:
- 用 fastapi.testclient.TestClient 测 /health, /chat, /playground。
- monkeypatch 掉 app.main.load_index 与 build_rag_chain, 让 lifespan 不触发真实
  索引/LLM 调用; 各测试再按需把 app.main._chain 设成 Mock 或 None 模拟"链就绪/未就绪"。
- 注意: 直接 monkeypatch `_chain` 不够 —— lifespan 在 `with TestClient(app)` 进入时
  会重新跑 load_index+build_rag_chain 覆盖 `_chain`, 所以必须先挡住这两个函数。

关于 import: app/ 是项目根下的普通目录, 不是 installed package(rag_agent 才是,
经 hatchling editable 安装进 .venv)。pytest prepend 模式只把 tests/ 插入 sys.path,
项目根不在内, 故需在顶部手工把项目根加进 sys.path, 让 `import app.main` 可用。
"""
import os
import sys

# 让 `import app.main` 在 pytest 下可用(app 不是 installed package)。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app.main as app_main
from app.main import ChatRequest, app


@pytest.fixture
def client():
    """启动 app 触发 lifespan, 但用假函数替换索引/链构建避免 LLM/索引调用。

    lifespan 跑完后 app.main._chain 是一个 Mock(build_rag_chain 的返回值),
    即"链就绪"状态; 需测"未就绪"分支的用例再自行把 _chain 置 None。
    """
    orig_load = app_main.load_index
    orig_build = app_main.build_rag_chain
    app_main.load_index = lambda name: object()
    app_main.build_rag_chain = lambda index: Mock(name="rag_chain")
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app_main.load_index = orig_load
        app_main.build_rag_chain = orig_build


# ---------- /health ----------

def test_health_chain_ready(client):
    """正常路径: 链已构建, /health 返回 status=ok + chain_ready=True。"""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chain_ready"] is True


def test_health_chain_not_ready(client, monkeypatch):
    """错误处理: 链未构建(LLM key 缺失等), /health 仍 200 但 chain_ready=False。"""
    monkeypatch.setattr(app_main, "_chain", None)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["chain_ready"] is False


# ---------- /chat ----------

def test_chat_normal(client):
    """正常路径: 链就绪时调用 chain.invoke(question) 并回包 {question, answer}。"""
    app_main._chain.invoke.return_value = "ReAct 交替推理与行动。"
    r = client.post("/chat", json={"question": "ReAct 是什么?"})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "ReAct 是什么?"
    assert body["answer"] == "ReAct 交替推理与行动。"
    app_main._chain.invoke.assert_called_once_with("ReAct 是什么?")


def test_chat_empty_question(client):
    """边界: 空 question 不被拦截, 原样透传给 chain.invoke。"""
    app_main._chain.invoke.return_value = "空问题的回答"
    r = client.post("/chat", json={"question": ""})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == ""
    assert body["answer"] == "空问题的回答"
    app_main._chain.invoke.assert_called_once_with("")


def test_chat_chain_not_ready(client, monkeypatch):
    """错误处理: 链未就绪时 /chat 返回 error 提示(注意是 200, 非 4xx/5xx)。"""
    monkeypatch.setattr(app_main, "_chain", None)
    r = client.post("/chat", json={"question": "anything"})
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    # 文案指向 LLM_API_KEY, 与 main.py 中提示一致
    assert "LLM_API_KEY" in body["error"]
    assert "answer" not in body


def test_chat_missing_question_422(client):
    """错误处理: 缺 question 字段 → pydantic 校验失败 → 422。"""
    r = client.post("/chat", json={})
    assert r.status_code == 422


def test_chat_wrong_question_type_422(client):
    """错误处理: question 非 string → 422。"""
    r = client.post("/chat", json={"question": 123})
    assert r.status_code == 422


def test_chat_invoke_raises_500(monkeypatch):
    """错误处理: chain.invoke 抛异常时 app 未捕获, 冒泡为 HTTP 500。

    需 raise_server_exceptions=False 让 TestClient 把服务端异常转成 500 响应,
    而非在客户端侧 re-raise。
    """
    monkeypatch.setattr(app_main, "load_index", lambda name: object())
    chain = Mock()
    chain.invoke.side_effect = RuntimeError("boom")
    monkeypatch.setattr(app_main, "build_rag_chain", lambda index: chain)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/chat", json={"question": "x"})
    assert r.status_code == 500


# ---------- /playground ----------

def test_playground_returns_html(client):
    """正常路径: /playground 返回 200 + HTML, 含标题与 /chat 调用。"""
    r = client.get("/playground")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    text = r.text
    assert "RAG" in text
    assert "/chat" in text
    assert "<html" in text.lower()


# ---------- ChatRequest 模型 ----------

def test_chat_request_model_accepts_str():
    """边界/单元: ChatRequest 接受字符串 question。"""
    req = ChatRequest(question="hi")
    assert req.question == "hi"


def test_chat_request_model_rejects_missing():
    """错误处理/单元: 缺 question 抛 ValidationError。"""
    with pytest.raises(ValidationError):
        ChatRequest()
