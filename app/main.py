"""FastAPI 服务: 暴露 RAG 问答接口。

弃用已废弃的 LangServe, 用裸 FastAPI(理由见 docs/adr/0003-弃用LangServe.md)。
启动: uv run uvicorn app.main:app --port 8000
交互: 浏览器打开 http://localhost:8000/playground
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag_agent.chain import build_rag_chain
from rag_agent.indexing import load_index

_chain = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载索引 + 构建 RAG 链, 避免每请求重建。

    LLM key 未配时链构建失败, 但 app 仍启动(/health 返回 chain_ready=false),
    便于在未配 key 时也能访问 /playground 排查。
    """
    global _chain
    try:
        index = load_index("default")
        _chain = build_rag_chain(index)
        print("[app] RAG 链就绪")
    except Exception as e:
        print(f"[app] RAG 链构建失败(检查 LLM key): {e}")
        _chain = None
    yield


app = FastAPI(title="RAG Agent Showcase", version="0.1.0", lifespan=lifespan)


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "chain_ready": _chain is not None}


@app.post("/chat")
def chat(req: ChatRequest):
    if _chain is None:
        return {"error": "RAG 链未就绪, 请检查 .env 的 LLM_API_KEY"}
    answer = _chain.invoke(req.question)
    return {"question": req.question, "answer": answer}


PLAYGROUND_HTML = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>RAG Playground</title>
<style>
body{font-family:system-serif;max-width:760px;margin:2rem auto;padding:0 1rem}
textarea{width:100%;font-size:1rem} button{margin:.5rem 0;padding:.5rem 1.5rem}
#a{white-space:pre-wrap;background:#f5f5f5;padding:1rem;margin-top:1rem;border-radius:6px;min-height:2rem}
</style></head>
<body><h1>RAG Agent Playground</h1>
<p>基于 5 篇 AI Agent 论文(ReAct/Reflexion/Toolformer/ToT/Voyager)的问答。</p>
<textarea id="q" rows="3" placeholder="例如: ReAct 是如何结合推理和行动的?"></textarea><br>
<button onclick="ask()">提问</button>
<div id="a">答案会显示在这里...</div>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim();
  if(!q) return;
  document.getElementById('a').textContent='思考中...';
  try{
    const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
    const d=await r.json();
    document.getElementById('a').textContent=d.answer||('错误: '+JSON.stringify(d));
  }catch(e){document.getElementById('a').textContent='请求失败: '+e}
}
</script></body></html>"""


@app.get("/playground", response_class=HTMLResponse)
def playground():
    return PLAYGROUND_HTML
