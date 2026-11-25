"""M8 — FastAPI service: /chat, /review (HIL), /users/{id}/memory (privacy), /health."""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from careflow.config import PROJECT_ROOT
from careflow.m3_memory.memory import memory, store
from careflow.m5_workflow import graph as wf
from careflow.m6_agents.mcp_client import close_mcp
from careflow.m7_observability.resilience import (
    CircuitOpenError, enable_litellm_callbacks, llm_breaker, trace_span,
)

log = logging.getLogger(__name__)

_USER_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"


@asynccontextmanager
async def lifespan(_: FastAPI):
    enable_litellm_callbacks()
    yield
    await wf.close_graph()
    await close_mcp()
    await store.close()


app = FastAPI(title="CareFlow", version="1.0", lifespan=lifespan)
UI_DIR = Path(__file__).parent / "ui"
app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None
    user_id: str | None = Field(None, pattern=_USER_PATTERN, description="Patient namespace for memory")
    channel: str = "web_form"


class ChatResponse(BaseModel):
    session_id: str
    response: str
    route: str | None = None
    pending_human: bool = False
    escalation_ticket: str | None = None


class ReviewRequest(BaseModel):
    session_id: str
    approved: bool
    note: str = ""


def _to_response(sid: str, st: dict) -> ChatResponse:
    tk = st.get("escalation_ticket") or {}
    return ChatResponse(
        session_id=sid, response=st.get("response", ""), route=st.get("route"),
        pending_human=st.get("pending_human", False), escalation_ticket=tk.get("ticket_id"),
    )


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/ui/")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "llm_circuit": llm_breaker.state}


@app.get("/eval/results")
async def eval_results() -> dict:
    """Committed evaluation outputs for the dashboard: retrieval comparison and end-to-end runs."""
    base = PROJECT_ROOT / "eval_results"
    out: dict = {"retrieval": None, "retrieval_variants": {}, "e2e_runs": []}
    retrieval = base / "retrieval_results.json"
    if retrieval.exists():
        out["retrieval"] = json.loads(retrieval.read_text(encoding="utf-8"))
    for name in ("retrieval_results_baseline_threshold-6.json", "retrieval_results_after_threshold_0.json"):
        p = base / name
        if p.exists():
            out["retrieval_variants"][name] = json.loads(p.read_text(encoding="utf-8"))["summary"]
    if out["retrieval"] is None and "retrieval_results_after_threshold_0.json" in out["retrieval_variants"]:
        out["retrieval"] = json.loads((base / "retrieval_results_after_threshold_0.json").read_text(encoding="utf-8"))
    for p in sorted(base.glob("e2e_results_*.json")):
        out["e2e_runs"].append({"file": p.name, **json.loads(p.read_text(encoding="utf-8"))})
    return out


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    sid = req.session_id or uuid.uuid4().hex
    try:
        with trace_span("chat", session_id=sid):
            st = await wf.run(req.message, sid, req.channel, user_id=req.user_id)
    except CircuitOpenError:
        raise HTTPException(503, "LLM temporarily unavailable; please retry shortly")
    out = _to_response(sid, st)
    if st.get("route") == "refuse" and st.get("guardrail_flags"):
        # an explicit status and reason for guardrail rejections (M8.C4); the body still carries the reply
        return JSONResponse(status_code=422, content={
            **out.model_dump(), "reason": st.get("refusal_reason"), "guardrail_flags": st["guardrail_flags"]})
    return out


@app.post("/review", response_model=ChatResponse)
async def review(req: ReviewRequest) -> ChatResponse:
    """Staff approve/decline a paused escalation."""
    try:
        st = await wf.resume(req.session_id, req.approved, req.note)
    except Exception as exc:
        raise HTTPException(404, f"No pending case for session: {exc}")
    return _to_response(req.session_id, st)


@app.delete("/users/{user_id}/memory")
async def delete_memory(user_id: str) -> dict:
    """Privacy path (M3.A4): removes one patient's episodic and semantic memory."""
    import re

    if not re.match(_USER_PATTERN, user_id):
        raise HTTPException(400, "invalid user_id")
    deleted = await memory.delete_user_data(user_id)
    return {"user_id": user_id, "keys_deleted": deleted}
