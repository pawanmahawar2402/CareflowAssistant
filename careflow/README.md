# CareFlow — AI Care Coordination Assistant

**Applied AI Professional Certification Program · IIT Hyderabad**  
**Problem Statement 1 · Healthcare Operations**

A multi-agent AI assistant that triages patient intake messages, answers policy/insurance questions from clinic documents, tracks referrals, and escalates clinical matters to human staff.

## Milestones

| # | What is built |
|---|---|
| **M1** | `PatientRequest` Pydantic model; LiteLLM parser extracting symptoms, urgency, insurance ID |
| **M2** | Single scheduling/insurance agent; mock EHR + eligibility + co-pay + flag-for-human tools |
| **M3** | Session + long-term memory (Redis); Qdrant index over clinic policy/insurance corpus |
| **M4** | Production RAG (hybrid BM25 + dense, reranking); groundedness evaluation |
| **M5** | LangGraph workflow: intake → eligibility → referral routing; human-in-the-loop checkpoint |
| **M6** | Full 4-agent team; EHR/scheduling MCP server |
| **M7** | LangFuse tracing; retry + fallback + circuit-breaker on API calls |
| **M8** | 20-case golden eval set; guardrails blocking clinical advice; FastAPI service |

## Quick start

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env          # add GEMINI_API_KEY
    python -m careflow.m4_rag.ingest
    docker run -d -p 6379:6379 redis:7-alpine     # memory store (M3); the API still runs without it
    python generate_data.py                       # synthetic tables, corpus and golden sets
    uvicorn careflow.m8_service.api:app --reload  # POST /chat, POST /review, DELETE /users/{id}/memory, GET /health
    python -m careflow.m8_service.eval_runner     # end-to-end golden set (needs GEMINI_API_KEY)
    python -m pytest tests/                       # 23 tests: LLM stubbed; Redis tests skip if Redis is down
