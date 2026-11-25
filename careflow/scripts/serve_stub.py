"""Launcher for the restart test (scripts/restart_test.sh).

Runs the real API with the real graph, Redis, SQLite checkpoints and MCP server. Only the LLM calls are
stubbed (intake parsing, fact extraction, safety critic), so the test uses no quota and is repeatable.
"""
import uvicorn

from careflow import llm
from careflow.m1_intake.models import PatientRequest
from careflow.m6_agents import team


async def parse(raw, channel="unknown"):
    if "MRD-100037" in raw:
        return PatientRequest(chief_complaint="c", raw_message=raw, channel=channel,
                              intent="eligibility_check", insurance_id="MRD-100037")
    if "REF-0003" in raw:
        return PatientRequest(chief_complaint="c", raw_message=raw, channel=channel, intent="referral_status")
    return PatientRequest(chief_complaint="c", raw_message=raw, channel=channel, intent="other")


async def astructured(messages, schema, **kw):
    if schema.__name__ == "_Facts":
        return schema.model_validate({}), None
    return schema(approved=True, issues=[]), None


team.parse_message = parse
llm.astructured = astructured

if __name__ == "__main__":
    uvicorn.run("careflow.m8_service.api:app", host="127.0.0.1", port=8011, log_level="warning")
