"""Thin FastAPI wrapper over the same agent: uvicorn app.main:app"""
from fastapi import FastAPI

from .run import build_agent

app = FastAPI(title="Support ticket triage agent")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/status")
def status() -> dict:
    _, _, hd, _ = build_agent()
    return hd.status()


@app.post("/run")
def run(verify_replies: bool = True) -> dict:
    agent, llm, _, _ = build_agent(verify_replies)
    summary = agent.run()
    return {"summary": summary.__dict__, "llm_usage": llm.usage.__dict__}
