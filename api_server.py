"""HTTP API for Srish, so the Srijan mobile app can talk to the bot.

The Streamlit app (agentic_bot.py) stays exactly as it is — both front-ends
call srish_core, so they always answer the same way.

Run locally:
    pip install -r requirements.txt
    uvicorn api_server:app --host 0.0.0.0 --port 8000

Ollama must be reachable from wherever this runs (it is the LLM backend).

Endpoints:
    GET  /health       — readiness, including whether Ollama is actually up
    POST /chat         — one conversation turn, whole reply at once
    POST /chat/stream  — same turn streamed as NDJSON (use this on mobile)
"""

import json
import logging
import os

import ollama
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import srish_core

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("srish.api")

MAX_MESSAGE_CHARS = 2000
MAX_HISTORY_TURNS = 20

# Comma-separated list, or "*" (the default) for local development.
_origins = os.environ.get("SRISH_ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = ["*"] if _origins.strip() == "*" else [
    o.strip() for o in _origins.split(",") if o.strip()
]

app = FastAPI(title="Srish API", version="1.0.0")

# Expo serves the app from a dev-server origin, and React Native web builds send
# a real Origin header, so CORS has to be permitted explicitly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    # Oldest first, excluding the current message.
    history: list[Turn] = []
    farm_id: str | None = None
    depth: bool = False


class ChatResponse(BaseModel):
    reply: str
    intent: str | None = None
    language: str | None = None
    # True when the router was unreachable: the off-topic filter and the KB
    # lookup were both skipped for this turn, so the answer is less trustworthy.
    degraded: bool = False
    # True when Ollama could not be reached at all. Clients should show this as
    # the bot being offline rather than rendering `reply` as a real answer.
    backend_down: bool = False
    used_kb: bool = False
    kb_disease: str | None = None
    farm_rows: list[dict] = []
    error: str | None = None


@app.get("/health")
def health() -> dict:
    """Readiness probe. Reports whether the models Srish needs are present."""
    models: list[str] = []
    ollama_ok = False
    try:
        listing = ollama.list()
        models = [
            m.get("model") or m.get("name")
            for m in listing.get("models", [])
        ]
        ollama_ok = True
    except Exception as e:
        log.warning("Ollama unreachable: %s", e)

    router_ok = srish_core.ROUTER_MODEL in models
    chat_ok = any(
        (m or "").split(":")[0] == srish_core.CHAT_MODEL.split(":")[0] for m in models
    )
    # "ready" is the only field a client needs to decide online vs offline.
    # The API process can be perfectly healthy while Ollama is stopped, in which
    # case Srish cannot answer anything -- so a 200 here does not mean usable.
    ready = ollama_ok and router_ok and chat_ok

    reasons = []
    if not ollama_ok:
        reasons.append("Ollama is not running")
    if ollama_ok and not router_ok:
        reasons.append(f"router model {srish_core.ROUTER_MODEL} is not installed")
    if ollama_ok and not chat_ok:
        reasons.append(f"chat model {srish_core.CHAT_MODEL} is not installed")

    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "not_ready_because": reasons,
        "ollama": ollama_ok,
        "orchestrator": srish_core.ORCHESTRATOR,
        "router_model": srish_core.ROUTER_MODEL,
        "router_model_present": router_ok,
        "chat_model": srish_core.CHAT_MODEL,
        "chat_model_present": chat_ok,
        "kb_entries": len(srish_core.load_knowledge_base()),
    }


EMPTY_REPLY = "Kuch poochhiye toh! Main aapke tomato plants ke baare mein bata sakti hoon. 🍅"


def _clean(req: ChatRequest) -> tuple[str, list[dict]]:
    return (
        (req.message or "").strip()[:MAX_MESSAGE_CHARS],
        [t.model_dump() for t in req.history[-MAX_HISTORY_TURNS:]],
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Run one turn and return the whole reply. Never raises — errors come back
    in `error`, so the app always has something to show."""
    message, history = _clean(req)
    if not message:
        return ChatResponse(reply=EMPTY_REPLY, intent="greeting")

    return ChatResponse(**srish_core.answer(
        message=message,
        history=history,
        farm_id=req.farm_id or srish_core.DEFAULT_FARM_ID,
        depth_mode=req.depth,
    ))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Same turn as /chat, streamed as newline-delimited JSON.

    Prefer this on mobile: routing still costs a few seconds, but the user sees
    words appear instead of staring at a spinner for the whole answer.

    Each line is one JSON object:
        {"type": "meta",  "intent": ..., "language": ..., "degraded": ...}
        {"type": "token", "text": "..."}        (many)
        {"type": "done",  "reply": "...", "farm_rows": [...], "error": null}
    """
    message, history = _clean(req)

    def gen():
        if not message:
            yield json.dumps({"type": "meta", "intent": "greeting"}) + "\n"
            yield json.dumps({"type": "token", "text": EMPTY_REPLY}) + "\n"
            yield json.dumps({"type": "done", "reply": EMPTY_REPLY}) + "\n"
            return

        try:
            for kind, payload in srish_core.answer_stream(
                message=message,
                history=history,
                farm_id=req.farm_id or srish_core.DEFAULT_FARM_ID,
                depth_mode=req.depth,
            ):
                if kind == "token":
                    yield json.dumps({"type": "token", "text": payload}) + "\n"
                else:
                    yield json.dumps({"type": kind, **payload}) + "\n"
        except Exception as e:  # a generator blowing up would hang the client
            log.exception("stream failed")
            yield json.dumps({"type": "done", "reply": "", "error": str(e)}) + "\n"

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        # Stops nginx and friends from buffering the stream into one lump.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
