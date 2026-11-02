"""Srish's brain, with no UI attached.

agentic_bot.py (Streamlit) and api_server.py (FastAPI, for the mobile app) both
call into here, so the two front-ends cannot drift apart in their answers.

Nothing in this module imports streamlit. Failures are logged and reported in
the returned dict instead of being drawn on screen — the caller decides how to
show them.
"""

import json
import logging
import os
import re

import httpx
import ollama

import farm_data

log = logging.getLogger("srish")

HERE = os.path.dirname(os.path.abspath(__file__))

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Routing and web extraction run on a local Ollama model by default, so Srish
# needs no API key and no internet. Set SRISH_ORCHESTRATOR=gemini to route via
# Gemini instead — that needs GEMINI_API_KEY and a project with API access.
ORCHESTRATOR = os.environ.get("SRISH_ORCHESTRATOR", "ollama").lower()
# One model does both jobs by default, and that is a deliberate choice about
# VRAM rather than about quality. The router and the chat model both have to be
# resident; if they do not both fit, Ollama evicts one on every single turn and
# each reload costs seconds. Measured on an RTX 3050 6GB:
#   qwen2.5:7b routing + SrishBot answering  13s to first word (constant swapping)
#   phi3:mini  routing + SrishBot answering  13s to first word (still swapping)
#   qwen2.5:7b doing both                     6s to first word (nothing evicted)
# qwen2.5:7b also routed 13/13 on the intent suite where phi3:mini got 12/13,
# and gives better agronomy than the 3B SrishBot.
# With 12GB+ of VRAM, SRISH_CHAT_MODEL=SrishBot brings back the fine-tuned
# persona at no latency cost, because both models then stay loaded.
ROUTER_MODEL = os.environ.get("SRISH_ROUTER_MODEL", "qwen2.5:7b")
CHAT_MODEL = os.environ.get("SRISH_CHAT_MODEL", "qwen2.5:7b")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
# How long Ollama keeps a model in memory after a request. This is not just a
# warm-start optimisation: Ollama's default is 5 minutes, and if that expires
# while a streamed response is still open, the unload races the stream and the
# request hangs instead of finishing. Keeping the model pinned avoids that and
# also saves the multi-second reload between questions.
KEEP_ALIVE = os.environ.get("SRISH_KEEP_ALIVE", "30m")
DEFAULT_FARM_ID = os.environ.get("SRISH_FARM_ID", "FARM_001")
KB_PATH = os.environ.get("SRISH_KB_PATH", os.path.join(HERE, "tomato_data_clean.json"))

_gemini = None
if ORCHESTRATOR == "gemini":
    try:
        import google.generativeai as genai

        _key = os.environ.get("GEMINI_API_KEY")
        if not _key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=_key)
        _gemini = genai.GenerativeModel(GEMINI_MODEL)
    except Exception as e:
        log.warning("Gemini unavailable (%s) — using local %s instead.", e, ROUTER_MODEL)
        ORCHESTRATOR = "ollama"

VALID_INTENTS = {
    "greeting", "tomato_disease", "tomato_general",
    "url_query", "off_topic", "farm_status", "farm_report",
}

NO_DATA_REPLY = (
    "Abhi koi farm sensor data available nahi hai. 🌱\n\n"
    "Sensor se koi reading nahi aayi — please thodi der baad try karein, "
    "ya check karein ki sensor connected hai."
)

# Used when the router flags off_topic but forgets to write direct_reply. Without
# this the turn falls through to SrishBot and the question gets answered anyway,
# which defeats the whole bouncer rule.
OFF_TOPIC_REPLY = {
    "hinglish": (
        "Main sirf tomato aur farming ke baare mein jaanti hoon! 🍅 "
        "Kya aapke plants mein koi dikkat hai?"
    ),
    "english": (
        "I only know about tomatoes and farming! 🍅 "
        "Is there something going on with your plants I can help with?"
    ),
}

_HINGLISH_HINTS = re.compile(
    r"\b(hai|hain|kya|kaise|kaisa|mera|meri|mere|nahi|nahin|dikhao|chahiye|karu|"
    r"karun|batao|abhi|khet|patte|paudha|sookh|peele|hafte|pichle|pichhle|thoda|"
    r"bahut|accha|theek|haan|namaste)\b",
    re.I,
)


def _looks_hinglish(text: str) -> bool:
    """Cheap language guess, used when the router omits the language field."""
    return bool(_HINGLISH_HINTS.search(text)) or bool(re.search(r"[ऀ-ॿ]", text))


# ── LLM BACKENDS ──────────────────────────────────────────────────────────────

def _llm_json(prompt: str) -> dict:
    """Run a routing prompt and return parsed JSON. Raises if it cannot."""
    if ORCHESTRATOR == "gemini" and _gemini is not None:
        raw = _gemini.generate_content(prompt).text
    else:
        raw = ollama.chat(
            model=ROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0, "top_p": 0.5},
            keep_alive=KEEP_ALIVE,
        )["message"]["content"]
    return json.loads(re.sub(r"```json|```", "", raw).strip())


def _llm_text(prompt: str) -> str:
    """Run a free-text prompt and return the reply."""
    if ORCHESTRATOR == "gemini" and _gemini is not None:
        return _gemini.generate_content(prompt).text.strip()
    return ollama.chat(
        model=ROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2},
        keep_alive=KEEP_ALIVE,
    )["message"]["content"].strip()


# ── KNOWLEDGE BASE ────────────────────────────────────────────────────────────

_kb_cache = None


def load_knowledge_base() -> list:
    """Load and memoise the disease KB. Returns [] if the file is missing."""
    global _kb_cache
    if _kb_cache is None:
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                _kb_cache = json.load(f)
        except FileNotFoundError:
            log.warning("Knowledge base not found at %s", KB_PATH)
            _kb_cache = []
        except Exception as e:
            log.error("Could not load knowledge base: %s", e)
            _kb_cache = []
    return _kb_cache


def get_kb_entry_by_name(disease_name: str, db: list) -> dict | None:
    """Fetch full KB entry by the disease name the router identified."""
    if not disease_name or not db:
        return None
    name_lower = disease_name.lower().strip()
    for entry in db:
        c = entry.get("content", {})
        if c.get("disease_name", "").lower().strip() == name_lower:
            return c
    for entry in db:  # fuzzy fallback
        c = entry.get("content", {})
        db_name = c.get("disease_name", "").lower()
        if name_lower in db_name or db_name in name_lower:
            return c
    return None


# ── ROUTING ───────────────────────────────────────────────────────────────────

def orchestrate(user_message: str, chat_history: list, tomato_db: list) -> dict:
    """Read the conversation and return a structured JSON plan for this turn.

    On router failure the plan carries "_degraded": True. Callers should surface
    that — a silent fallback switches off both the off-topic filter and the KB
    lookup while still looking like a healthy bot.
    """
    recent_turns = chat_history[-6:] if len(chat_history) > 6 else chat_history
    conv_snapshot = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}" for m in recent_turns
    )

    kb_summary = [
        {
            "disease_name": e.get("content", {}).get("disease_name", ""),
            "keywords": str(e.get("content", {}).get("signs_and_symptoms", ""))[:200],
        }
        for e in tomato_db
    ]

    orchestration_prompt = f"""You are the intent router for "Srish", a tomato plant & disease expert chatbot.

Classify the user's latest message and return JSON only.

=== RECENT CONVERSATION ===
{conv_snapshot}

=== USER'S LATEST MESSAGE ===
{user_message}

=== AVAILABLE KNOWLEDGE BASE DISEASES ===
{json.dumps(kb_summary, ensure_ascii=False)}

=== INTENTS ===
- greeting       : hi, hello, namaste, who are you
- tomato_disease : ANY plant problem or symptom (spots, yellowing, wilting, drying, bugs, rot)
- tomato_general : tomato/farming question that is NOT a symptom — how to grow, appearance,
                   watering, sunlight, soil, harvest, taste, height, colour
- farm_status    : asks for farm/sensor readings with NO time range ("farm data dikhao",
                   "soil moisture", "khet ka data", "farm status", "NPK")
- farm_report    : asks for data over a TIME RANGE ("last 10 days", "is hafte ki report",
                   "weekly report"). Also fill "days" (default 7).
- url_query      : the message contains a URL — put it in "url"
- off_topic      : NOT about plants/food/farming.

=== BOUNCER RULE ===
These are ALWAYS off_topic: cricket, IPL, FIFA, sports players (Kohli, Dhoni, Messi),
movies, actors, politicians (Modi), singers, tech companies unrelated to farming, and any
short form or acronym unrelated to tomatoes. A person's NAME is NEVER a plant, disease,
bacterium or virus — never invent a disease named after a person. For off_topic, write a
short warm refusal in "direct_reply" in the user's own language. Otherwise "direct_reply": null.

=== KB LOOKUP ===
Set "use_kb": true ONLY if intent is tomato_disease AND a disease above closely matches the
symptoms; then set "kb_disease_name" to the EXACT name from that list. Otherwise false/null.

=== RULES ===
1. Fill "reasoning" first — think about what the user is actually asking.
2. tomato_general covers ALL appearance, growth and gardening questions. Never off_topic.
3. language: "hinglish" if the message uses Hindi or Roman-Hindi words, else "english".
4. topic_changed: true if this message starts a new subject.

Return exactly these keys:
{{"reasoning": str, "intent": str, "days": int|null, "use_kb": bool, "kb_disease_name": str|null,
 "url": str|null, "topic_changed": bool, "context_summary": str, "direct_reply": str|null,
 "language": str}}

Return ONLY valid JSON. No markdown fences. No extra text.
"""

    try:
        plan = _llm_json(orchestration_prompt)
    except Exception as e:
        log.error("Router unavailable: %s", e)
        return {
            "intent": "tomato_general",
            "use_kb": False,
            "kb_disease_name": None,
            "url": None,
            "topic_changed": False,
            "context_summary": user_message,
            "direct_reply": None,
            "language": "hinglish" if _looks_hinglish(user_message) else "english",
            "_degraded": True,
            "_error": str(e),
        }

    # Local models occasionally drop a key or null it out — normalise before use.
    if plan.get("intent") not in VALID_INTENTS:
        plan["intent"] = "tomato_general"
    if plan.get("language") not in ("english", "hinglish"):
        plan["language"] = "hinglish" if _looks_hinglish(user_message) else "english"
    if not plan.get("context_summary"):
        plan["context_summary"] = user_message
    if plan.get("intent") == "off_topic" and not plan.get("direct_reply"):
        plan["direct_reply"] = OFF_TOPIC_REPLY[plan["language"]]
    if plan.get("intent") == "farm_report":
        try:
            plan["days"] = max(1, min(365, int(plan.get("days") or 7)))
        except (TypeError, ValueError):
            plan["days"] = 7
    return plan


def detect_farm_intent_override(message: str) -> str | None:
    """Keyword pre-filter.

    Runs after the router to catch cases where it sends a clear farm-data
    request to tomato_general. Returns 'farm_status', 'farm_report', or None.
    """
    msg = message.lower()

    report_triggers = [
        "farm report", "farm ka report", "report dikhao", "report chahiye",
        "last week", "last month", "is hafte", "pichhle", "din ka report",
        "weekly report", "monthly report", "last 7", "last 10", "last 30",
        "pichle", "hafte ki",
    ]
    status_triggers = [
        "farm data", "farm status", "farm reading", "farm info",
        "sensor data", "khet ka data", "khet data",
        "show farm", "about the farm", "farm ka", "farm mein",
        "soil moisture", "tell me farm", "what is the farm",
        "abhi farm", "farm conditions",
    ]

    for kw in report_triggers:
        if kw in msg:
            return "farm_report"
    for kw in status_triggers:
        if kw in msg:
            return "farm_status"
    return None


# ── WEB ───────────────────────────────────────────────────────────────────────

def fetch_url_smart(url: str, user_question: str, max_words: int = 750) -> str:
    """Fetch a page and extract the ~750 words that answer the user's question."""
    try:
        resp = httpx.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=12,
            follow_redirects=True,
        )
        resp.raise_for_status()

        raw_text = re.sub(r"<[^>]+>", " ", resp.text)
        raw_text = re.sub(r"\s+", " ", raw_text).strip()

        return _llm_text(
            f"""From the webpage content below, extract the most relevant {max_words} words
that directly and specifically answer the user's question.
Write as clean flowing prose — no HTML artifacts, no broken text.

USER'S QUESTION: {user_question}

WEBPAGE CONTENT:
{raw_text[:6000]}

Return ONLY the extracted relevant content, nothing else."""
        )
    except Exception as e:
        return f"[Website could not be read: {e}]"


# ── PROMPT ────────────────────────────────────────────────────────────────────

def build_system_prompt(
    plan: dict,
    context_summary: str,
    kb_entry: dict | None,
    url_content: str | None,
    depth_mode: bool,
) -> str:
    """Builds a dynamic SYSTEM prompt for this specific turn."""
    intent = plan.get("intent", "tomato_general")
    language = plan.get("language", "hinglish")

    # Bypass for greetings / off-topic, where the router wrote the reply itself.
    if plan.get("direct_reply"):
        return (
            "[STRICT TASK] Output EXACTLY the following text and nothing else. "
            f"Do not add, change, or translate anything.\n\nTEXT: {plan['direct_reply']}"
        )

    if language == "hinglish":
        lang_rule = (
            "Speak in warm, natural Hinglish (mix of Hindi and English). "
            "Use simple Hindi words the farmer understands. "
            "Do NOT use overly technical English unless needed."
        )
    else:
        lang_rule = "Speak in clear, friendly, practical English."

    if depth_mode:
        depth_rule = (
            "[FORMAT] Give a detailed, structured answer. "
            "Use headings, bullet points, and bold key terms."
        )
    else:
        depth_rule = (
            "[FORMAT] Give a concise, practical answer in 3-5 sentences or short bullets. "
            "Do NOT pad the answer. Do NOT repeat the question back."
        )

    base_rules = (
        "You are Srish, a friendly 25-year-old expert agricultural botanist specializing in tomatoes.\n"
        "ANTI-HALLUCINATION RULES:\n"
        "1. Answer ONLY based on facts given to you in this prompt or standard agricultural knowledge.\n"
        "2. NEVER invent disease names, treatments, or chemicals not mentioned here.\n"
        "3. If you truly don't know, say so honestly.\n"
        "4. ALWAYS directly answer the user's actual question first.\n"
        f"{lang_rule}\n"
        f"{depth_rule}\n\n"
    )

    if intent == "url_query" and url_content:
        if "[Website could not be read" in url_content:
            return base_rules + (
                "[TASK] Tell the user you couldn't access their website, "
                "but answer from general knowledge."
            )
        return base_rules + (
            "[TASK] Answer using the website content below as your primary source.\n"
            f"WEBSITE CONTENT:\n{url_content}\n\n"
            f"USER'S QUESTION: {context_summary}"
        )

    if intent == "tomato_disease" and kb_entry:
        return base_rules + (
            "[STRICT TASK] Answer using ONLY the database record below. "
            "DO NOT suggest treatments not listed here.\n\n"
            f"Disease: {kb_entry.get('disease_name', '')}\n"
            f"Symptoms: {kb_entry.get('signs_and_symptoms', '')}\n"
            f"Treatment: {kb_entry.get('cure__control_and_management', '')}\n\n"
            f"USER'S QUESTION: {context_summary}"
        )

    if intent == "tomato_disease":
        return base_rules + (
            "[TASK] The user is describing a plant problem. "
            "Ask 1-2 clarifying questions about the symptoms (colour of leaves, which part of "
            "plant, how long, etc.) so you can better diagnose the issue. Be warm and helpful.\n\n"
            f"USER'S QUESTION: {context_summary}"
        )

    if intent in ("farm_status", "farm_report"):
        return base_rules + (
            "[TASK] Read the LIVE_FARM_DATA or FARM_REPORT_FACTS block at the top of this prompt "
            "and narrate those exact numbers to the user in a friendly, readable way. "
            "If the block starts with NO_DATA or NO_FARM_DATA, tell the user honestly there is no data. "
            "NEVER invent or estimate any sensor value.\n\n"
            f"USER'S QUESTION: {context_summary}"
        )

    if intent == "tomato_general":
        return base_rules + (
            "[TASK] Answer this tomato/farming question directly. "
            "CRITICAL: Only state facts you are 100% certain about. "
            "Key facts to never get wrong:\n"
            "- Tomato flowers are YELLOW, not white\n"
            "- Water needs: 1-2 liters every 2-3 days for pot plants, not liters per week\n"
            "- If asked about varieties, only mention: Pusa Ruby, Pusa Early Dwarf, Arka Vikas, "
            "Naveen, Punjab Chhuhara\n"
            "- If unsure about a specific fact, say 'mujhe is baare mein pakki jankari nahi' "
            "instead of guessing\n\n"
            f"USER'S QUESTION: {context_summary}"
        )

    return base_rules + "[TASK] Answer helpfully based on your agricultural expertise."


# ── ONE FULL TURN ─────────────────────────────────────────────────────────────

def _prepare(
    message: str,
    history: list | None,
    farm_id: str,
    depth_mode: bool,
    tomato_db: list | None,
) -> tuple[dict, list | None]:
    """Do everything up to the final generation call.

    Returns (out, messages). When messages is None the turn is already finished —
    out["reply"] is filled in, either by the router answering directly or by a
    hard exit such as the farm service being unreachable.
    """
    history = history or []
    db = tomato_db if tomato_db is not None else load_knowledge_base()

    plan = orchestrate(message, history, db)

    override = detect_farm_intent_override(message)
    if override and plan.get("intent") not in ("farm_status", "farm_report"):
        plan["intent"] = override
        plan["direct_reply"] = None
        plan["use_kb"] = False

    out = {
        "reply": "",
        "intent": plan.get("intent"),
        "language": plan.get("language"),
        "degraded": bool(plan.get("_degraded")),
        "used_kb": False,
        "kb_disease": None,
        "farm_rows": [],
        "error": plan.get("_error"),
    }

    # The router answered directly (off-topic refusal / greeting).
    if plan.get("direct_reply"):
        out["reply"] = plan["direct_reply"]
        return out, None

    kb_entry = None
    if plan.get("use_kb") and plan.get("kb_disease_name"):
        kb_entry = get_kb_entry_by_name(plan["kb_disease_name"], db)
        if kb_entry:
            out["used_kb"] = True
            out["kb_disease"] = kb_entry.get("disease_name")

    url_content = None
    if plan.get("url"):
        url_content = fetch_url_smart(plan["url"], message)

    farm_facts = None
    if plan.get("intent") == "farm_status":
        farm_facts = farm_data.facts_latest_status(farm_id)
    elif plan.get("intent") == "farm_report":
        farm_facts, rows = farm_data.get_report(farm_id, int(plan.get("days", 7)))
        out["farm_rows"] = rows

    # Hard exit: don't ask the model to narrate numbers that don't exist.
    if farm_facts and farm_facts.startswith(("NO_FARM_DATA", "NO_DATA")):
        out["reply"] = NO_DATA_REPLY
        out["error"] = farm_facts
        return out, None

    system_instruction = build_system_prompt(
        plan=plan,
        context_summary=plan.get("context_summary", message),
        kb_entry=kb_entry,
        url_content=url_content,
        depth_mode=depth_mode,
    )
    if farm_facts:
        system_instruction = (
            farm_facts
            + "\n\nUse ONLY the numbers above — do NOT invent or estimate any value.\n\n"
            + system_instruction
        )

    messages = [{"role": "system", "content": system_instruction}]
    if not plan.get("topic_changed") and plan.get("intent") not in ("greeting", "off_topic"):
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": message})

    return out, messages


UNAVAILABLE_REPLY = "Srish abhi jawab nahi de paa rahi. Thodi der baad try karein. 🌱"


def answer(
    message: str,
    history: list | None = None,
    farm_id: str = DEFAULT_FARM_ID,
    depth_mode: bool = False,
    tomato_db: list | None = None,
) -> dict:
    """Run one complete turn and return the reply plus how it was reached.

    history is [{"role": "user"|"assistant", "content": str}, ...], oldest first,
    NOT including the current message.

    Returns {reply, intent, language, degraded, used_kb, kb_disease, farm_rows, error}.
    """
    out, messages = _prepare(message, history, farm_id, depth_mode, tomato_db)
    if messages is None:
        return out

    try:
        resp = ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            options={"temperature": 0.0, "top_p": 0.5},
            keep_alive=KEEP_ALIVE,
        )
        out["reply"] = resp["message"]["content"]
    except Exception as e:
        log.error("Chat model unavailable: %s", e)
        out["reply"] = UNAVAILABLE_REPLY
        out["error"] = str(e)

    return out


def answer_stream(
    message: str,
    history: list | None = None,
    farm_id: str = DEFAULT_FARM_ID,
    depth_mode: bool = False,
    tomato_db: list | None = None,
):
    """Same turn as answer(), but yields the reply as it is generated.

    Yields ("meta", dict) once with the routing decision, then ("token", str)
    repeatedly, then ("done", dict) with the assembled reply. Routing still
    costs a few seconds up front — the win is that the user sees words appear
    instead of waiting for the whole answer.
    """
    out, messages = _prepare(message, history, farm_id, depth_mode, tomato_db)

    meta = {k: v for k, v in out.items() if k != "reply"}
    yield "meta", meta

    # Already finished: a router reply or a hard exit. Emit it as one chunk so
    # clients only need to handle the streaming shape.
    if messages is None:
        yield "token", out["reply"]
        yield "done", out
        return

    chunks: list[str] = []
    try:
        for part in ollama.chat(
            model=CHAT_MODEL,
            messages=messages,
            options={"temperature": 0.0, "top_p": 0.5},
            stream=True,
            keep_alive=KEEP_ALIVE,
        ):
            piece = part.get("message", {}).get("content", "")
            if piece:
                chunks.append(piece)
                yield "token", piece
    except Exception as e:
        log.error("Chat model unavailable: %s", e)
        out["error"] = str(e)
        if not chunks:
            chunks.append(UNAVAILABLE_REPLY)
            yield "token", UNAVAILABLE_REPLY

    out["reply"] = "".join(chunks)
    yield "done", out
