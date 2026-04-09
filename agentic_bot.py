import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_cropper import st_cropper
from PIL import Image
import io
import ollama
import json
import re
import httpx
import google.generativeai as genai
import os

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SRISH — Tomato Leaf Expert 🍅",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── GEMINI SETUP ──────────────────────────────────────────────────────────────
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
gemini = genai.GenerativeModel("gemini-1.5-flash")   # free tier model

# ── RAG: LOAD KNOWLEDGE BASE ──────────────────────────────────────────────────
@st.cache_data
def load_knowledge_base():
    try:
        with open("tomato_data_clean.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        return []

tomato_db = load_knowledge_base()


# ════════════════════════════════════════════════════════════════════════════════
#   GEMINI ORCHESTRATOR — pure intelligence, zero hardcoded keywords
# ════════════════════════════════════════════════════════════════════════════════

def gemini_orchestrate(user_message: str, chat_history: list, tomato_db: list) -> dict:
    """
    Gemini reads the full conversation and returns a structured JSON plan.
    """
 
    recent_turns = chat_history[-6:] if len(chat_history) > 6 else chat_history
    conv_snapshot = "\n".join(
        f"{m['role'].upper()}: {m['content'][:300]}" for m in recent_turns
    )
 
    kb_summary = []
    for entry in tomato_db:
        c = entry.get("content", {})
        kb_summary.append({
            "disease_name": c.get("disease_name", ""),
            "keywords": str(c.get("signs_and_symptoms", ""))[:200]
        })
    kb_json = json.dumps(kb_summary, ensure_ascii=False)
 
    orchestration_prompt = f"""You are the intelligence routing agent for "Srish", a tomato plant & disease expert chatbot.
 
Analyze the user's latest message and return a JSON plan.
 
=== RECENT CONVERSATION ===
{conv_snapshot}
 
=== USER'S LATEST MESSAGE ===
{user_message}
 
=== AVAILABLE KNOWLEDGE BASE DISEASES ===
{kb_json}
 
=== INTENT DEFINITIONS (read carefully) ===
- "greeting"        : hi, hello, who are you, namaste, etc.
- "tomato_disease"  : user describes symptoms, spots, drying, bugs, yellowing, wilting, any plant problem
- "tomato_general"  : ANY question about tomatoes or farming that is NOT a disease symptom.
                      THIS INCLUDES: what does a tomato plant look like, how to grow tomatoes,
                      watering, sunlight, appearance, color, shape, height, smell, taste, harvest, soil, etc.
- "url_query"       : user pastes a URL/link
- "off_topic"       : completely unrelated to farming, plants, food, agriculture (sports, movies, politics, etc.)
 
=== KB LOOKUP RULES ===
- Set "use_kb": true ONLY if intent is "tomato_disease" AND a disease from the KB list closely matches symptoms.
- If "use_kb" is true, set "kb_disease_name" to the EXACT disease name from the KB list.
- If no KB match, set "use_kb": false and "kb_disease_name": null.
 
=== STRICT RULES ===
1. Fill "reasoning" first. Think step by step about what the user is asking.
2. "tomato_general" covers ALL appearance, growth, gardening questions — do NOT mark these as off_topic.
3. Only "off_topic" for things truly unrelated to plants/food/farming (cricket, movies, politicians).
4. Detect language: "hinglish" if message uses Hindi words or Roman Hindi. Otherwise "english".
5. For off_topic, write a polite refusal in "direct_reply".
6. For all other intents, set "direct_reply": null — let Ollama answer.
7. Fill "reasoning" first.
8. BOUNCER RULE — these topics are ALWAYS "off_topic", no exceptions:
   - Cricket, IPL, sports players (Kohli, Dhoni, Tendulkar, Messi, etc.)
   - Movies, actors, directors
   - Politicians, political parties
   - Music, singers
   - Technology companies (Google, Apple, etc.) unrelated to farming
   - Any person's name that is not a scientist or farmer
   If the user mentions ANY of the above, set intent="off_topic" and write a refusal in "direct_reply".
9. A person's NAME (Kohli, Sachin, Modi, Shah Rukh) is NEVER a plant, disease, bacterium, or virus.
   Do NOT invent a disease named after a person.
10. "tomato_general" covers ALL appearance, growth, gardening questions.
11. Detect language: "hinglish" if message uses Hindi words or Roman Hindi.
12.Also don't answer the terms like ipl, fifa like things or any short from or word that is not related to the tomatoes and it's diseases.
=== EXAMPLES ===
 
User: "waise ye tomato plant dikhta kaisa hai"
JSON: {{"reasoning": "User is asking about the appearance/looks of a tomato plant. This is a general tomato question about how the plant looks.", "intent": "tomato_general", "use_kb": false, "kb_disease_name": null, "url": null, "topic_changed": true, "context_summary": "User wants to know what a tomato plant looks like — its appearance, shape, leaves, height, color.", "direct_reply": null, "language": "hinglish"}}
 
User: "how the tomato plant looks like"
JSON: {{"reasoning": "User is asking about the visual appearance of a tomato plant. This is a general tomato question.", "intent": "tomato_general", "use_kb": false, "kb_disease_name": null, "url": null, "topic_changed": false, "context_summary": "User wants to know what a tomato plant looks like — its appearance, shape, leaves, color, height.", "direct_reply": null, "language": "english"}}
 
User: "meri plant ke patte peele ho rahe hain"
JSON: {{"reasoning": "User says their plant leaves are turning yellow. This is a disease/health symptom.", "intent": "tomato_disease", "use_kb": true, "kb_disease_name": "Early Blight", "url": null, "topic_changed": true, "context_summary": "User's tomato plant leaves are turning yellow.", "direct_reply": null, "language": "hinglish"}}
 
User: "Do you know Kohli?"
JSON: {{"reasoning": "User is asking about Virat Kohli, a cricketer. Completely unrelated to tomatoes.", "intent": "off_topic", "use_kb": false, "kb_disease_name": null, "url": null, "topic_changed": true, "context_summary": "User is asking about cricket.", "direct_reply": "Main ek tomato crop expert hoon! Mujhe cricket ke baare mein nahi pata. Kya main aapki fasal ki koi madad kar sakti hoon? 🍅", "language": "hinglish"}}
 
User: "Mera plant sookh raha hai"
JSON: {{"reasoning": "User says their plant is drying up. This is a symptom of a problem.", "intent": "tomato_disease", "use_kb": false, "kb_disease_name": null, "url": null, "topic_changed": true, "context_summary": "User's tomato plant is drying up and wilting.", "direct_reply": null, "language": "hinglish"}}

User: "do you know about kohli"
JSON: {{"reasoning": "Kohli is Virat Kohli, a famous Indian cricketer. This has nothing to do with tomatoes or farming.", "intent": "off_topic", "use_kb": false, "kb_disease_name": null, "url": null, "topic_changed": true, "context_summary": "User asked about cricketer Virat Kohli.", "direct_reply": "Main sirf tomato aur farming ke baare mein jaanti hoon! Kohli ke baare mein mujhse mat poochho 😄 Kya aapke tomato plants mein koi problem hai? 🍅", "language": "hinglish"}}

User: "do you know about ipl"
JSON: {{"reasoning": "IPL is the Indian Premier League, a cricket tournament. This is completely unrelated to tomatoes or farming.", "intent": "off_topic", "use_kb": false, "kb_disease_name": null, "url": null, "topic_changed": true, "context_summary": "User asked about IPL cricket.", "direct_reply": "IPL toh mujhe nahi pata Ji! Main toh bas tomato ki expert hoon 🍅 Kya khet mein koi dikkat hai?", "language": "hinglish"}}

User: "who is modi"
JSON: {{"reasoning": "Modi refers to a political leader. This is off-topic for a tomato expert bot.", "intent": "off_topic", "use_kb": false, "kb_disease_name": null, "url": null, "topic_changed": true, "context_summary": "User asked about a politician.", "direct_reply": "Politics mera subject nahi hai! Main sirf aapke tomato plants ki baat kar sakti hoon 🌿", "language": "hinglish"}}

User: "waise ye tomato plant dikhta kaisa hai"
JSON: {{"reasoning": "User is asking about the appearance of a tomato plant. General tomato question.", "intent": "tomato_general", "use_kb": false, "kb_disease_name": null, "url": null, "topic_changed": true, "context_summary": "User wants to know what a tomato plant looks like.", "direct_reply": null, "language": "hinglish"}}

User: "meri plant ke patte peele ho rahe hain"
JSON: {{"reasoning": "User says leaves are turning yellow. This is a disease symptom.", "intent": "tomato_disease", "use_kb": true, "kb_disease_name": "Early Blight", "url": null, "topic_changed": true, "context_summary": "User's tomato plant leaves are turning yellow.", "direct_reply": null, "language": "hinglish"}} 
=== YOUR TASK ===
Return ONLY valid JSON. No markdown fences. No extra text.
"""
    try:
        response = gemini.generate_content(orchestration_prompt)
        raw = response.text.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        plan = json.loads(raw)
        return plan
    except Exception:
        return {
            "intent": "tomato_general",
            "use_kb": False,
            "kb_disease_name": None,
            "url": None,
            "topic_changed": False,
            "context_summary": user_message,
            "direct_reply": None,
            "language": "english"
        }

def get_kb_entry_by_name(disease_name: str, db: list) -> dict | None:
    """Fetch full KB entry by disease name that Gemini identified."""
    if not disease_name or not db:
        return None
    name_lower = disease_name.lower().strip()
    for entry in db:
        c = entry.get("content", {})
        if c.get("disease_name", "").lower().strip() == name_lower:
            return c
    # Fuzzy fallback
    for entry in db:
        c = entry.get("content", {})
        db_name = c.get("disease_name", "").lower()
        if name_lower in db_name or db_name in name_lower:
            return c
    return None


def fetch_url_smart(url: str, user_question: str, max_words: int = 750) -> str:
    """
    Fetch a webpage and use Gemini to extract the most relevant ~750 words
    that directly answer the user's question. Smart, not mechanical.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        resp = httpx.get(url, headers=headers, timeout=12, follow_redirects=True)
        resp.raise_for_status()

        # Strip HTML tags
        raw_text = re.sub(r"<[^>]+>", " ", resp.text)
        raw_text = re.sub(r"\s+", " ", raw_text).strip()

        extraction_prompt = f"""From the webpage content below, extract the most relevant {max_words} words 
that directly and specifically answer the user's question.
Write as clean flowing prose — no HTML artifacts, no broken text.
Focus only on content relevant to the question.

USER'S QUESTION: {user_question}

WEBPAGE CONTENT:
{raw_text[:6000]}

Return ONLY the extracted relevant content, nothing else."""

        extraction = gemini.generate_content(extraction_prompt)
        return extraction.text.strip()

    except Exception as e:
        return f"[Website could not be read: {e}]"


def build_system_prompt(plan: dict, context_summary: str, kb_entry: dict | None, url_content: str | None, depth_mode: bool) -> str:
    """Builds a dynamic SYSTEM prompt for this specific turn."""
 
    intent = plan.get("intent", "tomato_general")
    language = plan.get("language", "hinglish")
 
    # AGENTIC BYPASS for greetings / off-topic
    if plan.get("direct_reply"):
        return (
            f"[STRICT TASK] Output EXACTLY the following text and nothing else. "
            f"Do not add, change, or translate anything.\n\nTEXT: {plan['direct_reply']}"
        )
 
    # ── LANGUAGE RULE ──
    if language == "hinglish":
        lang_rule = (
            "Speak in warm, natural Hinglish (mix of Hindi and English). "
            "Use simple Hindi words the farmer understands. "
            "Do NOT use overly technical English unless needed."
        )
    else:
        lang_rule = "Speak in clear, friendly, practical English."
 
    # ── DEPTH RULE ──
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
 
    # ── URL QUERY ──
    if intent == "url_query" and url_content:
        if "[Website could not be read" in url_content:
            return base_rules + "[TASK] Tell the user you couldn't access their website, but answer from general knowledge."
        return base_rules + (
            "[TASK] Answer using the website content below as your primary source.\n"
            f"WEBSITE CONTENT:\n{url_content}\n\n"
            f"USER'S QUESTION: {context_summary}"
        )
 
    # ── DISEASE WITH KB ENTRY ──
    if intent == "tomato_disease" and kb_entry:
        return base_rules + (
            "[STRICT TASK] Answer using ONLY the database record below. "
            "DO NOT suggest treatments not listed here.\n\n"
            f"Disease: {kb_entry.get('disease_name', '')}\n"
            f"Symptoms: {kb_entry.get('signs_and_symptoms', '')}\n"
            f"Treatment: {kb_entry.get('cure__control_and_management', '')}\n\n"
            f"USER'S QUESTION: {context_summary}"
        )
 
    # ── DISEASE WITHOUT KB ENTRY ──
    if intent == "tomato_disease":
        return base_rules + (
            "[TASK] The user is describing a plant problem. "
            "Ask 1-2 clarifying questions about the symptoms (color of leaves, which part of plant, how long, etc.) "
            "so you can better diagnose the issue. Be warm and helpful.\n\n"
            f"USER'S QUESTION: {context_summary}"
        )
 
    # ── GENERAL TOMATO QUESTION (appearance, growth, farming, etc.) ──
    if intent == "tomato_general":
        return base_rules + (
        "[TASK] Answer this tomato/farming question directly. "
        "CRITICAL: Only state facts you are 100% certain about. "
        "Key facts to never get wrong:\n"
        "- Tomato flowers are YELLOW, not white\n"
        "- Water needs: 1-2 liters every 2-3 days for pot plants, not liters per week\n"
        "- If asked about varieties, only mention: Pusa Ruby, Pusa Early Dwarf, Arka Vikas, Naveen, Punjab Chhuhara\n"
        "- If unsure about a specific fact, say 'mujhe is baare mein pakki jankari nahi' instead of guessing\n\n"
        f"USER'S QUESTION: {context_summary}"
    )
 
    return base_rules + "[TASK] Answer helpfully based on your agricultural expertise."
# ════════════════════════════════════════════════════════════════════════════════
#   CSS & ANIMATIONS
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
  --cream:        #fffdf7;
  --soft-green:   #5a8a5a;
  --light-green:  #e8f5e8;
  --tomato:       #e8503a;
  --tomato-light: #fde8e5;
  --leaf:         #4a7c59;
  --text:         #3a3028;
  --text-light:   #7a6e66;
  --petal-yellow: #fdf4d0;
  --border:       rgba(90,138,90,0.18);
  --card-shadow:  0 4px 40px rgba(90,138,90,0.10), 0 1px 8px rgba(0,0,0,0.05);
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--cream) !important;
  font-family: 'Lato', sans-serif !important;
  font-weight: 300;
  color: var(--text);
}
[data-testid="stAppViewContainer"] > .main > .block-container {
  max-width: 760px !important;
  padding: 0 24px 80px !important;
  margin: 0 auto;
}
[data-testid="stHeader"], [data-testid="stToolbar"],
section[data-testid="stSidebar"] { display: none !important; }
footer { visibility: hidden; }

@keyframes petalFall {
  0%   { transform: translateY(-60px) rotate(0deg) scale(0.8); opacity: 0; }
  10%  { opacity: 0.5; }
  90%  { opacity: 0.25; }
  100% { transform: translateY(110vh) rotate(400deg) scale(1); opacity: 0; }
}
.petals-wrap { position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }
.petal { position: absolute; border-radius: 60% 10% 60% 10%; opacity: 0; animation: petalFall linear infinite; }

.blob { position: fixed; border-radius: 50%; filter: blur(80px); pointer-events: none; z-index: 0; }
.blob-1 { width:520px; height:520px; background:rgba(232,80,58,0.07);  top:-120px; right:-120px; }
.blob-2 { width:400px; height:400px; background:rgba(90,138,90,0.08);  bottom:-100px; left:-100px; }
.blob-3 { width:280px; height:280px; background:rgba(253,244,208,0.70); top:40%; left:35%; }

@keyframes fadeDown {
  from { opacity: 0; transform: translateY(-18px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes bobble {
  0%,100% { transform: translateY(0) rotate(-4deg); }
  50%     { transform: translateY(-8px) rotate(4deg); }
}
.srish-header {
  text-align: center; padding: 56px 0 36px;
  animation: fadeDown 0.9s ease both;
  position: relative; z-index: 1;
}
.header-tomato { display: block; font-size: 3.6rem; animation: bobble 3.5s ease-in-out infinite; margin-bottom: 6px; }
.site-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(2.2rem, 6vw, 3.2rem); font-weight: 700;
  color: var(--tomato); letter-spacing: -0.01em; line-height: 1.1; margin: 0;
}
.site-title em { color: var(--leaf); font-style: italic; }
.site-tagline { margin-top: 14px; font-size: 1rem; color: var(--text-light); line-height: 1.75; max-width: 440px; margin-left: auto; margin-right: auto; }
.leaf-divider { display: flex; align-items: center; gap: 12px; justify-content: center; margin-top: 22px; opacity: 0.45; }
.leaf-divider::before, .leaf-divider::after { content: ''; flex: 1; max-width: 110px; height: 1px; background: linear-gradient(90deg, transparent, var(--soft-green), transparent); }

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(24px); }
  to   { opacity: 1; transform: translateY(0); }
}
.upload-card {
  background: #ffffff; border: 1.5px solid var(--border); border-radius: 24px;
  padding: 40px 36px 36px; box-shadow: var(--card-shadow);
  animation: fadeUp 0.9s ease 0.15s both; position: relative; z-index: 1; overflow: hidden; margin-top: 4px;
}
.upload-card::before { content:'🌿'; position:absolute; top:14px; left:18px; font-size:1.3rem; opacity:0.22; transform:rotate(-20deg); }
.upload-card::after  { content:'🌸'; position:absolute; top:14px; right:18px; font-size:1.2rem; opacity:0.22; transform:rotate(15deg); }
.card-label    { font-family:'Playfair Display',serif; font-size:1.35rem; font-weight:700; color:var(--text); text-align:center; margin-bottom:5px; }
.card-sublabel { text-align:center; font-size:0.88rem; color:var(--text-light); margin-bottom:24px; }

[data-testid="stFileUploader"] { background:var(--light-green) !important; border:2px dashed rgba(90,138,90,0.30) !important; border-radius:18px !important; }
[data-testid="stFileUploader"]:hover { border-color:var(--soft-green) !important; background:#d4ecd4 !important; }
[data-testid="stFileUploader"] label { font-family:'Playfair Display',serif !important; color:var(--leaf) !important; font-size:1rem !important; }
[data-testid="stFileUploaderDropzone"] { background:transparent !important; border:none !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { color:var(--text-light) !important; font-size:0.82rem !important; }
[data-testid="stBaseButton-secondary"] {
  background:var(--tomato) !important; color:#fff !important; border:none !important;
  border-radius:50px !important; font-family:'Lato',sans-serif !important;
  box-shadow:0 4px 14px rgba(232,80,58,0.30) !important;
}
[data-testid="stBaseButton-secondary"]:hover { box-shadow:0 6px 22px rgba(232,80,58,0.48) !important; transform:translateY(-1px) !important; }

.result-card { border-radius:18px; padding:22px 26px; margin-top:24px; border:1.5px solid var(--border); box-shadow:var(--card-shadow); animation:fadeUp 0.5s ease both; }
.result-card.success { background:linear-gradient(135deg,#f0faf0 0%,#e8f5e8 100%); border-color:rgba(90,138,90,0.25); }
.result-card.error   { background:linear-gradient(135deg,#fff5f4 0%,#fde8e5 100%); border-color:rgba(232,80,58,0.25); }
.result-card.warning { background:var(--petal-yellow); border-color:rgba(232,80,58,0.15); }
.result-title { font-family:'Playfair Display',serif; font-size:1.15rem; font-weight:700; margin-bottom:6px; }
.result-title.green { color:var(--leaf); }
.result-title.red   { color:var(--tomato); }
.result-desc { font-size:0.88rem; color:var(--text-light); line-height:1.7; }

.conf-bar-wrap { margin-top:16px; background:rgba(90,138,90,0.12); border-radius:50px; height:8px; overflow:hidden; }
.conf-bar-fill { height:100%; border-radius:50px; background:linear-gradient(90deg,var(--soft-green),var(--tomato)); transition:width 1s ease; }
.conf-label    { font-size:0.78rem; color:var(--text-light); margin-top:6px; text-align:right; }

.how-title { font-family:'Playfair Display',serif; font-size:1.45rem; color:var(--text); text-align:center; margin-bottom:24px; }
.how-title em { color:var(--tomato); font-style:italic; }

.srish-footer { margin-top:60px; text-align:center; font-size:0.78rem; color:var(--text-light); opacity:0.55; position:relative; z-index:1; }
.srish-footer span { color:var(--tomato); }

[data-testid="stSpinner"] { color:var(--leaf) !important; }
[data-testid="stImage"]   { border-radius:16px; overflow:hidden; }

.stChatMessage { background-color:rgba(255,255,255,0.6) !important; border-radius:15px; padding:10px; margin-bottom:10px; }
.stChatMessage p, .stChatMessage div, [data-testid="stChatMessageContent"] { color:var(--text) !important; }
[data-testid="stChatInput"] textarea { color:var(--text) !important; }

.intel-badge {
  display:inline-flex; align-items:center; gap:6px;
  background:linear-gradient(135deg,#f0faf0,#e8f5e8);
  border:1px solid rgba(90,138,90,0.3); border-radius:50px;
  padding:4px 14px; font-size:0.75rem; color:var(--leaf);
  font-family:'Lato',sans-serif; font-weight:700; letter-spacing:0.03em;
}
</style>

<div class="petals-wrap" id="petalsBg"></div>
<div class="blob blob-1"></div>
<div class="blob blob-2"></div>
<div class="blob blob-3"></div>

<script>
(function(){
  const bg = document.getElementById('petalsBg');
  if (!bg) return;
  const colors = ['#f9c8c0','#fde0cc','#d4edcc','#fdf3c0','#f5d5e5','#e0f0e0','#fce8d4'];
  for (let i = 0; i < 22; i++) {
    const p = document.createElement('div');
    p.className = 'petal';
    const sz = 8 + Math.random() * 16;
    p.style.cssText = `width:${sz}px;height:${sz*0.55}px;background:${colors[Math.floor(Math.random()*colors.length)]};left:${Math.random()*100}%;border-radius:${40+Math.random()*20}% ${8+Math.random()*10}% ${40+Math.random()*20}% ${8+Math.random()*10}%;animation-duration:${11+Math.random()*13}s;animation-delay:${-Math.random()*14}s;`;
    bg.appendChild(p);
  }
})();
</script>
""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="srish-header">
  <span class="header-tomato">🍅</span>
  <h1 class="site-title">SRISH &mdash; <em>Tomato Expert</em></h1>
  <p class="site-tagline">Leaf di Photo Upload Krdo</p>
  <div class="leaf-divider">🌿</div>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_gatekeeper():
    return YOLO("best.pt")

@st.cache_resource
def load_yolo():
    return YOLO("yolo.pt")

try:
    gatekeeper = load_gatekeeper()
    gk_loaded  = True
except Exception as e:
    gk_loaded = False
    st.markdown(f"""
    <div class="result-card error">
      <div class="result-title red">⚠️ Gatekeeper Model Missing</div>
      <div class="result-desc">Could not load the YOLO Gatekeeper model.<br>{e}</div>
    </div>""", unsafe_allow_html=True)

try:
    yolo_model  = load_yolo()
    yolo_loaded = True
except Exception as e:
    yolo_loaded = False
    st.markdown(f"""
    <div class="result-card error">
      <div class="result-title red">⚠️ Disease Model Missing</div>
      <div class="result-desc">Could not load the Leaf Miner detection model.<br>{e}</div>
    </div>""", unsafe_allow_html=True)

# ── GATEKEEPER ────────────────────────────────────────────────────────────────
def check_if_leaf(frame):
    results      = gatekeeper.predict(source=frame, verbose=False)
    top_class_id = results[0].probs.top1
    confidence   = float(results[0].probs.top1conf.item())
    class_name   = results[0].names[top_class_id]
    if class_name == 'Tomato_Leaf' and confidence >= 0.65:
        return True, confidence
    return False, confidence

def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bytes_to_pil(raw_bytes):
    return Image.open(io.BytesIO(raw_bytes))

# ── UPLOAD CARD ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-card">
  <div class="card-label">Analyse Your Leaf</div>
  <p class="card-sublabel">Dear User, Kindly Drop your Leaf Image — or take one live with your camera!</p>
</div>
""", unsafe_allow_html=True)

input_method = st.radio(
    "Input method",
    options=["📁 Upload Image", "📷 Take Photo (Camera)"],
    horizontal=True,
    label_visibility="collapsed",
)

raw_pil_image = None

if input_method == "📁 Upload Image":
    uploaded_img = st.file_uploader(
        "Upload leaf photo (JPG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        label_visibility="collapsed",
    )
    if uploaded_img is not None:
        raw_pil_image = bytes_to_pil(uploaded_img.read())
else:
    camera_photo = st.camera_input("📷 Point your camera at the tomato leaf and click the shutter button")
    if camera_photo is not None:
        raw_pil_image = bytes_to_pil(camera_photo.read())

# ── CROP ──────────────────────────────────────────────────────────────────────
final_pil_image = None

if raw_pil_image is not None:
    st.markdown("---")
    st.markdown("### ✂️ Would you like to crop the image before analysis?")
    crop_choice = st.radio(
        "Crop choice",
        options=["✅ Yes, let me crop it", "➡️ No, use the full image"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if crop_choice == "✅ Yes, let me crop it":
        st.info("🖱️ Drag the handles to select the area you want, then click **Proceed with Crop**.")
        cropped = st_cropper(raw_pil_image, realtime_update=True, box_color="#e8503a", aspect_ratio=None)
        st.image(cropped, caption="Preview of cropped area", use_column_width=True)
        if st.button("✅ Proceed with Crop"):
            final_pil_image = cropped
            st.success("Crop applied! Running analysis…")
    else:
        final_pil_image = raw_pil_image
        st.image(raw_pil_image, caption="Your image (full)", use_column_width=True)

# ── ANALYSIS ──────────────────────────────────────────────────────────────────
if final_pil_image is not None and gk_loaded:
    frame = pil_to_cv2(final_pil_image)

    with st.spinner("🌿 SRISH is verifying the leaf…"):
        is_leaf, leaf_score = check_if_leaf(frame)

    if is_leaf:
        bar_pct = min(max(leaf_score * 100, 0), 100)
        st.markdown(f"""
        <div class="result-card success">
          <div class="result-title green">✅ Tomato Leaf Confirmed!</div>
          <div class="result-desc">
            The gatekeeper recognised a genuine tomato leaf. Handing off to the AI disease scanner…
          </div>
          <div class="conf-bar-wrap">
            <div class="conf-bar-fill" style="width:{bar_pct:.0f}%"></div>
          </div>
          <div class="conf-label">Confidence score: {leaf_score*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        if yolo_loaded:
            with st.spinner("🎯 Running AI disease detection…"):
                results       = yolo_model.predict(source=frame, conf=0.25)
                annotated_bgr = results[0].plot()
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            num_det = len(results[0].boxes) if results[0].boxes is not None else 0

            if num_det > 0:
                st.markdown(f"""
                <div class="result-card error">
                  <div class="result-title red">🔬 Leaf Miner Damage Detected</div>
                  <div class="result-desc">
                    SRISH found <strong>{num_det} region{'s' if num_det!=1 else ''}</strong>
                    of potential Leaf Miner damage. The annotated image below shows exactly where.
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-card success">
                  <div class="result-title green">🌟 No Disease Detected</div>
                  <div class="result-desc">
                    Great news — SRISH found no signs of Leaf Miner damage. Your tomato plant looks healthy!
                  </div>
                </div>""", unsafe_allow_html=True)

            st.image(annotated_rgb, caption="SRISH · YOLOv8 Detection Results", use_column_width=True)
            st.balloons()

    else:
        st.markdown(f"""
        <div class="result-card error">
          <div class="result-title red">❌ Not a Tomato Leaf</div>
          <div class="result-desc">
            SRISH's gatekeeper couldn't confirm this is a tomato leaf.
            Please upload a clear, well-lit photo of a single tomato leaf.<br><br>
            <em>Score: {leaf_score*100:.1f}% — threshold is 60.0%</em>
          </div>
        </div>
        <div class="result-card warning" style="margin-top:12px;">
          <div class="result-title" style="color:var(--text)">💡 Tips for a better photo</div>
          <div class="result-desc">
            • Place the leaf on a plain light background<br>
            • Ensure the whole leaf is in frame &amp; in focus<br>
            • Avoid heavy shadows or very dark lighting<br>
            • Use JPG or PNG format
          </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
#   INTELLIGENT CHAT INTERFACE
# ════════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='margin-top:60px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div class="how-title">
  Chat with <em>Srish</em><br>
  <span class="intel-badge">🧠 Sponsored by DMSL</span>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello Ji! Main hoon Srish 🍅 — aapki tomato crop ki expert. "
                "Leaf ki photo upload karo upar, ya mujhse tomato diseases ke baare mein "
                "kuch bhi poochho. Main yahan hoon aapki help ke liye! 🌿"
            )
        }
    ]

# Display chat history
for message in st.session_state.messages:
    avatar = "🍅" if message["role"] == "assistant" else "🧑‍🌾"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ── CHAT INPUT ────────────────────────────────────────────────────────────────

# 1. Add the toggle switch right above the chat input
depth_mode = st.toggle("Detailed Answer Mode", help="Turn this on to get highly detailed, point-by-point answers like ChatGPT.")

if prompt := st.chat_input("Ask Srish about your tomatoes..."):

    st.chat_message("user", avatar="🧑‍🌾").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🍅"):
        try:
            with st.spinner("🧠 Understanding your question..."):
                plan = gemini_orchestrate(
                    user_message=prompt,
                    chat_history=st.session_state.messages[:-1],
                    tomato_db=tomato_db
                )

            # ── EARLY EXIT: Gemini already wrote the reply (off_topic / greeting) ──
            if plan.get("direct_reply"):
                reply = plan["direct_reply"]
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

            else:
                # ── Only reach Ollama for real tomato questions ──
                kb_entry = None
                if plan.get("use_kb") and plan.get("kb_disease_name"):
                    kb_entry = get_kb_entry_by_name(plan["kb_disease_name"], tomato_db)

                url_content = None
                if plan.get("url"):
                    with st.spinner("🌐 Reading website for you..."):
                        url_content = fetch_url_smart(plan["url"], prompt)

                system_instruction = build_system_prompt(
                    plan=plan,
                    context_summary=plan.get("context_summary", prompt),
                    kb_entry=kb_entry,
                    url_content=url_content,
                    depth_mode=depth_mode
                )

                messages_for_ollama = [{"role": "system", "content": system_instruction}]

                intent = plan.get("intent", "tomato_general")
                topic_changed = plan.get("topic_changed", False)

                if not topic_changed and intent not in ("greeting", "off_topic"):
                    recent_history = st.session_state.messages[-7:-1]  # excludes current user msg
                    messages_for_ollama.extend(recent_history)

                messages_for_ollama.append({"role": "user", "content": prompt})

                with st.spinner("🌿 Srish is preparing your answer..."):
                    response = ollama.chat(
                        model='SrishBot',
                        messages=messages_for_ollama,
                        options={"temperature": 0.0, "top_p": 0.5}
                    )

                reply = response['message']['content']
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

        except Exception as e:
            st.error(f"⚠️ Srish could not respond right now. Error: {e}")
# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="srish-footer">
  Made with <span>♥</span> &nbsp;·&nbsp; SRISH Tomato Leaf Expert
  &nbsp;·&nbsp; By ?+
</div>
""", unsafe_allow_html=True)