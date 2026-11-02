import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_cropper import st_cropper
from PIL import Image
import io
import ollama
import farm_data
import pandas as pd

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SRISH — Tomato Leaf Expert 🍅",
    page_icon="🍅",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── BRAIN ──────────────────────────────────────────────────────────────────────
# Routing, prompt building and the farm/web lookups all live in srish_core,
# which api_server.py shares. Keeping them there means the Streamlit UI and the
# Srijan mobile app can never drift apart in how they answer. This file is only
# the UI on top of that brain.
import srish_core
from srish_core import (
    build_system_prompt,
    detect_farm_intent_override,
    fetch_url_smart,
    get_kb_entry_by_name,
    orchestrate,
)


@st.cache_data
def load_knowledge_base():
    """Streamlit-cached view of the shared KB loader."""
    return srish_core.load_knowledge_base()


tomato_db = load_knowledge_base()
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
    return YOLO(r"C:\Users\Prateek\Downloads\best (5).pt")

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
  <span class="intel-badge">🧠</span>
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
                plan = orchestrate(
                    user_message=prompt,
                    chat_history=st.session_state.messages[:-1],
                    tomato_db=tomato_db
                )

            # ── KEYWORD OVERRIDE: correct Gemini mis-routing for farm queries ──
            if plan.get("_degraded"):
                st.warning(
                    "⚠️ Router unavailable — answering without intent routing, so the "
                    "off-topic filter and the knowledge base were both skipped."
                )

            farm_override = detect_farm_intent_override(prompt)
            if farm_override and plan.get("intent") not in ("farm_status", "farm_report"):
                plan["intent"] = farm_override
                plan["direct_reply"] = None
                plan["use_kb"] = False
                plan["context_summary"] = plan.get("context_summary", prompt)

            # ── EARLY EXIT: the router already wrote the reply (off_topic / greeting) ──
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

                # ── FARM DATA (live Supabase facts) ──
                farm_id = "FARM_001"
                farm_facts = None
                if plan.get("intent") == "farm_status":
                    farm_facts = farm_data.facts_latest_status(farm_id)
                elif plan.get("intent") == "farm_report":
                    days = int(plan.get("days", 7))
                    farm_facts, rows = farm_data.get_report(farm_id, days)
                    if rows:
                        df = pd.DataFrame(rows)
                        if "recorded_at" in df and "vwc_percent" in df:
                            st.line_chart(df.set_index("recorded_at")[["vwc_percent"]])

                # ── HARD NO-DATA EXIT: skip Ollama entirely if Supabase has nothing ──
                if farm_facts and farm_facts.startswith(("NO_FARM_DATA", "NO_DATA")):
                    reply = (
                        "Abhi koi farm sensor data available nahi hai. 🌱\n\n"
                        "Sensor se koi reading nahi aayi — please thodi der baad try karein, "
                        "ya check karein ki sensor connected hai."
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()

                system_instruction = build_system_prompt(
                    plan=plan,
                    context_summary=plan.get("context_summary", prompt),
                    kb_entry=kb_entry,
                    url_content=url_content,
                    depth_mode=depth_mode
                )
                # ── Inject real farm facts into the system prompt ──
                if farm_facts:
                    system_instruction = (
                        farm_facts
                        + "\n\nUse ONLY the numbers above — do NOT invent or estimate any value.\n\n"
                        + system_instruction
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