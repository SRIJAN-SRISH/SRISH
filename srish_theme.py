"""The Srish palette, shared so the chat app and the admin portal match.

agentic_bot.py keeps its own narrow, decorative layout. The portal needs wide
tables and charts, so it takes the same tokens, fonts and motion but its own
layout.
"""

PALETTE = {
    "cream": "#fffdf7",
    "soft_green": "#5a8a5a",
    "light_green": "#e8f5e8",
    "tomato": "#e8503a",
    "tomato_light": "#fde8e5",
    "leaf": "#4a7c59",
    "text": "#3a3028",
    "text_light": "#7a6e66",
    "petal_yellow": "#fdf4d0",
    "border": "rgba(90,138,90,0.18)",
}

# Ordered for charts: distinguishable, and on-brand rather than Streamlit default.
CHART_COLORS = ["#4a7c59", "#e8503a", "#5a8a5a", "#c9a227", "#7a6e66", "#8e6f9e"]

# Graphviz accepts names, #hex and hsv -- not rgba(), which it silently drops.
# This is PALETTE["border"] flattened against the cream background.
GRAPH_BORDER = "#cdddc9"

FONTS = (
    "@import url('https://fonts.googleapis.com/css2?"
    "family=Playfair+Display:ital,wght@0,400;0,700;1,400"
    "&family=Lato:wght@300;400;700;900&display=swap');"
)


def background() -> str:
    """The drifting blobs from the chat app, kept for visual continuity.

    Fixed and behind everything, so they never intercept a click.
    """
    return """
<div class="bg-blobs">
  <div class="blob blob-1"></div>
  <div class="blob blob-2"></div>
  <div class="blob blob-3"></div>
</div>
"""


def portal_css() -> str:
    """CSS for the admin portal: same palette, dashboard layout, larger type."""
    p = PALETTE
    return f"""
<style>
{FONTS}

*, *::before, *::after {{ box-sizing: border-box; }}

:root {{
  --cream:        {p['cream']};
  --soft-green:   {p['soft_green']};
  --light-green:  {p['light_green']};
  --tomato:       {p['tomato']};
  --tomato-light: {p['tomato_light']};
  --leaf:         {p['leaf']};
  --text:         {p['text']};
  --text-light:   {p['text_light']};
  --petal:        {p['petal_yellow']};
  --border:       {p['border']};
  --shadow:       0 4px 40px rgba(90,138,90,0.10), 0 1px 8px rgba(0,0,0,0.05);
  --shadow-lift:  0 14px 48px rgba(90,138,90,0.20), 0 3px 12px rgba(0,0,0,0.07);
  --ease:         cubic-bezier(.22,.8,.26,1);
}}

/* ── TYPE SCALE ──────────────────────────────────────────────────────────────
   Deliberately large: this gets read on a projector and over someone's
   shoulder, not just at arm's length on the author's laptop. */
html, body, [data-testid="stAppViewContainer"] {{
  background: var(--cream) !important;
  font-family: 'Lato', sans-serif !important;
  color: var(--text);
  font-size: 18px;
}}
[data-testid="stAppViewContainer"] * {{ font-family: 'Lato', sans-serif; }}
[data-testid="stAppViewContainer"] > .main > .block-container {{
  max-width: 1340px !important;
  padding: 1.5rem 2rem 6rem !important;
}}
[data-testid="stToolbar"] {{ display: none !important; }}
footer {{ visibility: hidden; }}

h1,h2,h3,h4 {{ font-family: 'Playfair Display', serif !important; color: var(--leaf); }}
h3 {{ font-size: 1.5rem !important; margin-top: 1.6rem !important; }}
h4 {{ font-size: 1.25rem !important; }}
p, li, label, .stMarkdown {{ font-size: 1.02rem; line-height: 1.7; }}

/* Tables and charts carry numbers people squint at, so bump them too. */
[data-testid="stDataFrame"] {{ font-size: 1rem !important; }}
[data-testid="stMetricValue"] {{ font-size: 2.1rem !important; font-weight: 900 !important; }}
[data-testid="stMetricLabel"] {{ font-size: 0.95rem !important; color: var(--text-light) !important; }}
.stTextInput input, .stSelectbox div, [data-testid="stChatInput"] textarea {{
  font-size: 1.02rem !important;
}}

/* ── BACKGROUND ──────────────────────────────────────────────────────────── */
.bg-blobs {{ position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden; }}
.blob {{ position: absolute; border-radius: 50%; filter: blur(85px); will-change: transform; }}
.blob-1 {{ width:520px; height:520px; background:rgba(232,80,58,0.07);
           top:-130px; right:-130px; animation: drift1 26s var(--ease) infinite; }}
.blob-2 {{ width:420px; height:420px; background:rgba(90,138,90,0.09);
           bottom:-120px; left:-110px; animation: drift2 32s var(--ease) infinite; }}
.blob-3 {{ width:300px; height:300px; background:rgba(253,244,208,0.60);
           top:44%; left:38%; animation: drift3 38s var(--ease) infinite; }}
@keyframes drift1 {{ 0%,100%{{transform:translate(0,0) scale(1)}}
                     50%{{transform:translate(-46px,40px) scale(1.09)}} }}
@keyframes drift2 {{ 0%,100%{{transform:translate(0,0) scale(1)}}
                     50%{{transform:translate(52px,-34px) scale(1.12)}} }}
@keyframes drift3 {{ 0%,100%{{transform:translate(0,0)}}
                     50%{{transform:translate(32px,30px)}} }}
[data-testid="stAppViewContainer"] > .main {{ position: relative; z-index: 1; }}

/* ── MOTION ──────────────────────────────────────────────────────────────── */
@keyframes fadeUp   {{ from {{opacity:0; transform:translateY(16px)}} to {{opacity:1; transform:none}} }}
@keyframes fadeDown {{ from {{opacity:0; transform:translateY(-14px)}} to {{opacity:1; transform:none}} }}
@keyframes bobble   {{ 0%,100%{{transform:translateY(0) rotate(-5deg)}}
                       50%{{transform:translateY(-7px) rotate(5deg)}} }}
@keyframes livePulse {{ 0%{{box-shadow:0 0 0 0 rgba(63,155,79,.55)}}
                        70%{{box-shadow:0 0 0 9px rgba(63,155,79,0)}}
                        100%{{box-shadow:0 0 0 0 rgba(63,155,79,0)}} }}
@keyframes downPulse {{ 0%{{box-shadow:0 0 0 0 rgba(232,80,58,.55)}}
                        70%{{box-shadow:0 0 0 9px rgba(232,80,58,0)}}
                        100%{{box-shadow:0 0 0 0 rgba(232,80,58,0)}} }}
@keyframes shimmer  {{ 0%{{background-position:-460px 0}} 100%{{background-position:460px 0}} }}
@keyframes growBar  {{ from {{transform:scaleX(0)}} to {{transform:scaleX(1)}} }}

/* Respect people who ask the OS to stop moving things. */
@media (prefers-reduced-motion: reduce) {{
  *, *::before, *::after {{
    animation-duration: .001ms !important; animation-iteration-count: 1 !important;
    transition-duration: .001ms !important;
  }}
}}

/* ── HEADER ──────────────────────────────────────────────────────────────── */
.portal-head {{
  display:flex; align-items:center; gap:16px; flex-wrap:wrap;
  border-bottom:1px solid var(--border); padding-bottom:16px; margin-bottom:4px;
  animation: fadeDown .7s var(--ease) both;
}}
.head-tomato {{ font-size:2.6rem; animation: bobble 3.6s ease-in-out infinite; line-height:1; }}
.portal-title {{
  font-family:'Playfair Display',serif; font-size:2.15rem; font-weight:700;
  color:var(--tomato); margin:0; line-height:1.08;
}}
.portal-title em {{ color:var(--leaf); font-style:italic; }}
.portal-sub {{ color:var(--text-light); font-size:1rem; }}
.portal-badge {{
  margin-left:auto; font-size:.78rem; letter-spacing:.09em; text-transform:uppercase;
  font-weight:900; background:var(--tomato-light); color:var(--tomato);
  padding:7px 14px; border-radius:999px; white-space:nowrap;
}}

/* ── STATUS PILLS ────────────────────────────────────────────────────────── */
.pill-row {{ display:flex; flex-wrap:wrap; gap:11px; margin:14px 0 22px; }}
.pill {{
  display:inline-flex; align-items:center; gap:9px; background:#fff;
  border:1px solid var(--border); border-radius:999px; padding:9px 17px;
  font-size:.95rem; box-shadow:var(--shadow);
  animation: fadeUp .55s var(--ease) both; transition: transform .25s var(--ease);
}}
.pill:hover {{ transform: translateY(-2px); }}
.pill:nth-child(1){{animation-delay:.04s}} .pill:nth-child(2){{animation-delay:.09s}}
.pill:nth-child(3){{animation-delay:.14s}} .pill:nth-child(4){{animation-delay:.19s}}
.pill:nth-child(5){{animation-delay:.24s}} .pill:nth-child(6){{animation-delay:.29s}}
.pill b {{ font-weight:900; }}
.dot {{ width:10px; height:10px; border-radius:50%; flex:0 0 auto; }}
.dot.up   {{ background:#3f9b4f; animation: livePulse 2.4s ease-out infinite; }}
.dot.down {{ background:var(--tomato); animation: downPulse 1.7s ease-out infinite; }}
.dot.warn {{ background:#d9a219; }}

/* ── CARDS ───────────────────────────────────────────────────────────────── */
.card {{
  background:#fff; border:1px solid var(--border); border-radius:16px;
  padding:20px 22px; box-shadow:var(--shadow); margin-bottom:14px;
  animation: fadeUp .6s var(--ease) both;
  transition: transform .28s var(--ease), box-shadow .28s var(--ease);
}}
.card:hover {{ transform: translateY(-3px); box-shadow: var(--shadow-lift); }}
.card h4 {{ margin:0 0 10px; font-size:1.1rem !important; }}
.metric {{ font-size:2.3rem; font-weight:900; color:var(--text); line-height:1.05;
           font-variant-numeric: tabular-nums; }}
.metric small {{ font-size:.86rem; color:var(--text-light); font-weight:400; }}
.metric.red {{ color:var(--tomato); }}
.muted {{ color:var(--text-light); font-size:.96rem; line-height:1.7; }}

/* A thin bar under a metric, so a ratio reads at a glance. */
.bar {{ height:6px; border-radius:999px; background:var(--light-green);
        margin-top:12px; overflow:hidden; }}
.bar > span {{ display:block; height:100%; border-radius:999px; background:var(--leaf);
               transform-origin:left; animation: growBar .9s var(--ease) both; }}
.bar > span.red {{ background:var(--tomato); }}

.warnbox, .okbox, .infobox {{
  padding:13px 16px; border-radius:10px; font-size:.97rem; margin:10px 0;
  animation: fadeUp .5s var(--ease) both; line-height:1.65;
}}
.warnbox {{ background:var(--tomato-light); border-left:4px solid var(--tomato); }}
.okbox   {{ background:var(--light-green);  border-left:4px solid var(--leaf); }}
.infobox {{ background:var(--petal);        border-left:4px solid #d9a219; }}

/* Loading skeleton, for the moment before a probe answers. */
.skel {{
  height:14px; border-radius:7px; margin:9px 0;
  background:linear-gradient(90deg, #f0efe8 25%, #f8f7f1 50%, #f0efe8 75%);
  background-size:920px 100%; animation: shimmer 1.3s linear infinite;
}}

/* ── STREAMLIT CHROME ────────────────────────────────────────────────────── */
div[data-baseweb="tab-list"] {{ gap:6px; border-bottom:1px solid var(--border); }}
button[data-baseweb="tab"] {{
  font-weight:900 !important; font-size:1.02rem !important;
  border-radius:11px 11px 0 0 !important; transition: background .22s var(--ease);
}}
button[data-baseweb="tab"]:hover {{ background:var(--light-green) !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color:var(--tomato) !important; }}
[data-baseweb="tab-highlight"] {{ background:var(--tomato) !important; height:3px !important; }}
[data-testid="stTabsPanel"] {{ animation: fadeUp .45s var(--ease) both; }}

.stButton > button {{
  background:var(--leaf); color:#fff; border:none; border-radius:11px;
  font-weight:900; font-size:1rem; padding:.55rem 1.3rem;
  transition: transform .2s var(--ease), background .2s var(--ease);
}}
.stButton > button:hover {{ background:var(--soft-green); color:#fff; transform:translateY(-2px); }}

[data-testid="stChatMessage"] {{
  background:#fff; border:1px solid var(--border); border-radius:14px;
  box-shadow:var(--shadow); animation: fadeUp .45s var(--ease) both; padding:4px 6px;
}}
[data-testid="stExpander"] details {{
  border:1px solid var(--border) !important; border-radius:12px !important; background:#fff;
}}
code {{ font-size:.92rem !important; background:var(--light-green) !important;
        padding:2px 6px !important; border-radius:5px !important; }}
pre code {{ background:transparent !important; }}
</style>
"""
