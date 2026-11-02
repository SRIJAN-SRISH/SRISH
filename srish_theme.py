"""The Srish palette, shared so the chat app and the admin portal match.

agentic_bot.py keeps its own narrow, decorative layout. The portal needs wide
tables and charts, so it takes the same tokens and fonts but its own layout.
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
    "&family=Lato:wght@300;400;700&display=swap');"
)


def portal_css() -> str:
    """CSS for the admin portal: same palette, layout suited to dashboards."""
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
  --border:       {p['border']};
  --card-shadow:  0 4px 40px rgba(90,138,90,0.10), 0 1px 8px rgba(0,0,0,0.05);
}}

html, body, [data-testid="stAppViewContainer"] {{
  background: var(--cream) !important;
  font-family: 'Lato', sans-serif !important;
  color: var(--text);
}}
[data-testid="stAppViewContainer"] > .main > .block-container {{
  max-width: 1280px !important;
  padding: 1.5rem 2rem 5rem !important;
}}
[data-testid="stToolbar"] {{ display: none !important; }}
footer {{ visibility: hidden; }}

h1, h2, h3 {{ font-family: 'Playfair Display', serif !important; color: var(--leaf); }}

/* Header */
.portal-head {{
  display: flex; align-items: baseline; gap: 14px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 14px; margin-bottom: 8px; flex-wrap: wrap;
}}
.portal-title {{
  font-family: 'Playfair Display', serif; font-size: 1.9rem; font-weight: 700;
  color: var(--tomato); margin: 0; line-height: 1.1;
}}
.portal-title em {{ color: var(--leaf); font-style: italic; }}
.portal-sub {{ color: var(--text-light); font-size: 0.9rem; }}
.portal-badge {{
  margin-left: auto; font-size: 0.72rem; letter-spacing: 0.08em;
  text-transform: uppercase; font-weight: 700;
  background: var(--tomato-light); color: var(--tomato);
  padding: 5px 11px; border-radius: 999px; white-space: nowrap;
}}

/* Status pills */
.pill-row {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 6px 0 18px; }}
.pill {{
  display: inline-flex; align-items: center; gap: 7px;
  background: #fff; border: 1px solid var(--border); border-radius: 999px;
  padding: 7px 14px; font-size: 0.84rem; box-shadow: var(--card-shadow);
}}
.pill b {{ font-weight: 700; }}
.dot {{ width: 9px; height: 9px; border-radius: 50%; flex: 0 0 auto; }}
.dot.up   {{ background: #3f9b4f; }}
.dot.down {{ background: {p['tomato']}; }}
.dot.warn {{ background: #d9a219; }}

/* Cards */
.card {{
  background: #fff; border: 1px solid var(--border); border-radius: 14px;
  padding: 18px 20px; box-shadow: var(--card-shadow); margin-bottom: 14px;
}}
.card h4 {{
  margin: 0 0 10px; font-family: 'Playfair Display', serif;
  color: var(--leaf); font-size: 1.05rem;
}}
.metric {{ font-size: 1.7rem; font-weight: 700; color: var(--text); line-height: 1.1; }}
.metric small {{ font-size: 0.8rem; color: var(--text-light); font-weight: 400; }}
.muted {{ color: var(--text-light); font-size: 0.86rem; line-height: 1.6; }}
.warnbox {{
  background: var(--tomato-light); border-left: 3px solid var(--tomato);
  padding: 11px 14px; border-radius: 8px; font-size: 0.86rem; margin: 8px 0;
}}
.okbox {{
  background: var(--light-green); border-left: 3px solid var(--leaf);
  padding: 11px 14px; border-radius: 8px; font-size: 0.86rem; margin: 8px 0;
}}

/* Tabs */
button[data-baseweb="tab"] {{ font-family: 'Lato', sans-serif !important; font-weight: 700 !important; }}
[data-baseweb="tab-highlight"] {{ background: var(--tomato) !important; }}

.stButton > button {{
  background: var(--leaf); color: #fff; border: none; border-radius: 10px;
  font-weight: 700; padding: 0.5rem 1.2rem;
}}
.stButton > button:hover {{ background: var(--soft-green); color: #fff; }}

code, pre {{ font-size: 0.82rem !important; }}
</style>
"""
