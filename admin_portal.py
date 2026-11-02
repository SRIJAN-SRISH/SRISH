"""SRISH admin portal — internal tool for the team.

One place to see whether the whole system is actually working: the vision
models, the chat brain, the agent routing, and the farm sensor data.

    streamlit run admin_portal.py

Every interactive section is wrapped in @st.fragment. Streamlit's default is to
re-execute the entire script on any widget change, which made asking a question
look like the page was reloading. A fragment reruns only itself, so sending a
chat message no longer rebuilds the charts, re-probes Ollama or resets the tab.

Set SRISH_ADMIN_PASSWORD to require a password. Without it the portal is open
to anyone who can reach the port, which is fine on localhost and not fine
anywhere else.
"""

import os
import time
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import farm_data
import srish_core
from srish_theme import CHART_COLORS, GRAPH_BORDER, PALETTE, background, portal_css

st.set_page_config(
    page_title="SRISH — Admin Portal",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(portal_css(), unsafe_allow_html=True)
st.markdown(background(), unsafe_allow_html=True)

ADMIN_PASSWORD = os.environ.get("SRISH_ADMIN_PASSWORD", "")
P = PALETTE


# ── ACCESS ────────────────────────────────────────────────────────────────────

def gate() -> bool:
    """Password prompt, only when SRISH_ADMIN_PASSWORD is set."""
    if not ADMIN_PASSWORD or st.session_state.get("admin_ok"):
        return True

    st.markdown(
        "<div class='portal-head'><span class='head-tomato'>🍅</span><div>"
        "<p class='portal-title'>SRISH <em>admin</em></p>"
        "<span class='portal-sub'>Internal access only</span></div></div>",
        unsafe_allow_html=True,
    )
    with st.form("gate"):
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Enter"):
            if pw == ADMIN_PASSWORD:
                st.session_state["admin_ok"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
    return False


if not gate():
    st.stop()


# ── PROBES ────────────────────────────────────────────────────────────────────
# Cached, so a fragment rerun reads memory instead of hitting the network.

@st.cache_data(ttl=30, show_spinner=False)
def probe_ollama() -> dict:
    """Which Ollama models exist, and are the two Srish needs among them."""
    try:
        import ollama

        models = [m.get("model") or m.get("name") for m in ollama.list().get("models", [])]
        base = {(m or "").split(":")[0] for m in models}
        return {
            "up": True,
            "models": sorted(m for m in models if m),
            "router_ok": srish_core.ROUTER_MODEL.split(":")[0] in base,
            "chat_ok": srish_core.CHAT_MODEL.split(":")[0] in base,
        }
    except Exception as e:
        return {"up": False, "error": str(e), "models": [],
                "router_ok": False, "chat_ok": False}


@st.cache_data(ttl=20, show_spinner=False)
def probe_loaded() -> list:
    """Models currently resident, and how much of each actually sits on the GPU.

    Ollama reports size_vram alongside size. When size_vram is smaller, the
    remainder is running on CPU RAM, which is far slower per token. On a 6GB
    card a 7B model does not fit whole, so this is the single most useful
    number for explaining why an answer felt slow.
    """
    try:
        import ollama

        out = []
        for m in (ollama.ps() or {}).get("models", []):
            total = m.get("size") or 0
            vram = m.get("size_vram") or 0
            out.append({
                "model": m.get("model") or m.get("name"),
                "size_gb": round(total / 1e9, 2),
                "on_gpu_gb": round(vram / 1e9, 2),
                "gpu_pct": round(100 * vram / total) if total else 0,
                "context": m.get("context_length"),
                "until": str(m.get("expires_at", ""))[:19].replace("T", " "),
            })
        return out
    except Exception:
        return []


@st.cache_data(ttl=20, show_spinner=False)
def probe_gpu() -> dict:
    """GPU name and memory, via nvidia-smi. Empty dict if there is no NVIDIA GPU."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return {}
        name, total, used, util, temp = [x.strip() for x in out.stdout.strip().split(",")]
        return {"name": name, "total_mb": int(total), "used_mb": int(used),
                "util_pct": int(util), "temp_c": int(temp)}
    except Exception:
        return {}


@st.cache_data(ttl=60, show_spinner=False)
def probe_vision() -> dict:
    try:
        import srish_vision

        return {"ok": True, **srish_vision.models_status()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@st.cache_data(ttl=60, show_spinner=False)
def probe_sensors(farm_id: str) -> dict:
    rows = farm_data._rows(farm_id, days=None, limit=1000)
    return {"up": rows is not None, "rows": rows or []}


@st.cache_data(ttl=60, show_spinner=False)
def probe_farms() -> list:
    return farm_data._sb("farm_profiles", {"select": "*", "order": "farm_id"}) or []


@st.cache_data(ttl=30, show_spinner=False)
def probe_n8n() -> bool:
    try:
        import requests

        requests.post(farm_data.N8N_FARM_URL,
                      json={"farm_id": "PING", "mode": "status", "days": 1}, timeout=4)
        return True
    except Exception:
        return False


def dot(state: str) -> str:
    return f"<span class='dot {state}'></span>"


def card(title: str, value, sub: str = "", red: bool = False, ratio: float | None = None) -> str:
    bar = ""
    if ratio is not None:
        pct = max(0, min(100, round(ratio * 100)))
        bar = f"<div class='bar'><span class='{'red' if red else ''}' " \
              f"style='width:{pct}%'></span></div>"
    return (f"<div class='card'><h4>{title}</h4>"
            f"<div class='metric{' red' if red else ''}'>{value}"
            f"{f'<small> {sub}</small>' if sub else ''}</div>{bar}</div>")


def data_age(rows: list) -> int | None:
    if not rows:
        return None
    newest = max(r["uploaded_at"] for r in rows)
    return (datetime.now(timezone.utc)
            - datetime.fromisoformat(newest.replace("Z", "+00:00"))).days


# ── HEADER ────────────────────────────────────────────────────────────────────

oll = probe_ollama()
vis = probe_vision()
farms = probe_farms()
farm_ids = [f["farm_id"] for f in farms] or [srish_core.DEFAULT_FARM_ID]
sb_up = bool(farms)

st.markdown(
    "<div class='portal-head'><span class='head-tomato'>🍅</span><div>"
    "<p class='portal-title'>SRISH <em>admin portal</em></p>"
    "<span class='portal-sub'>Vision · chat · agent routing · farm telemetry</span></div>"
    "<span class='portal-badge'>internal</span></div>",
    unsafe_allow_html=True,
)


@st.fragment
def status_strip():
    """Live status pills, refreshable on their own."""
    o, v, loaded = probe_ollama(), probe_vision(), probe_loaded()
    gpu = probe_gpu()
    gk = v.get("ok") and v.get("gatekeeper", {}).get("present")
    warm = {(m["model"] or "").split(":")[0] for m in loaded}

    # Worst split across resident models: anything under 100% is partly on CPU.
    min_gpu_pct = min((m["gpu_pct"] for m in loaded), default=None)
    if min_gpu_pct is None:
        compute = f"{dot('warn')} compute <b>nothing loaded</b>"
    elif min_gpu_pct >= 99:
        compute = f"{dot('up')} compute <b>100% GPU</b>"
    else:
        compute = (f"{dot('warn')} compute <b>{min_gpu_pct}% GPU / "
                   f"{100 - min_gpu_pct}% CPU</b>")

    pills = [
        f"{dot('up' if o['up'] else 'down')} Ollama <b>{'up' if o['up'] else 'down'}</b>",
        f"{dot('up' if o['router_ok'] else 'down')} router <b>{srish_core.ROUTER_MODEL}</b>",
        f"{dot('up' if o['chat_ok'] else 'down')} chat <b>{srish_core.CHAT_MODEL}</b>",
        f"{dot('up' if warm else 'warn')} in memory "
        f"<b>{', '.join(sorted(warm)) if warm else 'nothing warm'}</b>",
        compute,
        f"{dot('up' if sb_up else 'down')} Supabase <b>{'reachable' if sb_up else 'down'}</b>",
        f"{dot('up' if gk else 'down')} vision <b>{'ready' if gk else 'missing'}</b>",
    ]
    if gpu:
        pills.append(
            f"{dot('up')} {gpu['name'].replace('NVIDIA GeForce ', '')} "
            f"<b>{gpu['used_mb']}/{gpu['total_mb']} MB · {gpu['temp_c']}°C</b>")
    c1, c2 = st.columns([11, 1])
    c1.markdown(
        "<div class='pill-row'>"
        + "".join(f"<span class='pill'>{p}</span>" for p in pills)
        + "</div>", unsafe_allow_html=True)
    with c2:
        st.write("")
        if st.button("↻", help="Re-probe everything", key="restatus"):
            probe_ollama.clear()
            probe_loaded.clear()
            probe_vision.clear()
            probe_farms.clear()
            probe_sensors.clear()
            st.rerun(scope="fragment")

    if not warm and o["up"]:
        st.markdown(
            "<div class='infobox'>No model is resident in memory, so the next question "
            "pays a multi-second load before it can start answering. "
            "<code>scripts/start_srish.ps1</code> warms them at login.</div>",
            unsafe_allow_html=True)


status_strip()

tab_over, tab_farm, tab_graph, tab_vision, tab_chat = st.tabs(
    ["Overview", "Farm data", "Agent & nodes", "Vision", "Chat"]
)


# ── OVERVIEW ──────────────────────────────────────────────────────────────────

@st.fragment
def overview():
    rows = probe_sensors(farm_ids[0])["rows"]
    age = data_age(rows)
    kb = len(srish_core.load_knowledge_base())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("Knowledge base", kb, "diseases"), unsafe_allow_html=True)
    c2.markdown(card("Farms", len(farms), "profiles"), unsafe_allow_html=True)
    c3.markdown(card("Sensor rows", len(rows), f"for {farm_ids[0]}"), unsafe_allow_html=True)
    c4.markdown(
        card("Data age", age if age is not None else "—", "days old",
             red=bool(age and age > 7), ratio=min(1, (age or 0) / 60)),
        unsafe_allow_html=True)

    st.markdown("### What is and is not working")

    checks = [
        ("Ollama reachable", oll["up"],
         oll.get("error", "") or f"{len(oll['models'])} models installed"),
        (f"Router model `{srish_core.ROUTER_MODEL}`", oll["router_ok"],
         "present" if oll["router_ok"] else "not installed — routing will fail"),
        (f"Chat model `{srish_core.CHAT_MODEL}`", oll["chat_ok"],
         "present" if oll["chat_ok"] else "not installed — answers will fail"),
        ("Supabase readable", sb_up,
         f"{len(farms)} farm profiles" if sb_up else "could not read farm_profiles"),
        ("Sensor rows present", bool(rows),
         f"{len(rows)} rows" if rows else "no rows for this farm"),
    ]
    if farm_data.FARM_BACKEND == "n8n":
        up = probe_n8n()
        checks.append(("n8n webhook", up,
                       "answering" if up else f"unreachable at {farm_data.N8N_FARM_URL}"))
    if vis.get("ok"):
        for name in ("gatekeeper", "detector"):
            i = vis.get(name, {})
            checks.append((f"Vision — {name}", i.get("present", False),
                           f"{i.get('size_mb')} MB" if i.get("present")
                           else f"missing at {i.get('path')}"))
    else:
        checks.append(("Vision models", False, vis.get("error", "could not import")))

    passed = sum(1 for _, ok, _ in checks if ok)
    st.markdown(
        f"<div class='{'okbox' if passed == len(checks) else 'warnbox'}'>"
        f"<b>{passed} of {len(checks)} checks passing.</b></div>", unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame([{"check": c, "ok": "✅" if ok else "❌", "detail": d}
                      for c, ok, d in checks]),
        hide_index=True, use_container_width=True)

    st.markdown("### Where the model is actually running")
    loaded, gpu = probe_loaded(), probe_gpu()

    if gpu:
        g1, g2, g3 = st.columns(3)
        g1.markdown(card("GPU", gpu["name"].replace("NVIDIA GeForce ", ""), ""),
                    unsafe_allow_html=True)
        g2.markdown(
            card("VRAM", f"{gpu['used_mb']}", f"of {gpu['total_mb']} MB",
                 red=gpu["used_mb"] / gpu["total_mb"] > 0.9,
                 ratio=gpu["used_mb"] / gpu["total_mb"]),
            unsafe_allow_html=True)
        g3.markdown(card("Utilisation", f"{gpu['util_pct']}%", f"{gpu['temp_c']}°C",
                         ratio=gpu["util_pct"] / 100), unsafe_allow_html=True)
    else:
        st.markdown("<div class='muted'>No NVIDIA GPU detected by nvidia-smi — if that is "
                    "wrong, the model is running entirely on CPU.</div>",
                    unsafe_allow_html=True)

    if not loaded:
        st.markdown("<div class='infobox'>Nothing is loaded right now, so there is no "
                    "split to report. Ask a question, or run "
                    "<code>scripts/start_srish.ps1</code>.</div>", unsafe_allow_html=True)
    else:
        st.dataframe(pd.DataFrame(loaded), hide_index=True, use_container_width=True)
        for m in loaded:
            if m["gpu_pct"] >= 99:
                st.markdown(
                    f"<div class='okbox'><b>{m['model']} is fully on the GPU</b> "
                    f"({m['on_gpu_gb']} GB). This is the fast path.</div>",
                    unsafe_allow_html=True)
            else:
                spill = round(m["size_gb"] - m["on_gpu_gb"], 2)
                st.markdown(
                    f"<div class='warnbox'><b>{m['model']} is split "
                    f"{m['gpu_pct']}% GPU / {100 - m['gpu_pct']}% CPU.</b> "
                    f"{m['on_gpu_gb']} GB of {m['size_gb']} GB is in VRAM; the remaining "
                    f"{spill} GB runs on CPU RAM, which is far slower per token and is the "
                    "main reason an answer feels slow.<br>"
                    "Options, cheapest first: lower <code>num_ctx</code> in the Modelfile "
                    f"(currently {m.get('context')}), or run a model that fits whole — a "
                    "4B at Q4 leaves headroom on a 6 GB card where a 7B does not.</div>",
                    unsafe_allow_html=True)

    if age and age > 7:
        st.markdown(
            f"<div class='warnbox'><b>Telemetry is {age} days old.</b> A report asked for "
            "a recent window will correctly return nothing. The charts therefore show the "
            "full history that does exist, not a recent window.</div>",
            unsafe_allow_html=True)

    st.markdown(
        "<div class='muted'><b>Config in effect</b> &nbsp;·&nbsp; "
        f"orchestrator <code>{srish_core.ORCHESTRATOR}</code> &nbsp;·&nbsp; "
        f"keep-alive <code>{srish_core.KEEP_ALIVE}</code> &nbsp;·&nbsp; "
        f"farm backend <code>{farm_data.FARM_BACKEND}</code> &nbsp;·&nbsp; "
        f"default farm <code>{srish_core.DEFAULT_FARM_ID}</code></div>",
        unsafe_allow_html=True)


with tab_over:
    overview()


# ── FARM DATA ─────────────────────────────────────────────────────────────────

@st.fragment
def farm_panel():
    """Charts and quality audit. Changing farm reruns only this."""
    left, right = st.columns([1, 3])
    farm_id = left.selectbox("Farm", farm_ids, key="farm_pick")
    sensors = probe_sensors(farm_id)

    if not sensors["up"]:
        st.markdown("<div class='warnbox'>Could not reach Supabase.</div>",
                    unsafe_allow_html=True)
        return
    if not sensors["rows"]:
        st.markdown(f"<div class='warnbox'>No sensor rows for {farm_id}.</div>",
                    unsafe_allow_html=True)
        return

    df = pd.DataFrame(sensors["rows"])
    # uploaded_at, never recorded_at: timestamp_valid is 0 on every row that
    # shipped without an RTC sync, and those all carry 1970-01-01.
    df["ts"] = pd.to_datetime(df["uploaded_at"], errors="coerce", utc=True)
    df = df.sort_values("ts")
    valid_rtc = int(df["timestamp_valid"].fillna(0).astype(int).sum())

    right.markdown(
        f"<div class='muted' style='padding-top:30px'>{len(df)} rows &nbsp;·&nbsp; "
        f"{df['ts'].min():%d %b %Y} → {df['ts'].max():%d %b %Y} &nbsp;·&nbsp; "
        f"nodes {', '.join(sorted(df['node_id'].dropna().unique()))} &nbsp;·&nbsp; "
        f"valid RTC on {valid_rtc}/{len(df)} rows</div>", unsafe_allow_html=True)

    if valid_rtc == 0:
        st.markdown(
            "<div class='warnbox'><b>No row has a valid RTC timestamp.</b> Every "
            "<code>recorded_at</code> reads 1970-01-01, so charts are plotted against "
            "<code>uploaded_at</code> instead.</div>", unsafe_allow_html=True)

    prof = next((f for f in farms if f["farm_id"] == farm_id), {})

    st.markdown("### Soil")
    soil = [k for k in farm_data.SOIL_KEYS if df[k].notna().any()]
    if soil:
        pick = st.multiselect(
            "Fields", soil, key="soil_pick",
            default=[k for k in ("vwc_percent", "soil_temp", "ph") if k in soil] or soil[:3])
        if pick:
            st.line_chart(df.set_index("ts")[pick], color=CHART_COLORS[:len(pick)],
                          height=320)

        if "vwc_percent" in df and df["vwc_percent"].notna().any():
            fc, wp, mad = (prof.get("field_capacity_vwc"), prof.get("wilting_point_vwc"),
                           prof.get("mad_percent"))
            latest = df["vwc_percent"].dropna().iloc[-1]
            if None not in (fc, wp, mad):
                trigger = fc - (mad / 100) * (fc - wp)
                if latest < wp:
                    st.markdown(
                        f"<div class='warnbox'><b>Soil moisture {latest:.1f}% is below the "
                        f"wilting point ({wp}%).</b> Irrigation trigger for this farm is "
                        f"{trigger:.1f}%. On these numbers the crop is water-stressed.</div>",
                        unsafe_allow_html=True)
                elif latest < trigger:
                    st.markdown(
                        f"<div class='warnbox'>Soil moisture {latest:.1f}% is under the "
                        f"irrigation trigger ({trigger:.1f}%).</div>", unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<div class='okbox'>Soil moisture {latest:.1f}% is above the "
                        f"{trigger:.1f}% trigger.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='muted'>No soil readings in this data.</div>",
                    unsafe_allow_html=True)

    st.markdown("### Environment")
    env = [k for k in farm_data.ENV_KEYS if df[k].notna().any()]
    if env:
        st.line_chart(df.set_index("ts")[env], color=CHART_COLORS[:len(env)], height=300)
    else:
        st.markdown("<div class='muted'>No environment readings.</div>",
                    unsafe_allow_html=True)

    st.markdown("### Reading quality")
    q = []
    for k in farm_data.ALL_KEYS:
        if k not in df:
            continue
        vals = df[k].dropna()
        if vals.empty:
            q.append({"field": k, "rows": 0, "note": "never reported"})
            continue
        note = ""
        lo, hi = farm_data.PLAUSIBLE.get(k, (None, None))
        if lo is not None and not (lo <= vals.iloc[-1] <= hi):
            note = f"latest {vals.iloc[-1]:g} outside plausible [{lo},{hi}]"
        elif len(vals) >= 5 and vals.min() == vals.max():
            note = f"stuck at {vals.iloc[-1]:g} across {len(vals)} rows"
        q.append({"field": k, "rows": len(vals),
                  "min": round(float(vals.min()), 2), "max": round(float(vals.max()), 2),
                  "latest": round(float(vals.iloc[-1]), 2), "note": note})
    flagged = [r for r in q if r.get("note")]
    if flagged:
        st.markdown(
            f"<div class='warnbox'><b>{len(flagged)} field(s) look wrong</b>, not just "
            "sparse: " + ", ".join(r["field"] for r in flagged) + ".</div>",
            unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(q), hide_index=True, use_container_width=True)

    with st.expander("What Srish is told — the exact text handed to the LLM"):
        st.markdown("<div class='muted'>If a number is not in here, Srish cannot report "
                    "it. This is what stops invented readings.</div>",
                    unsafe_allow_html=True)
        st.code(farm_data.facts_analysis(farm_id, days=3650), language="text")

    with st.expander("Raw rows"):
        st.dataframe(df.drop(columns=["ts"]), hide_index=True, use_container_width=True)


with tab_farm:
    farm_panel()


# ── AGENT & NODES ─────────────────────────────────────────────────────────────

with tab_graph:
    st.markdown("### Agent routing")
    st.markdown(
        "<div class='muted'>Every message takes exactly one path through this graph. "
        "The router picks which, and that choice is what <code>meta.intent</code> "
        "reports.</div>", unsafe_allow_html=True)

    agent_dot = f"""
digraph agent {{
  rankdir=LR;
  bgcolor="transparent";
  node [shape=box style="rounded,filled" fontname="Lato" fontsize=11
        color="{GRAPH_BORDER}" penwidth=1.2];
  edge [fontname="Lato" fontsize=9 color="{P['text_light']}"];

  msg   [label="user message" fillcolor="{P['petal_yellow']}"];
  router[label="router\\n{srish_core.ROUTER_MODEL}" fillcolor="{P['tomato_light']}"
         fontcolor="{P['tomato']}"];
  kw    [label="keyword override\\n(farm_* rescue)" fillcolor="#ffffff"
         style="rounded,dashed,filled"];

  greet [label="greeting" fillcolor="{P['light_green']}"];
  off   [label="off_topic\\nrefusal, no LLM" fillcolor="{P['tomato_light']}"];
  dis   [label="tomato_disease" fillcolor="{P['light_green']}"];
  gen   [label="tomato_general" fillcolor="{P['light_green']}"];
  url   [label="url_query" fillcolor="{P['light_green']}"];
  fs    [label="farm_status" fillcolor="{P['light_green']}"];
  fr    [label="farm_report" fillcolor="{P['light_green']}"];

  kb    [label="knowledge base\\n{len(srish_core.load_knowledge_base())} diseases"
         shape=cylinder fillcolor="#ffffff"];
  web   [label="fetch + extract page" fillcolor="#ffffff"];
  sb    [label="farm facts\\n({farm_data.FARM_BACKEND})" shape=cylinder fillcolor="#ffffff"];

  prompt[label="build system prompt" fillcolor="{P['petal_yellow']}"];
  chat  [label="answer\\n{srish_core.CHAT_MODEL}" fillcolor="{P['tomato_light']}"
         fontcolor="{P['tomato']}"];
  out   [label="reply" fillcolor="{P['light_green']}"];

  msg -> router;
  router -> kw [label="plan"];
  kw -> greet; kw -> off; kw -> dis; kw -> gen; kw -> url; kw -> fs; kw -> fr;

  dis -> kb [label="if match"];
  url -> web;
  fs -> sb; fr -> sb;

  greet -> prompt; dis -> prompt; gen -> prompt;
  kb -> prompt; web -> prompt; sb -> prompt;
  prompt -> chat -> out;
  off -> out [label="direct, bypasses the model"];

  sb -> out [label="NO_DATA: hard exit,\\nmodel never sees it" style=dashed
             color="{P['tomato']}" fontcolor="{P['tomato']}"];
}}
"""
    st.graphviz_chart(agent_dot, use_container_width=True)
    st.markdown(
        "<div class='muted'>Two edges carry the trust guarantees. <b>off_topic</b> skips "
        "the model entirely, so a refusal can never drift into an answer. The dashed "
        "<b>NO_DATA</b> edge exits before generation, which is what stops Srish inventing "
        "sensor values when there are none.</div>", unsafe_allow_html=True)

    st.markdown("### Node connectivity")
    sensors_all = {fid: probe_sensors(fid) for fid in farm_ids}

    node_lines = []
    for f in farms:
        fid = f["farm_id"]
        rows = sensors_all.get(fid, {}).get("rows", [])
        seen = sorted({r.get("node_id") for r in rows if r.get("node_id")})
        declared = f.get("node_id")

        node_lines.append(
            f'  "{fid}" [label="{fid}\\n{f.get("field_id","")} · '
            f'{f.get("field_area_m2","?")} m²" fillcolor="{P["light_green"]}"];')
        node_lines.append(f'  farm -> "{fid}";')

        if declared and declared not in seen:
            # Profile names a node that has never uploaded anything.
            node_lines.append(
                f'  "{declared}" [label="{declared}\\nno data" '
                f'fillcolor="{P["tomato_light"]}" fontcolor="{P["tomato"]}"];')
            node_lines.append(
                f'  "{fid}" -> "{declared}" [style=dashed color="{P["tomato"]}" '
                f'label="silent"];')
        for n in seen:
            node_lines.append(
                f'  "{n}" [label="{n}\\n{sum(1 for r in rows if r.get("node_id") == n)} rows" '
                f'fillcolor="#ffffff"];')
            node_lines.append(f'  "{fid}" -> "{n}";')

    nodes_dot = f"""
digraph nodes {{
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fontname="Lato" fontsize=11
        color="{GRAPH_BORDER}" penwidth=1.2];
  edge [fontname="Lato" fontsize=9 color="{P['text_light']}"];
  farm [label="Supabase\\nsensor_records" shape=cylinder fillcolor="{P['petal_yellow']}"];
{chr(10).join(node_lines)}
}}
"""
    st.graphviz_chart(nodes_dot, use_container_width=True)

    silent = [f for f in farms if f.get("node_id") and f["node_id"] not in
              {r.get("node_id") for r in sensors_all.get(f["farm_id"], {}).get("rows", [])}]
    if silent:
        st.markdown(
            f"<div class='warnbox'><b>{len(silent)} of {len(farms)} farms name a node in "
            "their profile that has never uploaded a row.</b> Shown dashed above: "
            + ", ".join(f"{f['farm_id']} → {f['node_id']}" for f in silent) + ".</div>",
            unsafe_allow_html=True)

    with st.expander("Farm profiles"):
        st.dataframe(pd.DataFrame(farms), hide_index=True, use_container_width=True)


# ── VISION ────────────────────────────────────────────────────────────────────

@st.fragment
def vision_panel():
    """Upload and detect. Uploading reruns only this fragment."""
    st.markdown("### Leaf verification and disease detection")
    st.markdown(
        "<div class='muted'>The gatekeeper classifier runs first. The detector only sees "
        "images it accepts, so a photo of something else is rejected rather than scanned "
        "for lesions.</div>", unsafe_allow_html=True)

    if not vis.get("ok"):
        st.markdown(f"<div class='warnbox'>Vision unavailable: {vis.get('error')}</div>",
                    unsafe_allow_html=True)
        return
    if not vis.get("gatekeeper", {}).get("present"):
        st.markdown(
            "<div class='warnbox'><b>Gatekeeper weights missing.</b><br>Looked in "
            f"<code>{vis['gatekeeper']['path']}</code>. Weights are gitignored, so put "
            "<code>keeper.pt</code> and <code>yolo.pt</code> next to the scripts, or set "
            "<code>SRISH_GATEKEEPER_PATH</code> / <code>SRISH_DETECTOR_PATH</code>.</div>",
            unsafe_allow_html=True)
        return

    import srish_vision

    src = st.radio("Source", ["Upload", "Camera"], horizontal=True,
                   label_visibility="collapsed", key="vis_src")
    img_bytes = None
    if src == "Upload":
        up = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "webp"],
                              label_visibility="collapsed", key="vis_up")
        if up:
            img_bytes = up.getvalue()
    else:
        shot = st.camera_input("Capture", label_visibility="collapsed", key="vis_cam")
        if shot:
            img_bytes = shot.getvalue()

    if not img_bytes:
        return

    pil = srish_vision.bytes_to_pil(img_bytes)
    frame = srish_vision.pil_to_cv2(pil)
    c1, c2 = st.columns(2)
    c1.image(pil, caption="Input", use_container_width=True)

    with c2:
        with st.status("Checking it is a tomato leaf...", expanded=False) as s:
            t0 = time.perf_counter()
            is_leaf, conf, cls = srish_vision.check_if_leaf(frame)
            gk_ms = (time.perf_counter() - t0) * 1000
            s.update(label=f"Gatekeeper: {cls} at {conf:.0%} ({gk_ms:.0f} ms)",
                     state="complete")

        if not is_leaf:
            st.markdown(
                f"<div class='warnbox'><b>Rejected by the gatekeeper.</b><br>Classified as "
                f"<code>{cls}</code> at {conf:.0%} confidence; needs "
                f"<code>{srish_vision.LEAF_CLASS}</code> at "
                f"{srish_vision.LEAF_CONFIDENCE:.0%} or more.<br>The detector was not run."
                "</div>", unsafe_allow_html=True)
            return

        st.markdown(f"<div class='okbox'>Accepted as a tomato leaf ({conf:.0%} "
                    f"confidence, {gk_ms:.0f} ms).</div>", unsafe_allow_html=True)

        if not vis.get("detector", {}).get("present"):
            st.markdown("<div class='warnbox'>Detector weights missing, so no disease "
                        "scan.</div>", unsafe_allow_html=True)
            return

        with st.status("Looking for Leaf Miner damage...", expanded=False) as s:
            t0 = time.perf_counter()
            res = srish_vision.detect_disease(frame)
            det_ms = (time.perf_counter() - t0) * 1000
            s.update(label=f"Detector: {res['count']} found ({det_ms:.0f} ms)",
                     state="complete")

        st.image(res["annotated"][:, :, ::-1], caption="Detections",
                 use_container_width=True)
        if res["found"]:
            st.markdown(f"<div class='warnbox'><b>{res['count']} detection(s)</b> in "
                        f"{det_ms:.0f} ms.</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(res["detections"]), hide_index=True,
                         use_container_width=True)
        else:
            st.markdown("<div class='okbox'>No Leaf Miner damage found. Note the detector "
                        "only knows that one class.</div>", unsafe_allow_html=True)


with tab_vision:
    vision_panel()


# ── CHAT ──────────────────────────────────────────────────────────────────────

@st.fragment
def chat_panel():
    """The chat, isolated.

    This fragment is the whole reason the portal uses them: without it, sending
    a message re-ran the entire script, which re-probed Ollama, rebuilt every
    chart and looked exactly like a page reload.
    """
    st.markdown("### Chat, with the routing exposed")
    st.markdown(
        "<div class='muted'>Same brain the mobile app calls. Each answer carries the "
        "intent chosen, whether the knowledge base was used, and the timings — which is "
        "the quickest way to tell a routing bug from a generation one.</div>",
        unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    depth = c1.toggle("Detailed answers", value=False, key="chat_depth")
    chat_farm = c2.selectbox("Farm context", farm_ids, key="chat_farm")
    with c3:
        st.write("")
        if st.button("Clear conversation", key="chat_clear"):
            st.session_state["portal_msgs"] = []
            st.session_state["portal_turns"] = []
            st.rerun(scope="fragment")

    msgs = st.session_state.setdefault("portal_msgs", [])
    turns = st.session_state.setdefault("portal_turns", [])

    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                st.caption(m["meta"])

    prompt = st.chat_input("Ask Srish something...", key="chat_in")
    if prompt:
        msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            meta_slot = st.empty()
            body_slot = st.empty()
            meta_slot.caption("routing…")

            t0 = time.perf_counter()
            t_meta = t_first = None
            acc, info = "", {}
            try:
                for kind, payload in srish_core.answer_stream(
                    message=prompt,
                    history=[{"role": m["role"], "content": m["content"]} for m in msgs[:-1]],
                    farm_id=chat_farm,
                    depth_mode=depth,
                ):
                    if kind == "meta":
                        t_meta = time.perf_counter() - t0
                        meta_slot.caption(
                            f"intent `{payload.get('intent')}` · "
                            f"lang `{payload.get('language')}` · routed in {t_meta:.1f}s"
                            + ("  ·  ⚠️ DEGRADED" if payload.get("degraded") else ""))
                    elif kind == "token":
                        if t_first is None:
                            t_first = time.perf_counter() - t0
                        acc += payload
                        body_slot.markdown(acc + " ▌")
                    else:
                        info = payload
                        body_slot.markdown(payload.get("reply", acc))
            except Exception as e:
                acc = f"Failed: {e}"
                body_slot.markdown(acc)

            total = time.perf_counter() - t0
            reply = info.get("reply", acc)
            body_slot.markdown(reply)

            bits = [f"intent `{info.get('intent')}`", f"lang `{info.get('language')}`"]
            if info.get("used_kb"):
                bits.append(f"KB `{info.get('kb_disease')}`")
            bits.append(f"route {t_meta:.1f}s" if t_meta else "route —")
            bits.append(f"1st word {t_first:.1f}s" if t_first else "1st word —")
            bits.append(f"total {total:.1f}s")
            if info.get("degraded"):
                bits.append("⚠️ **degraded** — router down, off-topic filter and KB skipped")
            if info.get("error"):
                bits.append(f"error `{str(info['error'])[:70]}`")
            caption = " · ".join(bits)
            meta_slot.caption(caption)

            msgs.append({"role": "assistant", "content": reply, "meta": caption})
            turns.append({
                "asked": prompt[:46],
                "intent": info.get("intent"),
                "kb": info.get("kb_disease") or "",
                "route_s": round(t_meta, 1) if t_meta else None,
                "first_s": round(t_first, 1) if t_first else None,
                "total_s": round(total, 1),
                "degraded": bool(info.get("degraded")),
            })

    if turns:
        with st.expander(f"Turn log — {len(turns)} question(s) this session"):
            st.dataframe(pd.DataFrame(turns[::-1]), hide_index=True,
                         use_container_width=True)


with tab_chat:
    chat_panel()
