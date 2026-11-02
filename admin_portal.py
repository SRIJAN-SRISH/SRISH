"""SRISH admin portal — internal tool for the team.

One place to see whether the whole system is actually working: the vision
models, the chat brain, the agent routing, and the farm sensor data.

    streamlit run admin_portal.py

Set SRISH_ADMIN_PASSWORD to require a password. Without it the portal is open
to anyone who can reach the port, which is fine on localhost and not fine
anywhere else.
"""

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import farm_data
import srish_core
from srish_theme import CHART_COLORS, GRAPH_BORDER, PALETTE, portal_css

st.set_page_config(
    page_title="SRISH — Admin Portal",
    page_icon="🍅",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(portal_css(), unsafe_allow_html=True)

ADMIN_PASSWORD = os.environ.get("SRISH_ADMIN_PASSWORD", "")


# ── ACCESS ────────────────────────────────────────────────────────────────────

def gate() -> bool:
    """Password prompt, only when SRISH_ADMIN_PASSWORD is set."""
    if not ADMIN_PASSWORD:
        return True
    if st.session_state.get("admin_ok"):
        return True

    st.markdown(
        "<div class='portal-head'><div><p class='portal-title'>SRISH <em>admin</em></p>"
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


# ── HEALTH PROBES ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def probe_ollama() -> dict:
    """Which Ollama models exist, and are the two Srish needs among them."""
    try:
        import ollama

        models = [
            m.get("model") or m.get("name")
            for m in ollama.list().get("models", [])
        ]
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


@st.cache_data(ttl=30)
def probe_vision() -> dict:
    try:
        import srish_vision

        return {"ok": True, **srish_vision.models_status()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@st.cache_data(ttl=60)
def probe_sensors(farm_id: str) -> dict:
    """Pull every row for a farm, so each tab works off one fetch."""
    rows = farm_data._rows(farm_id, days=None, limit=1000)
    if rows is None:
        return {"up": False, "rows": []}
    return {"up": True, "rows": rows}


@st.cache_data(ttl=60)
def probe_farms() -> list:
    return farm_data._sb("farm_profiles", {"select": "*", "order": "farm_id"}) or []


@st.cache_data(ttl=30)
def probe_n8n() -> bool:
    """Is the n8n webhook answering at all? Only relevant on that backend."""
    try:
        import requests

        requests.post(farm_data.N8N_FARM_URL, json={"farm_id": "PING",
                      "mode": "status", "days": 1}, timeout=4)
        return True
    except Exception:
        return False


def dot(state: str) -> str:
    return f"<span class='dot {state}'></span>"


# ── HEADER ────────────────────────────────────────────────────────────────────

oll = probe_ollama()
vis = probe_vision()
farms = probe_farms()
farm_ids = [f["farm_id"] for f in farms] or [srish_core.DEFAULT_FARM_ID]

st.markdown(
    "<div class='portal-head'>"
    "<div><p class='portal-title'>SRISH <em>admin portal</em></p>"
    "<span class='portal-sub'>Vision · chat · agent routing · farm telemetry</span></div>"
    "<span class='portal-badge'>internal</span>"
    "</div>",
    unsafe_allow_html=True,
)

sb_up = bool(farms)
pills = [
    f"{dot('up' if oll['up'] else 'down')} Ollama <b>{'up' if oll['up'] else 'down'}</b>",
    f"{dot('up' if oll['router_ok'] else 'down')} router <b>{srish_core.ROUTER_MODEL}</b>",
    f"{dot('up' if oll['chat_ok'] else 'down')} chat <b>{srish_core.CHAT_MODEL}</b>",
    f"{dot('up' if sb_up else 'down')} Supabase <b>{'reachable' if sb_up else 'down'}</b>",
    f"{dot('up' if farm_data.FARM_BACKEND == 'supabase' else 'warn')} "
    f"farm backend <b>{farm_data.FARM_BACKEND}</b>",
    f"{dot('up' if vis.get('ok') and vis.get('gatekeeper', {}).get('present') else 'down')} "
    f"vision weights <b>{'present' if vis.get('ok') and vis.get('gatekeeper', {}).get('present') else 'missing'}</b>",
]
st.markdown(f"<div class='pill-row'>{''.join(f'<span class=pill>{p}</span>' for p in pills)}</div>",
            unsafe_allow_html=True)

tab_over, tab_farm, tab_graph, tab_vision, tab_chat = st.tabs(
    ["Overview", "Farm data", "Agent & nodes", "Vision", "Chat"]
)


# ── OVERVIEW ──────────────────────────────────────────────────────────────────

with tab_over:
    c1, c2, c3, c4 = st.columns(4)

    sensors = probe_sensors(farm_ids[0])
    rows = sensors["rows"]

    with c1:
        st.markdown("<div class='card'><h4>Knowledge base</h4>"
                    f"<div class='metric'>{len(srish_core.load_knowledge_base())}"
                    "<small> diseases</small></div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='card'><h4>Farms</h4><div class='metric'>{len(farms)}"
                    "<small> profiles</small></div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><h4>Sensor rows</h4><div class='metric'>{len(rows)}"
                    f"<small> for {farm_ids[0]}</small></div></div>", unsafe_allow_html=True)
    with c4:
        if rows:
            newest = max(r["uploaded_at"] for r in rows)
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(newest.replace("Z", "+00:00"))).days
            st.markdown(f"<div class='card'><h4>Data age</h4><div class='metric'>{age}"
                        f"<small> days old</small></div></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'><h4>Data age</h4>"
                        "<div class='metric'>—</div></div>", unsafe_allow_html=True)

    st.markdown("#### What is and is not working")

    checks = []
    checks.append(("Ollama reachable", oll["up"],
                   oll.get("error", "") or f"{len(oll['models'])} models installed"))
    checks.append((f"Router model `{srish_core.ROUTER_MODEL}`", oll["router_ok"],
                   "present" if oll["router_ok"] else "not installed — routing will fail"))
    checks.append((f"Chat model `{srish_core.CHAT_MODEL}`", oll["chat_ok"],
                   "present" if oll["chat_ok"] else "not installed — answers will fail"))
    checks.append(("Supabase readable", sb_up,
                   f"{len(farms)} farm profiles" if sb_up else "could not read farm_profiles"))
    checks.append(("Sensor rows present", bool(rows),
                   f"{len(rows)} rows" if rows else "no rows for this farm"))

    if farm_data.FARM_BACKEND == "n8n":
        n8n_up = probe_n8n()
        checks.append(("n8n webhook", n8n_up,
                       "answering" if n8n_up else f"unreachable at {farm_data.N8N_FARM_URL}"))

    if vis.get("ok"):
        for name in ("gatekeeper", "detector"):
            info = vis.get(name, {})
            checks.append((f"Vision — {name}", info.get("present", False),
                           f"{info.get('size_mb')} MB" if info.get("present")
                           else f"missing at {info.get('path')}"))
    else:
        checks.append(("Vision models", False, vis.get("error", "could not import")))

    st.dataframe(
        pd.DataFrame([{"check": c, "ok": "✅" if ok else "❌", "detail": d}
                      for c, ok, d in checks]),
        hide_index=True, use_container_width=True,
    )

    if rows:
        newest = max(r["uploaded_at"] for r in rows)
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(newest.replace("Z", "+00:00"))).days
        if age > 7:
            st.markdown(
                f"<div class='warnbox'><b>Telemetry is {age} days old.</b> The newest "
                f"reading is {newest[:10]}. Reports asked for a recent window will "
                "correctly return nothing — the charts below therefore show the full "
                "history that does exist, not a recent window.</div>",
                unsafe_allow_html=True)

    st.markdown(
        "<div class='muted'><b>Config in effect</b><br>"
        f"orchestrator <code>{srish_core.ORCHESTRATOR}</code> · "
        f"keep-alive <code>{srish_core.KEEP_ALIVE}</code> · "
        f"farm backend <code>{farm_data.FARM_BACKEND}</code> · "
        f"default farm <code>{srish_core.DEFAULT_FARM_ID}</code></div>",
        unsafe_allow_html=True)


# ── FARM DATA ─────────────────────────────────────────────────────────────────

with tab_farm:
    left, right = st.columns([1, 3])
    with left:
        farm_id = st.selectbox("Farm", farm_ids)
    sensors = probe_sensors(farm_id)

    if not sensors["up"]:
        st.markdown("<div class='warnbox'>Could not reach Supabase.</div>",
                    unsafe_allow_html=True)
    elif not sensors["rows"]:
        st.markdown(f"<div class='warnbox'>No sensor rows for {farm_id}.</div>",
                    unsafe_allow_html=True)
    else:
        rows = sensors["rows"]
        df = pd.DataFrame(rows)

        # uploaded_at, never recorded_at: timestamp_valid is 0 on every row that
        # shipped without an RTC sync, and those all carry 1970-01-01.
        df["ts"] = pd.to_datetime(df["uploaded_at"], errors="coerce", utc=True)
        df = df.sort_values("ts")

        valid_rtc = int(df["timestamp_valid"].fillna(0).astype(int).sum())
        with right:
            st.markdown(
                f"<div class='muted' style='padding-top:28px'>"
                f"{len(df)} rows · {df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d} · "
                f"nodes {', '.join(sorted(df['node_id'].dropna().unique()))} · "
                f"valid RTC on {valid_rtc}/{len(df)} rows</div>",
                unsafe_allow_html=True)

        if valid_rtc == 0:
            st.markdown(
                "<div class='warnbox'><b>No row has a valid RTC timestamp.</b> "
                "Every <code>recorded_at</code> reads 1970-01-01, so all charts are "
                "plotted against <code>uploaded_at</code> instead.</div>",
                unsafe_allow_html=True)

        prof = next((f for f in farms if f["farm_id"] == farm_id), {})

        st.markdown("#### Soil")
        soil = [k for k in farm_data.SOIL_KEYS if df[k].notna().any()] if len(df) else []
        if soil:
            pick = st.multiselect("Soil fields", soil,
                                  default=[k for k in ("vwc_percent", "soil_temp", "ph")
                                           if k in soil] or soil[:3])
            if pick:
                st.line_chart(df.set_index("ts")[pick], color=CHART_COLORS[:len(pick)])

            if "vwc_percent" in df and df["vwc_percent"].notna().any():
                fc = prof.get("field_capacity_vwc")
                wp = prof.get("wilting_point_vwc")
                mad = prof.get("mad_percent")
                latest = df["vwc_percent"].dropna().iloc[-1]
                if None not in (fc, wp, mad):
                    trigger = fc - (mad / 100) * (fc - wp)
                    if latest < wp:
                        st.markdown(
                            f"<div class='warnbox'><b>Soil moisture {latest:.1f}% is below "
                            f"the wilting point ({wp}%).</b> Irrigation trigger for this "
                            f"farm is {trigger:.1f}%. On these numbers the crop is "
                            "water-stressed.</div>", unsafe_allow_html=True)
                    elif latest < trigger:
                        st.markdown(
                            f"<div class='warnbox'>Soil moisture {latest:.1f}% is under the "
                            f"irrigation trigger ({trigger:.1f}%).</div>",
                            unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f"<div class='okbox'>Soil moisture {latest:.1f}% is above the "
                            f"{trigger:.1f}% trigger.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='muted'>No soil readings in this data.</div>",
                        unsafe_allow_html=True)

        st.markdown("#### Environment")
        env = [k for k in farm_data.ENV_KEYS if df[k].notna().any()]
        if env:
            st.line_chart(df.set_index("ts")[env], color=CHART_COLORS[:len(env)])
        else:
            st.markdown("<div class='muted'>No environment readings.</div>",
                        unsafe_allow_html=True)

        st.markdown("#### Reading quality")
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
            q.append({
                "field": k, "rows": len(vals),
                "min": round(float(vals.min()), 2), "max": round(float(vals.max()), 2),
                "latest": round(float(vals.iloc[-1]), 2), "note": note,
            })
        st.dataframe(pd.DataFrame(q), hide_index=True, use_container_width=True)

        st.markdown("#### What Srish is told")
        st.markdown("<div class='muted'>The exact text handed to the LLM. If a number "
                    "is not here, Srish cannot report it.</div>", unsafe_allow_html=True)
        st.code(farm_data.facts_analysis(farm_id, days=3650), language="text")

        with st.expander("Raw rows"):
            st.dataframe(df.drop(columns=["ts"]), hide_index=True, use_container_width=True)


# ── AGENT & NODES ─────────────────────────────────────────────────────────────

with tab_graph:
    st.markdown("#### Agent routing")
    st.markdown(
        "<div class='muted'>Every message takes exactly one path through this graph. "
        "The router decides which, and that decision is what <code>meta.intent</code> "
        "reports.</div>", unsafe_allow_html=True)

    p = PALETTE
    agent_dot = f"""
digraph agent {{
  rankdir=LR;
  bgcolor="transparent";
  node [shape=box style="rounded,filled" fontname="Lato" fontsize=10
        color="{GRAPH_BORDER}" penwidth=1];
  edge [fontname="Lato" fontsize=8 color="{p['text_light']}"];

  msg   [label="user message" fillcolor="{p['petal_yellow']}"];
  router[label="router\\n{srish_core.ROUTER_MODEL}" fillcolor="{p['tomato_light']}"
         fontcolor="{p['tomato']}"];
  kw    [label="keyword override\\n(farm_* rescue)" fillcolor="#ffffff" style="rounded,dashed,filled"];

  greet [label="greeting" fillcolor="{p['light_green']}"];
  off   [label="off_topic\\nrefusal, no LLM" fillcolor="{p['tomato_light']}"];
  dis   [label="tomato_disease" fillcolor="{p['light_green']}"];
  gen   [label="tomato_general" fillcolor="{p['light_green']}"];
  url   [label="url_query" fillcolor="{p['light_green']}"];
  fs    [label="farm_status" fillcolor="{p['light_green']}"];
  fr    [label="farm_report" fillcolor="{p['light_green']}"];

  kb    [label="knowledge base\\n{len(srish_core.load_knowledge_base())} diseases" shape=cylinder
         fillcolor="#ffffff"];
  web   [label="fetch + extract page" fillcolor="#ffffff"];
  sb    [label="farm facts\\n({farm_data.FARM_BACKEND})" shape=cylinder fillcolor="#ffffff"];

  prompt[label="build system prompt" fillcolor="{p['petal_yellow']}"];
  chat  [label="answer\\n{srish_core.CHAT_MODEL}" fillcolor="{p['tomato_light']}"
         fontcolor="{p['tomato']}"];
  out   [label="reply" fillcolor="{p['light_green']}"];

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
             color="{p['tomato']}" fontcolor="{p['tomato']}"];
}}
"""
    st.graphviz_chart(agent_dot, use_container_width=True)

    st.markdown(
        "<div class='muted'>Two edges matter for trust. The <b>off_topic</b> edge skips "
        "the model entirely, so a refusal can never drift into an answer. The dashed "
        "<b>NO_DATA</b> edge exits before generation, which is what stops Srish inventing "
        "sensor values when there are none.</div>", unsafe_allow_html=True)

    st.markdown("#### Node connectivity")
    sensors_all = {fid: probe_sensors(fid) for fid in farm_ids}

    node_lines = []
    for f in farms:
        fid = f["farm_id"]
        rows = sensors_all.get(fid, {}).get("rows", [])
        seen = sorted({r.get("node_id") for r in rows if r.get("node_id")})
        declared = f.get("node_id")

        node_lines.append(
            f'  "{fid}" [label="{fid}\\n{f.get("field_id","")} · '
            f'{f.get("field_area_m2","?")} m²" fillcolor="{p["light_green"]}"];'
        )
        node_lines.append(f'  farm -> "{fid}";')

        if declared and declared not in seen:
            # Profile names a node that has never uploaded anything.
            node_lines.append(
                f'  "{declared}" [label="{declared}\\nno data" '
                f'fillcolor="{p["tomato_light"]}" fontcolor="{p["tomato"]}"];'
            )
            node_lines.append(
                f'  "{fid}" -> "{declared}" [style=dashed color="{p["tomato"]}" '
                f'label="silent"];'
            )
        for n in seen:
            node_lines.append(
                f'  "{n}" [label="{n}\\n{sum(1 for r in rows if r.get("node_id") == n)} rows" '
                f'fillcolor="#ffffff"];'
            )
            node_lines.append(f'  "{fid}" -> "{n}";')

    nodes_dot = f"""
digraph nodes {{
  rankdir=LR; bgcolor="transparent";
  node [shape=box style="rounded,filled" fontname="Lato" fontsize=10
        color="{GRAPH_BORDER}" penwidth=1];
  edge [fontname="Lato" fontsize=8 color="{p['text_light']}"];
  farm [label="Supabase\\nsensor_records" shape=cylinder fillcolor="{p['petal_yellow']}"];
{chr(10).join(node_lines)}
}}
"""
    st.graphviz_chart(nodes_dot, use_container_width=True)

    silent = [f for f in farms
              if f.get("node_id") and f["node_id"] not in
              {r.get("node_id") for r in sensors_all.get(f["farm_id"], {}).get("rows", [])}]
    if silent:
        st.markdown(
            "<div class='warnbox'><b>"
            f"{len(silent)} of {len(farms)} farms have a node in their profile that has "
            "never uploaded a row.</b> Shown dashed above: "
            + ", ".join(f"{f['farm_id']} → {f['node_id']}" for f in silent)
            + ".</div>", unsafe_allow_html=True)

    with st.expander("Farm profiles"):
        st.dataframe(pd.DataFrame(farms), hide_index=True, use_container_width=True)


# ── VISION ────────────────────────────────────────────────────────────────────

with tab_vision:
    st.markdown("#### Leaf verification and disease detection")
    st.markdown(
        "<div class='muted'>The gatekeeper classifier runs first. The detector only "
        "sees images it accepts, so a photo of something else is rejected rather than "
        "scanned for lesions.</div>", unsafe_allow_html=True)

    if not vis.get("ok"):
        st.markdown(f"<div class='warnbox'>Vision unavailable: "
                    f"{vis.get('error')}</div>", unsafe_allow_html=True)
    elif not vis.get("gatekeeper", {}).get("present"):
        st.markdown(
            "<div class='warnbox'><b>Gatekeeper weights missing.</b><br>"
            f"Looked in <code>{vis['gatekeeper']['path']}</code>. Weights are gitignored, "
            "so put <code>keeper.pt</code> and <code>yolo.pt</code> next to the scripts, "
            "or set <code>SRISH_GATEKEEPER_PATH</code> / <code>SRISH_DETECTOR_PATH</code>."
            "</div>", unsafe_allow_html=True)
    else:
        import srish_vision

        src = st.radio("Source", ["Upload", "Camera"], horizontal=True,
                       label_visibility="collapsed")
        img_bytes = None
        if src == "Upload":
            up = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "webp"],
                                  label_visibility="collapsed")
            if up:
                img_bytes = up.getvalue()
        else:
            shot = st.camera_input("Capture", label_visibility="collapsed")
            if shot:
                img_bytes = shot.getvalue()

        if img_bytes:
            pil = srish_vision.bytes_to_pil(img_bytes)
            frame = srish_vision.pil_to_cv2(pil)
            c1, c2 = st.columns(2)
            c1.image(pil, caption="Input", use_container_width=True)

            with st.spinner("Checking it is a tomato leaf..."):
                is_leaf, conf, cls = srish_vision.check_if_leaf(frame)

            if not is_leaf:
                c2.markdown(
                    f"<div class='warnbox'><b>Rejected by the gatekeeper.</b><br>"
                    f"Classified as <code>{cls}</code> at {conf:.0%} confidence; needs "
                    f"<code>{srish_vision.LEAF_CLASS}</code> at "
                    f"{srish_vision.LEAF_CONFIDENCE:.0%} or more.<br>"
                    "The detector was not run.</div>", unsafe_allow_html=True)
            else:
                c2.markdown(f"<div class='okbox'>Accepted as a tomato leaf "
                            f"({conf:.0%} confidence).</div>", unsafe_allow_html=True)
                if not vis.get("detector", {}).get("present"):
                    c2.markdown("<div class='warnbox'>Detector weights missing, so no "
                                "disease scan.</div>", unsafe_allow_html=True)
                else:
                    with st.spinner("Looking for Leaf Miner damage..."):
                        res = srish_vision.detect_disease(frame)
                    c2.image(res["annotated"][:, :, ::-1], caption="Detections",
                             use_container_width=True)
                    if res["found"]:
                        c2.markdown(f"<div class='warnbox'><b>{res['count']} detection(s)."
                                    "</b></div>", unsafe_allow_html=True)
                        c2.dataframe(pd.DataFrame(res["detections"]), hide_index=True,
                                     use_container_width=True)
                    else:
                        c2.markdown("<div class='okbox'>No Leaf Miner damage found. "
                                    "Note the detector only knows that one class.</div>",
                                    unsafe_allow_html=True)

    with st.expander("Weights"):
        st.dataframe(
            pd.DataFrame([
                {"model": k, "present": v.get("present"),
                 "size_mb": v.get("size_mb"), "path": v.get("path")}
                for k, v in vis.items() if isinstance(v, dict)
            ]), hide_index=True, use_container_width=True)


# ── CHAT ──────────────────────────────────────────────────────────────────────

with tab_chat:
    st.markdown("#### Chat, with the routing exposed")
    st.markdown(
        "<div class='muted'>Same brain the mobile app calls. Each answer shows the "
        "intent chosen and whether the knowledge base was used, which is the quickest "
        "way to tell a routing bug from a generation one.</div>",
        unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1, 2])
    depth = c1.toggle("Detailed answers", value=False)
    chat_farm = c2.selectbox("Farm context", farm_ids, key="chat_farm")
    if c3.button("Clear conversation"):
        st.session_state["portal_msgs"] = []
        st.rerun()

    msgs = st.session_state.setdefault("portal_msgs", [])

    for m in msgs:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("meta"):
                st.caption(m["meta"])

    if prompt := st.chat_input("Ask Srish something..."):
        msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            holder = st.empty()
            meta_holder = st.caption("routing...")
            acc, info = "", {}
            try:
                for kind, payload in srish_core.answer_stream(
                    message=prompt,
                    history=[{"role": m["role"], "content": m["content"]}
                             for m in msgs[:-1]],
                    farm_id=chat_farm,
                    depth_mode=depth,
                ):
                    if kind == "meta":
                        meta_holder.caption(
                            f"intent `{payload.get('intent')}` · "
                            f"lang `{payload.get('language')}`"
                            + ("  ·  ⚠️ DEGRADED" if payload.get("degraded") else "")
                        )
                    elif kind == "token":
                        acc += payload
                        holder.markdown(acc)
                    else:
                        info = payload
                        holder.markdown(payload.get("reply", acc))
            except Exception as e:
                acc = f"Failed: {e}"
                holder.markdown(acc)

            bits = [f"intent `{info.get('intent')}`", f"lang `{info.get('language')}`"]
            if info.get("used_kb"):
                bits.append(f"KB `{info.get('kb_disease')}`")
            if info.get("degraded"):
                bits.append("⚠️ **degraded** — router down, off-topic filter and KB skipped")
            if info.get("error"):
                bits.append(f"error `{str(info['error'])[:80]}`")
            caption = " · ".join(bits)
            meta_holder.caption(caption)

            msgs.append({"role": "assistant",
                         "content": info.get("reply", acc), "meta": caption})
