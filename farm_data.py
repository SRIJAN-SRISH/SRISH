"""Live farm facts for Srish, formatted as plain text the LLM can narrate.

Two backends, chosen with FARM_BACKEND:

  supabase  (default) reads sensor_records and farm_profiles straight from
            Supabase. No extra service to run.
  n8n       posts to the n8n webhook, which aggregates server-side.

Every public function returns text that either starts with a NO_DATA/
NO_FARM_DATA marker or carries labelled numbers. Callers only have to check
the prefix, and srish_core refuses to invoke the LLM on a NO_ marker so it
cannot invent sensor values.
"""

import os
from datetime import datetime, timedelta, timezone

import requests

FARM_BACKEND = os.environ.get("FARM_BACKEND", "supabase").lower()

# Single n8n webhook. The workflow branches internally on "mode" and always
# returns a consistent JSON shape.
N8N_FARM_URL = os.environ.get("N8N_FARM_URL", "http://localhost:5678/webhook/farm-data")

# This is the Srijan app's publishable (anon) key, which already ships in the
# app's own source and is protected by row-level security -- app_users, for
# instance, refuses to be read with it. It is not a service-role key and gives
# no write or admin access. Override both for a different project.
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL", "https://hrubrokdmvnpzejqjuqa.supabase.co"
).rstrip("/")
SUPABASE_KEY = os.environ.get(
    "SUPABASE_KEY", "sb_publishable_ki003MnbTVLvNppgRrZq1A_IKZJQlc9"
)

# Mirrors services/api/sensorService.jsx in the app, so both read the same set.
SOIL_KEYS = ["vwc_percent", "soil_temp", "ec", "ph", "nitrogen", "phosphorus", "potassium"]
ENV_KEYS = ["air_temp", "humidity", "pressure", "rainfall"]
ALL_KEYS = SOIL_KEYS + ENV_KEYS

# Ranges a working sensor should report. Anything outside is flagged rather than
# dropped, so the farmer hears "this reading looks wrong" instead of silence.
PLAUSIBLE = {
    "vwc_percent": (0, 60),
    "soil_temp": (5, 50),
    "ec": (0, 5),
    "ph": (4.0, 9.0),
    "air_temp": (0, 55),
    "humidity": (5, 100),
    "pressure": (850, 1100),
}


# ── SUPABASE ──────────────────────────────────────────────────────────────────

def _sb(path: str, params: dict) -> list | None:
    """GET from the Supabase REST API. Returns None on any failure."""
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/{path}",
            params=params,
            headers={
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Accept": "application/json",
            },
            timeout=20,
        )
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _rows(farm_id: str, days: int | None = None, limit: int = 500) -> list | None:
    """Newest-first sensor rows for a farm, optionally limited to a window.

    Ordered and filtered by uploaded_at, never recorded_at: Node B ships
    timestamp_valid=0 whenever its RTC was not synced, and every such row
    carries recorded_at = 1970-01-01, which would make any window empty.
    """
    params = {
        "select": "*",
        "farm_id": f"eq.{farm_id}",
        "order": "uploaded_at.desc,id.desc",
        "limit": str(limit),
    }
    if days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        params["uploaded_at"] = f"gte.{cutoff.isoformat()}"
    return _sb("sensor_records", params)


def _newest_age(farm_id: str) -> str:
    """How stale the data is, for an honest 'no recent readings' message."""
    rows = _rows(farm_id, days=None, limit=1)
    if not rows:
        return "no readings at all for this farm"
    stamp = rows[0].get("uploaded_at") or ""
    try:
        when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - when).days
        return f"newest reading is {stamp[:10]}, {age} days ago"
    except Exception:
        return f"newest reading is {stamp[:10]}"


def _trend(values: list[float]) -> str:
    """Compare the newer half against the older half.

    values arrive newest-first, so values[:h] is the recent half.
    """
    if len(values) < 4:
        return "flat"
    h = len(values) // 2
    recent = sum(values[:h]) / h
    older = sum(values[h:]) / len(values[h:])
    if older == 0:
        return "flat"
    change = (recent - older) / abs(older)
    if change > 0.1:
        return "rising"
    if change < -0.1:
        return "falling"
    return "flat"


def _stats(rows: list, with_range: bool) -> dict:
    """Per-field stats in the shape _format_fields expects.

    Fields that are null in every row are left out entirely rather than
    reported as zero -- a sensor that never reported is not a reading of 0.
    """
    out: dict = {}
    for key in ALL_KEYS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        vals = [float(v) for v in vals if isinstance(v, (int, float))]
        if not vals:
            continue

        s: dict = {"latest": round(vals[0], 2)}
        if with_range and len(vals) > 1:
            s["mean"] = round(sum(vals) / len(vals), 2)
            s["min"] = round(min(vals), 2)
            s["max"] = round(max(vals), 2)
            s["trend"] = _trend(vals)

        lo, hi = PLAUSIBLE.get(key, (None, None))
        if lo is not None and not (lo <= vals[0] <= hi):
            s["anomaly"] = True
        # A sensor stuck on one value across many readings is also suspect.
        elif len(vals) >= 5 and min(vals) == max(vals):
            s["anomaly"] = True

        out[key] = s
    return out


def _thresholds(farm_id: str) -> dict:
    """Agronomic limits from farm_profiles, so the LLM can judge OK vs NOT OK."""
    rows = _sb("farm_profiles", {"select": "*", "farm_id": f"eq.{farm_id}", "limit": "1"})
    if not rows:
        return {}
    p = rows[0]
    fc, wp = p.get("field_capacity_vwc"), p.get("wilting_point_vwc")
    mad = p.get("mad_percent")

    t = {
        "field_capacity_vwc": fc,
        "wilting_point_vwc": wp,
        "mad_percent": mad,
    }
    # Below this, the crop is past the allowed depletion and wants irrigating.
    if None not in (fc, wp, mad):
        t["irrigate_below_vwc"] = round(fc - (mad / 100) * (fc - wp), 1)
    return {k: v for k, v in t.items() if v is not None}


# ── N8N ───────────────────────────────────────────────────────────────────────

def _call_n8n(farm_id: str, mode: str, days: int) -> dict:
    """POSTs {farm_id, mode, days} to the n8n workflow and returns parsed JSON.

    On any network/HTTP failure returns {"status": "ERROR", "error": ...} so
    callers have one shape to check instead of catching exceptions everywhere.
    """
    try:
        r = requests.post(
            N8N_FARM_URL,
            json={"farm_id": farm_id, "mode": mode, "days": days},
            timeout=20,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


# ── FORMATTING ────────────────────────────────────────────────────────────────

def _format_fields(fields: dict) -> list[str]:
    """Renders whatever per-field stats the backend produced.

    Uses .get() throughout since status responses may only carry "latest"
    while report/analysis responses carry the full stat set.
    """
    lines = []
    for name, s in fields.items():
        parts = []
        if "latest" in s:
            parts.append(f"latest={s['latest']}")
        if "mean" in s:
            parts.append(f"avg={s['mean']}")
        if "min" in s and "max" in s:
            parts.append(f"range=[{s['min']},{s['max']}]")
        if "trend" in s:
            parts.append(f"trend={s['trend']}")
        flag = " ⚠ ANOMALY" if s.get("anomaly") else ""
        lines.append(f"- {name}: {', '.join(parts)}{flag}")
    return lines


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def facts_latest_status(farm_id: str) -> str:
    """Current snapshot, formatted as facts for the LLM."""
    if FARM_BACKEND == "n8n":
        a = _call_n8n(farm_id, mode="status", days=1)
        if a.get("status") == "ERROR":
            return f"NO_DATA: farm data service unreachable ({a.get('error')})"
        if a.get("status") != "OK":
            return "NO_FARM_DATA: no sensor rows found for this farm."
        return "\n".join(["LIVE_FARM_DATA:"] + _format_fields(a.get("fields", {})))

    rows = _rows(farm_id, days=None, limit=20)
    if rows is None:
        return "NO_DATA: could not reach the farm database."
    if not rows:
        return "NO_FARM_DATA: no sensor rows found for this farm."

    fields = _stats(rows, with_range=False)
    if not fields:
        return "NO_FARM_DATA: sensor rows exist but every reading is empty."

    stamp = (rows[0].get("uploaded_at") or "")[:19].replace("T", " ")
    return "\n".join(
        [f"LIVE_FARM_DATA (as of {stamp} UTC, node {rows[0].get('node_id', '?')}):"]
        + _format_fields(fields)
    )


def get_report(farm_id: str, days: int = 7) -> tuple[str, list[dict]]:
    """Time-range report. Returns (facts_text, rows) -- rows are for charting."""
    if FARM_BACKEND == "n8n":
        a = _call_n8n(farm_id, mode="report", days=days)
        if a.get("status") == "ERROR":
            return f"NO_DATA: farm data service unreachable ({a.get('error')})", []
        if a.get("status") != "OK":
            return "NO_FARM_DATA: no sensor rows found for this window.", []
        lines = [
            f"FARM_REPORT_FACTS ({a.get('n_readings', 0)} readings, "
            f"{a.get('window_start', '?')} to {a.get('window_end', '?')}):"
        ]
        lines.extend(_format_fields(a.get("fields", {})))
        return "\n".join(lines), a.get("rows", [])

    rows = _rows(farm_id, days=days)
    if rows is None:
        return "NO_DATA: could not reach the farm database.", []
    if not rows:
        # Deliberately not falling back to older readings: a report labelled
        # "last N days" must not quietly show data from outside that window.
        return (
            f"NO_FARM_DATA: no sensor rows in the last {days} days "
            f"({_newest_age(farm_id)}).",
            [],
        )

    fields = _stats(rows, with_range=True)
    if not fields:
        return "NO_FARM_DATA: sensor rows exist but every reading is empty.", []

    start = (rows[-1].get("uploaded_at") or "")[:10]
    end = (rows[0].get("uploaded_at") or "")[:10]
    lines = [f"FARM_REPORT_FACTS ({len(rows)} readings, {start} to {end}):"]
    lines.extend(_format_fields(fields))

    # Charting wants oldest-first, and only rows with a usable timestamp.
    chart = [
        {"recorded_at": r["uploaded_at"], "vwc_percent": r.get("vwc_percent")}
        for r in reversed(rows)
        if r.get("uploaded_at") and r.get("vwc_percent") is not None
    ]
    return "\n".join(lines), chart


def facts_analysis(farm_id: str, days: int = 3) -> str:
    """Numbers plus the farm's agronomic thresholds, so the LLM can judge
    OK vs NOT OK instead of just reciting values."""
    if FARM_BACKEND == "n8n":
        a = _call_n8n(farm_id, mode="analysis", days=days)
        if a.get("status") == "ERROR":
            return f"NO_DATA: farm data service unreachable ({a.get('error')})"
        if a.get("status") != "OK":
            return "NO_FARM_DATA: no sensor rows found for this window."
        lines = [
            f"FARM_ANALYSIS_FACTS ({a.get('n_readings', 0)} readings, "
            f"{a.get('window_start', '?')} to {a.get('window_end', '?')}):"
        ]
        lines.extend(_format_fields(a.get("fields", {})))
        if a.get("unhealthy_readings"):
            lines.append(f"- {a['unhealthy_readings']} readings flagged unhealthy")
        thresholds = a.get("thresholds")
        if thresholds:
            lines.append("\nFARM_AGRONOMIC_THRESHOLDS (use these to judge OK vs NOT OK):")
            for key, val in thresholds.items():
                lines.append(f"- {key}: {val}")
        return "\n".join(lines)

    rows = _rows(farm_id, days=days)
    if rows is None:
        return "NO_DATA: could not reach the farm database."
    if not rows:
        return (
            f"NO_FARM_DATA: no sensor rows in the last {days} days "
            f"({_newest_age(farm_id)})."
        )

    fields = _stats(rows, with_range=True)
    if not fields:
        return "NO_FARM_DATA: sensor rows exist but every reading is empty."

    start = (rows[-1].get("uploaded_at") or "")[:10]
    end = (rows[0].get("uploaded_at") or "")[:10]
    lines = [f"FARM_ANALYSIS_FACTS ({len(rows)} readings, {start} to {end}):"]
    lines.extend(_format_fields(fields))

    unhealthy = sum(1 for r in rows if r.get("health_flag"))
    if unhealthy:
        lines.append(f"- {unhealthy} readings carry a non-zero health_flag")

    thresholds = _thresholds(farm_id)
    if thresholds:
        lines.append("\nFARM_AGRONOMIC_THRESHOLDS (use these to judge OK vs NOT OK):")
        for key, val in thresholds.items():
            lines.append(f"- {key}: {val}")

    return "\n".join(lines)
