import os
import requests

# Single n8n webhook. The workflow branches internally on "mode" and always
# returns a consistent JSON shape — no direct Supabase access from this file.
N8N_FARM_URL = os.environ.get("N8N_FARM_URL", "http://localhost:5678/webhook/farm-data")


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


def _format_fields(fields: dict) -> list[str]:
    """Renders whatever per-field stats n8n sends back (latest/mean/min/max/trend/anomaly).

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


def facts_latest_status(farm_id: str) -> str:
    """Current/live snapshot, formatted as facts for the LLM."""
    a = _call_n8n(farm_id, mode="status", days=1)
    if a.get("status") == "ERROR":
        return f"NO_DATA: farm data service unreachable ({a.get('error')})"
    if a.get("status") != "OK":
        return "NO_FARM_DATA: no sensor rows found for this farm."

    lines = ["LIVE_FARM_DATA:"]
    lines.extend(_format_fields(a.get("fields", {})))
    return "\n".join(lines)


def get_report(farm_id: str, days: int = 7) -> tuple[str, list[dict]]:
    """Time-range report. Returns (facts_text, rows) — rows are used for charting."""
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


def facts_analysis(farm_id: str, days: int = 3) -> str:
    """Judgment-style analysis: numbers plus the farm's agronomic thresholds,
    so the LLM can decide OK vs NOT OK instead of just reciting values."""
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
