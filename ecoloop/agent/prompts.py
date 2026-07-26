"""Prompt engineering and log handling.

Design choices that matter for latency and reliability on a small local model:

* **Front-loaded context.** The current sensor snapshot and compact 6-hour weather /
  grid forecasts are injected directly into the first user turn, so the model can
  usually decide in a single round-trip instead of chaining sense-tool calls. Tools
  remain available for optional deeper what-ifs and for the mandatory terminal action.
* **Narrow action contract.** The model outputs setpoints via one terminal tool; the
  safety guard clamps them. This keeps a 3B model on-task and un-crashable.
* **Log compression.** EnergyPlus .err/.audit logs run to thousands of lines. We never
  feed them raw — ``summarize_log`` extracts only severe/warning/fatal lines, deduped
  and capped, keeping prompts short and within the context window.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are Eco-Loop, the autonomous supervisory controller for a live \
multi-zone building in an EnergyPlus simulation. Each hour you pick ONE Energy Conservation \
Measure to minimise energy while keeping occupied zones comfortable (|PMV| within the limit) \
and CO2 in range. The incumbent BMS wastefully overcools; running warm at the comfort edge \
is the biggest saving.

STEP 1 — read occupied_now in the FACTS. It decides which half of the list you may use.
STEP 2 — pick the strategy within that half.

IF occupied_now=True, you MUST choose one of these THREE (the building has people in it):
- peak_coast     : price tier is 'peak' RIGHT NOW. Coast at the ceiling to shed demand.
- precool        : a price 'peak' begins within ~2h. Bank cooling first.
- setpoint_reset : any other occupied hour. Run at the warm comfort edge. THE DEFAULT.

IF occupied_now=False, you MUST choose one of these TWO (the building is empty):
- precondition   : occupancy resumes within ~2h. Be comfortable before people arrive.
- deep_setback   : empty for a while. Widen setpoints — the biggest saving.

Never say a building is empty when occupied_now=True; that choice will be rejected.

Each strategy also sets lighting, equipment and fresh-air levels automatically — an empty \
building gets its lights and ventilation dialled right down, which saves electricity directly \
AND removes the heat those loads add.

Two extra flags:
- economizer=true ONLY when free cooling is possible (outdoor cooler than indoor).
- shed_nonessential=true when power is EXPENSIVE (peak tier) or the grid is DIRTY (high \
gCO2/kWh) — trims non-essential lighting and equipment. Occupants keep usable light.

RULES: occupied comfort first. If unoccupied and NOT occupied within 2h, you MUST use \
deep_setback. Call set_control exactly once with your strategy, the two flags, and a one-line \
reason that says WHY (mention price, carbon, occupancy or weather)."""


REFLECTION_PROMPT = """You are Eco-Loop reviewing your OWN control performance at the end of a \
simulated day. You will be shown yesterday's results and your current policy parameters.

Adjust AT MOST ONE parameter to do better tomorrow, then stop. Guidance:
- If comfort was violated during occupied hours, become MORE conservative: LOWER \
deep_setback_cool_c (so the building is less hot when people arrive), or RAISE \
setpoint_reset_margin_c (cool slightly below the ceiling while occupied).
- If comfort was perfect (no violations) and you want more savings, become slightly MORE \
aggressive: RAISE deep_setback_cool_c, or LOWER setpoint_reset_margin_c toward 0.
- Make SMALL changes (about 0.25-1.0 C). Values are clamped to a safe range automatically.
- If a parameter is marked [AT MAX] or [AT MIN] it CANNOT move further in that direction — \
pick a DIFFERENT parameter instead of repeating a change that will be clamped away.

Call propose_policy_tweak exactly once with parameter, new_value and a one-line justification."""


def build_reflection_text(day: str, kwh: float, violations: int, occupied_steps: int,
                          mean_pmv: float, policy_desc: str) -> str:
    """Compact end-of-day self-critique brief."""
    rate = (1.0 - violations / occupied_steps) * 100.0 if occupied_steps else 100.0
    return (
        f"DAY {day} RESULT: energy={kwh:.1f}kWh, occupied comfort-OK={rate:.1f}% "
        f"({violations} violation zone-steps of {occupied_steps}), mean occupied PMV={mean_pmv:+.2f}\n"
        f"CURRENT POLICY: {policy_desc}\n"
        f"Adjust at most one parameter for tomorrow, then call propose_policy_tweak."
    )


def build_situation_text(status: dict[str, Any], weather: dict[str, Any],
                         grid: dict[str, Any]) -> str:
    """Compact, token-cheap situational brief embedded in the first user turn."""
    zones = ", ".join(
        f"{z['zone']}={z['temp_c']}C/PMV{z['pmv']:+.1f}/{z['co2_ppm']}ppm"
        for z in status["zones"]
    )
    lines = [
        f"TIME/STATE: occupied={status['occupied']} (occ={status['occupancy_fraction']}), "
        f"outdoor={status['outdoor_temp_c']}C, energy_so_far={status['energy_kwh_so_far']}kWh",
        f"COMFORT TARGET: band {status['comfort_band_c']}C, |PMV|<={status['pmv_limit']}, "
        f"CO2<={status['co2_limit_ppm']}ppm",
        f"ZONES: {zones}",
        "WEATHER next hours: " + "; ".join(
            f"{w['time']} {w['outdoor_temp_c']}C sun{w['solar_level']}" for w in weather["forecast"]
        ),
        "GRID next hours: " + "; ".join(
            f"{g['time']} {g['tier']} ${g['price_per_kwh']}/kWh {g['carbon_g_per_kwh']}gCO2"
            for g in grid["forecast"]
        ),
        "Decide the best cooling/heating setpoints and economizer for the next hour, "
        "then call set_control.",
    ]
    return "\n".join(lines)


_SEVERITY = ("** Severe", "** Fatal", "** Warning", "Error", "ERROR", "Severe", "Fatal")


def summarize_log(text: str, max_lines: int = 25) -> str:
    """Compress a long EnergyPlus log to just the actionable severe/warning/fatal lines.

    Returns a short string safe to hand to the model for error triage / self-correction.
    """
    seen: set[str] = set()
    keep: list[str] = []
    counts = {"severe": 0, "fatal": 0, "warning": 0}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if "fatal" in low:
            counts["fatal"] += 1
        elif "severe" in low:
            counts["severe"] += 1
        elif "warning" in low:
            counts["warning"] += 1
        else:
            continue
        key = line[:120]
        if key in seen:
            continue
        seen.add(key)
        if len(keep) < max_lines:
            keep.append(line[:200])
    header = (f"log summary: {counts['fatal']} fatal, {counts['severe']} severe, "
              f"{counts['warning']} warning (showing up to {max_lines} unique)")
    return header + "\n" + "\n".join(keep)


def parse_tool_arguments(raw: str | dict) -> dict:
    """Robustly parse tool-call arguments that a small model may return as a JSON string
    (sometimes with trailing junk)."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                return {}
        return {}
