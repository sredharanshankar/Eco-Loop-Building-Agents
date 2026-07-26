"""Presentation dashboard — an interactive replay, and a live streaming mode.

Two surfaces over one UI:

* **Replay** (``python run.py live-dashboard``) writes a single self-contained HTML file
  from a finished run. Judges open it and press play: the whole week replays with the
  agent's real logged reasoning. Nothing to install, no server, cannot stall — this is
  the artifact you submit.
* **Live** (``python run.py agent --live``) starts a stdlib HTTP server that streams the
  *running* control loop into the same UI, so you can watch EnergyPlus feed the agent and
  the agent inject setpoints back, in real time. Used for the demo video.

Deliberately dependency-free: Python's ``http.server`` and hand-rolled SVG charts, with
no CDN and no external fonts, so it renders offline on a judge's machine where external
scripts may be blocked.
"""

from __future__ import annotations

import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# Series kept for the charts. Column-oriented and trimmed on purpose: the full 36-column
# telemetry for 673 steps would bloat the embedded payload for no visual gain.
_AGENT_COLS = ["time", "outdoor_temp", "occupancy", "cumulative_kwh", "step_kwh",
               "mean_zone_temp", "cooling_setpoint", "heating_setpoint",
               "max_abs_pmv", "comfort_violations", "cumulative_cost", "cumulative_co2_kg",
               # Required feedback metrics: indoor air quality and PMV comfort, plus the
               # grid signals the agent reasons against (price tier / carbon intensity).
               "peak_co2", "carbon", "price",
               # End-use split, so savings can be attributed rather than just totalled.
               "cumulative_kwh_hvac", "cumulative_kwh_lights", "cumulative_kwh_plugs",
               "light_level", "vent_level"]
_BASE_COLS = ["time", "cumulative_kwh", "mean_zone_temp", "cooling_setpoint",
              "cumulative_cost", "cumulative_co2_kg", "peak_co2", "max_abs_pmv",
              "cumulative_kwh_hvac", "cumulative_kwh_lights", "cumulative_kwh_plugs",
              # Needed so peak demand and comfort can be computed up to the playhead,
              # rather than the KPI row showing end-of-run totals during a replay.
              "step_kwh", "comfort_violations", "occupancy"]


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _read_series(path: Path, cols: list[str]) -> dict[str, list]:
    """Read selected columns as parallel arrays, preserving genuine gaps.

    The very first telemetry row is the ``init`` record, written before any control
    action exists, so the setpoint/level columns are blank. Coercing those to 0.0 made
    the setpoint line dive off the bottom of the chart; ``None`` instead lets the
    renderer skip them.
    """
    out: dict[str, list] = {c: [] for c in cols}
    if not path.exists():
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            for c in cols:
                v = row.get(c, "")
                if c == "time":
                    out[c].append(v)
                elif v is None or v == "":
                    out[c].append(None)
                else:
                    out[c].append(round(_f(v), 3))
    return out


def _baseline_if_compatible(results_dir: Path, backend: str | None,
                            first_time: str | None = None) -> dict[str, list]:
    """Load the baseline series only if it can be validly compared.

    A leftover baseline from a different backend (or a different start date) silently
    produces a garbage comparison — an RC agent charted against an EnergyPlus baseline
    shows the AI "using twice the energy". Better to show no ghost than a wrong one.
    """
    none: dict[str, list] = {c: [] for c in _BASE_COLS}
    base = _read_series(results_dir / "baseline_timeseries.csv", _BASE_COLS)
    if not base["time"]:
        return none
    summ = _load_json(results_dir / "baseline_summary.json", {})
    base_backend = summ.get("backend")
    if backend:
        # Fail closed: an unverifiable baseline is treated as incompatible. Missing
        # metadata is exactly how a stale CSV sneaks into a comparison unnoticed.
        if not base_backend:
            print("[live] ignoring baseline - no baseline_summary.json to verify it against")
            return none
        if base_backend != backend:
            print(f"[live] ignoring baseline from a '{base_backend}' run "
                  f"(this run is '{backend}') - comparison would be invalid")
            return none
    if first_time and base["time"][0] != first_time:
        print("[live] ignoring baseline - it starts at a different time than this run")
        return none
    return base


def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return default
    return default


def build_payload(results_dir: str | Path) -> dict[str, Any]:
    """Assemble everything the UI needs from a finished run."""
    d = Path(results_dir)
    agent = _read_series(d / "agent_timeseries.csv", _AGENT_COLS)
    agent_summary = _load_json(d / "agent_summary.json", {})
    base = _baseline_if_compatible(d, agent_summary.get("backend"),
                                   agent["time"][0] if agent["time"] else None)
    # Guard against comparing runs of different lengths (e.g. a stale baseline).
    n = len(agent["time"])
    if base["time"] and len(base["time"]) != n:
        m = min(n, len(base["time"]))
        agent = {k: v[:m] for k, v in agent.items()}
        base = {k: v[:m] for k, v in base.items()}
        n = m
    # Comfort is scored per zone-step in telemetry.py; the UI needs the zone count to
    # reproduce that exactly, otherwise its tile disagrees with the headline summary.
    occ_steps = sum(1 for v in agent.get("occupancy", []) if (v or 0) > 0.1)
    zones = agent_summary.get("occupied_zone_steps", 0) / occ_steps if occ_steps else 1
    return {
        "mode": "replay",
        "running": False,
        "n_steps": n,
        "n_zones": max(1, round(zones)),
        "agent": agent,
        "baseline": base,
        "decisions": _load_json(d / "agent_decisions.json", []),
        "reflections": _load_json(d / "agent_reflections.json", []),
        "savings": _load_json(d / "dashboard_savings.json", {}),
        "stats": _load_json(d / "agent_runstats.json", {}),
        "summary": _load_json(d / "agent_summary.json", {}),
        "chaos": _load_json(d / "chaos_report.json", {}),
    }


# ---------------------------------------------------------------------------
# Live state: written by the control loop, read by the HTTP handler.
# ---------------------------------------------------------------------------
class LiveState:
    """Thread-safe snapshot of an in-flight run.

    The control loop calls :meth:`push` once per timestep. Everything is bounded and
    lock-protected, and every method is exception-safe — a dashboard problem must never
    be able to disturb the control loop.
    """

    def __init__(self, backend: str, n_steps: int, results_dir: str | Path = "results",
                 n_zones: int = 1):
        self._lock = threading.Lock()
        self.backend = backend
        self.n_steps = n_steps
        self.n_zones = max(1, int(n_zones))
        self.running = True
        self.agent: dict[str, list] = {c: [] for c in _AGENT_COLS}
        self.decisions: list[dict] = []
        self.reflections: list[dict] = []
        self.stats: dict[str, Any] = {}
        # A baseline from an earlier phase (compare runs it first) gives the live view
        # its comparison ghost. Absent for a bare `agent --live`, which is fine — and
        # deliberately dropped if it came from an incompatible run.
        self.baseline = _baseline_if_compatible(Path(results_dir), backend)

    def push(self, obs, meta: dict) -> None:
        try:
            row = obs.to_row()
            with self._lock:
                for c in _AGENT_COLS:
                    v = row.get(c, 0)
                    self.agent[c].append(v if c == "time" else round(_f(v), 3))
                src = meta.get("source")
                if src in ("llm", "fallback"):
                    aa = obs.applied_action or {}
                    self.decisions.append({
                        "time": row.get("time"), "source": src,
                        "latency_s": meta.get("latency_s"),
                        "cooling_setpoint": aa.get("cooling_setpoint"),
                        "heating_setpoint": aa.get("heating_setpoint"),
                        "economizer": aa.get("economizer"),
                        "reason": aa.get("reason", ""),
                        "safety": meta.get("safety"),
                    })
        except Exception:  # noqa: BLE001 - telemetry must never break control
            pass

    def push_reflection(self, entry: dict) -> None:
        try:
            with self._lock:
                self.reflections.append(entry)
        except Exception:  # noqa: BLE001
            pass

    def finish(self, stats: dict | None = None) -> None:
        with self._lock:
            self.running = False
            if stats:
                self.stats = stats

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "mode": "live", "running": self.running,
                "backend": self.backend, "n_steps": self.n_steps,
                "n_zones": self.n_zones,
                "agent": {k: list(v) for k, v in self.agent.items()},
                "baseline": self.baseline,
                "decisions": list(self.decisions),
                "reflections": list(self.reflections),
                "stats": dict(self.stats),
                "savings": {}, "summary": {}, "chaos": {},
            }


class _Handler(BaseHTTPRequestHandler):
    state: LiveState | None = None

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        if self.path.startswith("/api/state"):
            body = json.dumps(self.state.snapshot() if self.state else {}).encode()
            self._send(body, "application/json")
        elif self.path in ("/", "/index.html"):
            self._send(render_html(None, live=True).encode(), "text/html; charset=utf-8")
        else:
            self.send_error(404)

    def _send(self, body: bytes, ctype: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass   # browser navigated away mid-response

    def log_message(self, *args) -> None:
        pass       # keep the control-loop console clean


def start_live_server(state: LiveState, port: int = 8765) -> HTTPServer | None:
    """Start the dashboard server on a daemon thread. Returns None if the port is busy."""
    try:
        _Handler.state = state
        httpd = HTTPServer(("127.0.0.1", port), _Handler)
    except OSError as exc:
        print(f"[live] could not start dashboard on port {port}: {exc}")
        return None
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    print(f"[live] dashboard running at http://127.0.0.1:{port}  (Ctrl-C stops the run)")
    return httpd


def write_replay(results_dir: str | Path, out_path: str | Path | None = None) -> str:
    """Write the self-contained replay dashboard and return its path."""
    d = Path(results_dir)
    out = Path(out_path) if out_path else d / "dashboard_live.html"
    payload = build_payload(d)
    if not payload["n_steps"]:
        raise SystemExit(f"no timeseries found in {d} — run a simulation first")
    out.write_text(render_html(payload, live=False), encoding="utf-8")
    return str(out)


def render_html(payload: dict[str, Any] | None, live: bool) -> str:
    data_js = "null" if live else json.dumps(payload, separators=(",", ":"))
    return _TEMPLATE.replace("__LIVE__", "true" if live else "false").replace(
        "__DATA__", data_js)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Eco-Loop — Autonomous Building Control</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#0e1116;color:#e9edf2;font-family:system-ui,'Segoe UI',Arial,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:18px 20px 32px}
header{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-bottom:6px}
h1{font-size:20px;margin:0;font-weight:700;letter-spacing:.2px}
.sub{color:#8d97a3;font-size:12.5px;margin:2px 0 16px}
.badge{font-size:11px;padding:4px 9px;border-radius:999px;border:1px solid #2a3140;color:#9fb0c3;background:#161b23}
.badge.ok{border-color:#1f6b3f;color:#79e0a4;background:#102a1c}
.badge.live{border-color:#8a2f2f;color:#ff8f8f;background:#2a1212}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:#ff5c5c;margin-right:5px;animation:p 1.2s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.25}}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:11px;margin-bottom:16px}
.kpi{background:#141920;border:1px solid #222a35;border-radius:12px;padding:12px 14px}
.kpi .l{font-size:11px;color:#8d97a3;text-transform:uppercase;letter-spacing:.5px}
.kpi .v{font-size:25px;font-weight:750;margin:4px 0 2px}
.kpi .s{font-size:11px;color:#79838f}
.good{color:#54d98c}.bad{color:#ff7b72}.neutral{color:#e9edf2}
.main{display:grid;grid-template-columns:1fr 372px;gap:14px}
.card{background:#141920;border:1px solid #222a35;border-radius:12px;padding:14px 16px;margin-bottom:14px}
.card h3{margin:0 0 10px;font-size:13px;font-weight:650;color:#c6d0dc;letter-spacing:.3px}
.ctrl{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
button{background:#1d2530;color:#e9edf2;border:1px solid #2c3644;border-radius:8px;padding:8px 15px;
  font-size:13px;cursor:pointer;font-weight:600}
button:hover{background:#26303d}button:disabled{opacity:.4;cursor:default}
button.primary{background:#1f6b45;border-color:#2a8657}button.primary:hover{background:#25805292}
input[type=range]{flex:1;min-width:180px;accent-color:#3fa06a}
select{background:#1d2530;color:#e9edf2;border:1px solid #2c3644;border-radius:8px;padding:7px 9px;font-size:12.5px}
.clock{font-variant-numeric:tabular-nums;font-size:14px;font-weight:650;color:#cfe3d6;min-width:168px}
svg{width:100%;height:auto;display:block;overflow:visible}
.now{display:grid;grid-template-columns:1fr 1fr;gap:9px}
.now div{background:#111720;border:1px solid #1e2632;border-radius:9px;padding:8px 10px}
.now .l{font-size:10px;color:#8d97a3;text-transform:uppercase}
.now .v{font-size:16px;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums}
.feed{max-height:460px;overflow-y:auto;padding-right:4px}
.ev{border-left:3px solid #2c3644;padding:8px 10px;margin-bottom:8px;background:#111720;border-radius:0 8px 8px 0}
.ev.llm{border-left-color:#3fa06a}.ev.fallback{border-left-color:#c9913f}
.ev.reflect{border-left-color:#7b6ce0;background:#161329}
.ev .t{font-size:10.5px;color:#8d97a3;font-variant-numeric:tabular-nums;display:flex;justify-content:space-between}
.ev .strat{font-size:12.5px;font-weight:700;margin:3px 0;color:#dfe7ef}
.ev .r{font-size:11.5px;color:#9aa6b3;line-height:1.45}
.ev .note{font-size:10.5px;color:#6b7683;font-style:italic;margin-top:6px;
  padding-top:6px;border-top:1px solid #1c232e;line-height:1.35}
.tag{display:inline-block;font-size:9.5px;padding:2px 6px;border-radius:5px;background:#2a2036;
  color:#c9b6ea;border:1px solid #3d3054;margin-top:5px}
.tag.safety{background:#33240f;color:#e8c07a;border-color:#5a4116}
.legend{display:flex;gap:14px;font-size:11px;color:#8d97a3;margin-top:6px;flex-wrap:wrap}
.legend i{display:inline-block;width:11px;height:3px;border-radius:2px;margin-right:5px;vertical-align:middle}
.empty{color:#79838f;font-size:12px;padding:14px 2px}
@media(max-width:1080px){.main{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}}
</style></head><body><div class="wrap">

<header>
  <h1>Eco-Loop Building Agents</h1>
  <span class="badge" id="modeBadge">replay</span>
  <span class="badge" id="stepsBadge">—</span>
  <span class="badge ok" id="fbBadge">—</span>
  <span class="badge ok" id="chaosBadge" style="display:none">—</span>
</header>
<div class="sub">Autonomous LLM closed-loop control of a live EnergyPlus building — baseline BMS vs Eco-Loop AI on identical weather, occupancy and tariffs.</div>

<div class="kpis">
  <div class="kpi"><div class="l">Energy saved</div><div class="v good" id="kEnergy">—</div><div class="s" id="kEnergyS">—</div></div>
  <div class="kpi"><div class="l">Cost saved</div><div class="v good" id="kCost">—</div><div class="s" id="kCostS">—</div></div>
  <div class="kpi"><div class="l">Carbon saved</div><div class="v good" id="kCarbon">—</div><div class="s" id="kCarbonS">—</div></div>
  <div class="kpi"><div class="l">Peak demand</div><div class="v good" id="kPeak">—</div><div class="s" id="kPeakS">—</div></div>
  <div class="kpi"><div class="l">Occupied comfort</div><div class="v neutral" id="kComfort">—</div><div class="s" id="kComfortS">—</div></div>
</div>

<div class="main">
 <div>
  <div class="card">
    <div class="ctrl">
      <button class="primary" id="play">▶ Play</button>
      <button id="restart">⟲ Restart</button>
      <button id="followBtn" style="display:none">⏭ Follow live</button>
      <input type="range" id="scrub" min="0" max="0" value="0">
      <select id="speed">
        <option value="1">1×</option><option value="2">2×</option>
        <option value="4" selected>4×</option><option value="8">8×</option><option value="16">16×</option>
      </select>
      <span class="clock" id="clock">—</span>
    </div>
  </div>

  <div class="card">
    <h3>Cumulative electricity — the savings gap widening</h3>
    <svg id="energyChart" viewBox="0 0 900 260" preserveAspectRatio="none"></svg>
    <div class="legend">
      <span><i style="background:#c1554f"></i>Baseline BMS</span>
      <span><i style="background:#3fa06a"></i>Eco-Loop AI</span>
      <span><i style="background:#2f3a49"></i>energy avoided</span>
    </div>
  </div>

  <div class="card">
    <h3>Zone temperature vs comfort band</h3>
    <svg id="tempChart" viewBox="0 0 900 250" preserveAspectRatio="none"></svg>
    <div class="legend">
      <span><i style="background:#2a5c43"></i>occupied comfort band</span>
      <span><i style="background:#d7924a"></i>outdoor</span>
      <span><i style="background:#c1554f"></i>baseline zone</span>
      <span><i style="background:#3fa06a"></i>AI zone</span>
      <span><i style="background:#6fa8dc"></i>AI cooling setpoint</span>
    </div>
  </div>

  <div class="card">
    <h3>Where the savings come from — electricity by end use</h3>
    <svg id="enduseChart" viewBox="0 0 900 170" preserveAspectRatio="none"></svg>
    <div class="legend">
      <span><i style="background:#4d8fd6"></i>heating &amp; cooling</span>
      <span><i style="background:#e0b64a"></i>lighting</span>
      <span><i style="background:#9b7fd4"></i>equipment / plug loads</span>
    </div>
  </div>

  <div class="card">
    <h3>Occupant comfort (PMV) and indoor air quality (CO₂)</h3>
    <svg id="comfortChart" viewBox="0 0 900 230" preserveAspectRatio="none"></svg>
    <div class="legend">
      <span><i style="background:#2a5c43"></i>acceptable PMV band (±0.7)</span>
      <span><i style="background:#3fa06a"></i>AI comfort (PMV)</span>
      <span><i style="background:#c1554f"></i>baseline comfort (PMV)</span>
      <span><i style="background:#7fc4d8"></i>CO₂ level</span>
      <span><i style="background:#a05252"></i>CO₂ limit</span>
    </div>
  </div>
 </div>

 <div>
  <div class="card">
    <h3>Live building state</h3>
    <div class="now">
      <div><div class="l">Sim time</div><div class="v" id="nTime">—</div></div>
      <div><div class="l">Outdoor</div><div class="v" id="nOut">—</div></div>
      <div><div class="l">Mean zone</div><div class="v" id="nZone">—</div></div>
      <div><div class="l">Occupancy</div><div class="v" id="nOcc">—</div></div>
      <div><div class="l">Cooling SP</div><div class="v" id="nCool">—</div></div>
      <div><div class="l">Worst PMV</div><div class="v" id="nPmv">—</div></div>
      <div><div class="l">Energy used</div><div class="v" id="nKwh">—</div></div>
      <div><div class="l">vs baseline</div><div class="v good" id="nSave">—</div></div>
      <div><div class="l">Air quality</div><div class="v" id="nCo2">—</div></div>
      <div><div class="l">Grid carbon</div><div class="v" id="nCarbon">—</div></div>
      <div><div class="l">Lighting</div><div class="v" id="nLight">—</div></div>
      <div><div class="l">Power price</div><div class="v" id="nPrice">—</div></div>
    </div>
  </div>
  <div class="card">
    <h3>What the AI is doing, and why</h3>
    <div class="feed" id="feed"><div class="empty">Press play to watch the AI make its decisions.</div></div>
  </div>
 </div>
</div>
</div>

<script>
const LIVE = __LIVE__;
let DATA = __DATA__;
let idx = 0, playing = false, timer = null;
// In live mode the view sticks to the newest step — until you scrub or press play, at
// which point you take control and a "Follow live" button appears to hand it back.
let following = LIVE;

const $ = id => document.getElementById(id);

/* Plain-English translation. The agent thinks in ECM strategy names and the safety guard
   emits terse flags; neither means anything to a non-engineer watching a demo, so every
   card is rendered as a human sentence with the raw agent text kept as a footnote. */
const PLAY = {
  setpoint_reset: ["Running warm to save energy",
    "Nobody feels uncomfortable at this temperature, so the building sits at the warm edge of the comfort range instead of being over-cooled."],
  precool: ["Cooling early, before power gets expensive",
    "Electricity prices are about to spike, so the building is being chilled now while power is still cheap."],
  peak_coast: ["Coasting through the expensive hours",
    "Power is at its priciest right now, so the air conditioning eases off and the building coasts on the coolness already stored in its walls."],
  precondition: ["Getting comfortable before people arrive",
    "Staff arrive shortly, so the building is being brought to a pleasant temperature in time for them."],
  deep_setback: ["Building is empty — easing right off",
    "There is nobody here to keep comfortable, so cooling is dialled right back. This is where the biggest savings come from."]
};
const SAFETY_PLAIN = {
  occupied_comfort_cap: "Kept within the comfort limit for the people inside",
  comfort_override: "Overridden to bring the building back to a comfortable temperature",
  deadband_widened: "Adjusted so heating and cooling don't fight each other",
  nonfinite_setpoint_replaced: "An invalid value was caught and replaced with a safe one",
  iaq_override_economizer: "Extra fresh air brought in to keep the air quality healthy"
};
function humanise(reason, source){
  const sp = (reason||'').indexOf(':');
  const key = sp > 0 ? reason.slice(0, sp).trim() : '';
  const note = sp > 0 ? reason.slice(sp+1).trim() : (reason||'');
  if(PLAY[key]) return { title: PLAY[key][0], text: PLAY[key][1], note };
  return { title: source === 'fallback' ? 'Backup plan in control' : 'Decision',
           text: note || 'Adjusting the building for the hour ahead.', note: '' };
}
const fmtT = s => { if(!s) return '—'; const d = new Date(s);
  return d.toLocaleDateString('en',{weekday:'short'}) + ' ' +
    String(d.getHours()).padStart(2,'0') + ':' + String(d.getMinutes()).padStart(2,'0'); };

function n(){ return DATA ? (DATA.agent.time||[]).length : 0; }
function haveBase(){ return DATA && DATA.baseline && (DATA.baseline.time||[]).length > 0; }

/* ---------- charts (hand-rolled SVG, no libraries) ---------- */
function path(vals, x0,y0,w,h, lo,hi, upto){
  // Missing values (the pre-control `init` row) are gaps, not zeros — draw a fresh
  // sub-path after each one instead of plunging the line to the bottom of the chart.
  const N = Math.max(1, vals.length-1); let d='', pen=false;
  for(let i=0;i<=upto && i<vals.length;i++){
    const v = vals[i];
    if(v === null || v === undefined || !isFinite(v)){ pen=false; continue; }
    const x = x0 + w*i/N, y = y0+h - h*((v-lo)/((hi-lo)||1));
    d += (pen ? 'L':'M') + x.toFixed(1) + ' ' + y.toFixed(1);
    pen = true;
  }
  return d;
}
function gridY(x0,y0,w,h,lo,hi,unit){
  let g=''; for(let k=0;k<=4;k++){ const y=y0+h-h*k/4, v=lo+(hi-lo)*k/4;
    g += `<line x1="${x0}" y1="${y.toFixed(1)}" x2="${x0+w}" y2="${y.toFixed(1)}" stroke="#1c232e"/>`
       + `<text x="${x0-7}" y="${(y+3.5).toFixed(1)}" fill="#6d7783" font-size="10" text-anchor="end">${v.toFixed(0)}${unit}</text>`; }
  return g;
}
function dayLines(x0,y0,w,h){
  const t = DATA.agent.time, N=Math.max(1,t.length-1); let g='';
  for(let i=1;i<t.length;i++){
    if(new Date(t[i]).getDate() !== new Date(t[i-1]).getDate()){
      const x = x0 + w*i/N;
      g += `<line x1="${x.toFixed(1)}" y1="${y0}" x2="${x.toFixed(1)}" y2="${y0+h}" stroke="#232c38" stroke-dasharray="3 4"/>`
         + `<text x="${(x+4).toFixed(1)}" y="${y0+11}" fill="#5d6773" font-size="9.5">${new Date(t[i]).toLocaleDateString('en',{month:'short',day:'numeric'})}</text>`;
    }
  }
  return g;
}
function playhead(x0,y0,w,h,i){
  const N=Math.max(1,n()-1), x=x0+w*i/N;
  return `<line x1="${x.toFixed(1)}" y1="${y0}" x2="${x.toFixed(1)}" y2="${y0+h}" stroke="#7f8b99" stroke-width="1.2"/>`
       + `<circle cx="${x.toFixed(1)}" cy="${y0-4}" r="3.5" fill="#cfd8e3"/>`;
}

function drawEnergy(){
  const X=44,Y=14,W=840,H=210;
  const a = DATA.agent.cumulative_kwh||[], b = haveBase()? DATA.baseline.cumulative_kwh : [];
  const num = arr => arr.filter(v => v !== null && v !== undefined && isFinite(v));
  const hi = Math.max(1, Math.max(...num(a.slice(0,n())), ...num(b.length?b:[0]))) * 1.06;
  let s = gridY(X,Y,W,H,0,hi,'') + dayLines(X,Y,W,H);
  if(b.length){
    // shaded gap = energy the AI avoided
    const N=Math.max(1,a.length-1); let d='', started=false;
    const ok = v => v !== null && v !== undefined && isFinite(v);
    for(let i=0;i<=idx&&i<b.length;i++){ if(!ok(b[i])) continue;
      const x=X+W*i/N,y=Y+H-H*(b[i]/hi); d+=(started?'L':'M')+x.toFixed(1)+' '+y.toFixed(1); started=true; }
    for(let i=Math.min(idx,a.length-1);i>=0;i--){ if(!ok(a[i])) continue;
      const x=X+W*i/N,y=Y+H-H*(a[i]/hi); d+='L'+x.toFixed(1)+' '+y.toFixed(1); }
    if(started && idx>0) s += `<path d="${d}Z" fill="#2f3a49" opacity=".55"/>`;
    s += `<path d="${path(b,X,Y,W,H,0,hi,idx)}" fill="none" stroke="#c1554f" stroke-width="2"/>`;
  }
  s += `<path d="${path(a,X,Y,W,H,0,hi,idx)}" fill="none" stroke="#3fa06a" stroke-width="2.4"/>`;
  s += playhead(X,Y,W,H,idx);
  s += `<text x="${X}" y="${Y+H+20}" fill="#6d7783" font-size="10">kWh consumed since start</text>`;
  $('energyChart').innerHTML = s;
}

function drawTemp(){
  const X=44,Y=14,W=840,H=200;
  const z=DATA.agent.mean_zone_temp||[], o=DATA.agent.outdoor_temp||[],
        c=DATA.agent.cooling_setpoint||[], bz=haveBase()?DATA.baseline.mean_zone_temp:[];
  const all=[...z.slice(0,n()),...o.slice(0,n())].filter(v=>isFinite(v));
  const lo=Math.min(18,Math.floor(Math.min(...all)-1)), hi=Math.ceil(Math.max(...all)+1);
  let s = gridY(X,Y,W,H,lo,hi,'°') + dayLines(X,Y,W,H);
  const yOf=v=>Y+H-H*((v-lo)/((hi-lo)||1));
  s += `<rect x="${X}" y="${yOf(26).toFixed(1)}" width="${W}" height="${(yOf(22)-yOf(26)).toFixed(1)}" fill="#2a5c43" opacity=".22"/>`;
  s += `<path d="${path(o,X,Y,W,H,lo,hi,idx)}" fill="none" stroke="#d7924a" stroke-width="1.3" opacity=".85"/>`;
  if(bz.length) s += `<path d="${path(bz,X,Y,W,H,lo,hi,idx)}" fill="none" stroke="#c1554f" stroke-width="1.6" opacity=".9"/>`;
  s += `<path d="${path(c,X,Y,W,H,lo,hi,idx)}" fill="none" stroke="#6fa8dc" stroke-width="1.3" stroke-dasharray="4 3" opacity=".9"/>`;
  s += `<path d="${path(z,X,Y,W,H,lo,hi,idx)}" fill="none" stroke="#3fa06a" stroke-width="2.2"/>`;
  s += playhead(X,Y,W,H,idx);
  $('tempChart').innerHTML = s;
}

function drawEnduse(){
  // Stacked horizontal bars, baseline above AI, so the shrinking of each end use is
  // directly comparable. This is what proves the savings are whole-building, not just HVAC.
  const A=DATA.agent, B=DATA.baseline, i=Math.min(idx,n()-1);
  if(i<0 || A.cumulative_kwh_hvac==null){ $('enduseChart').innerHTML=''; return; }
  const get=(o,k)=> (o && o[k] && o[k][i]!=null) ? o[k][i] : 0;
  const rows=[
    {label:'Baseline BMS', v:[get(B,'cumulative_kwh_hvac'),get(B,'cumulative_kwh_lights'),get(B,'cumulative_kwh_plugs')]},
    {label:'Eco-Loop AI',  v:[get(A,'cumulative_kwh_hvac'),get(A,'cumulative_kwh_lights'),get(A,'cumulative_kwh_plugs')]}
  ];
  const total=r=>r.v.reduce((a,b)=>a+b,0);
  const max=Math.max(1,...rows.map(total))*1.02;
  const cols=['#4d8fd6','#e0b64a','#9b7fd4'], names=['HVAC','Lighting','Equipment'];
  const X=104,W=700; let s='',y=26;
  rows.forEach(r=>{
    if(!haveBase() && r.label.startsWith('Baseline')){ y+=64; return; }
    let x=X;
    s+=`<text x="${X-10}" y="${y+24}" fill="#c6d0dc" font-size="12" text-anchor="end">${r.label}</text>`;
    r.v.forEach((val,k)=>{
      const w=W*val/max;
      if(w>0.4){
        s+=`<rect x="${x.toFixed(1)}" y="${y}" width="${w.toFixed(1)}" height="34" fill="${cols[k]}" rx="2"/>`;
        if(w>52) s+=`<text x="${(x+w/2).toFixed(1)}" y="${y+22}" fill="#0e1116" font-size="11.5"
          font-weight="700" text-anchor="middle">${val.toFixed(0)}</text>`;
      }
      x+=w;
    });
    s+=`<text x="${(x+9).toFixed(1)}" y="${y+22}" fill="#e9edf2" font-size="12" font-weight="700">${total(r).toFixed(0)} kWh</text>`;
    y+=64;
  });
  if(haveBase()){
    const bt=total(rows[0]), at=total(rows[1]);
    const pct = bt>0 ? (bt-at)/bt*100 : 0;
    s+=`<text x="${X}" y="152" fill="#54d98c" font-size="12.5" font-weight="700">`
      +`${pct>=0?'−':'+'}${Math.abs(pct).toFixed(1)}% total electricity so far`
      +`<tspan fill="#79838f" font-weight="400"> — ${names.map((nm,k)=>{
          const b=rows[0].v[k], a=rows[1].v[k];
          return nm+' '+(b>0?((b-a)/b*100>=0?'−':'+')+Math.abs((b-a)/b*100).toFixed(0)+'%':'—');
        }).join(', ')}</tspan></text>`;
  }
  $('enduseChart').innerHTML=s;
}

function drawComfort(){
  // PMV and CO2 are both explicitly required feedback metrics, and they are the evidence
  // that the energy savings were not taken out of the occupants.
  const X=44,Y=14,W=800,H=150;
  const A=DATA.agent, B=DATA.baseline;
  const pmv=A.max_abs_pmv||[], co2=A.peak_co2||[], bpmv=(B&&B.max_abs_pmv)||[];
  if(!pmv.length){ $('comfortChart').innerHTML=''; return; }
  const numOf = arr => arr.filter(v => v !== null && v !== undefined && isFinite(v));
  const lo=-1.5, hi=1.5;
  const yOf=v=>Y+H-H*((v-lo)/(hi-lo));
  let s = gridY(X,Y,W,H,lo,hi,'') + dayLines(X,Y,W,H);
  s += `<rect x="${X}" y="${yOf(0.7).toFixed(1)}" width="${W}" height="${(yOf(-0.7)-yOf(0.7)).toFixed(1)}"
        fill="#2a5c43" opacity=".24"/>`;
  if(bpmv.length) s += `<path d="${path(bpmv,X,Y,W,H,lo,hi,idx)}" fill="none" stroke="#c1554f" stroke-width="1.5" opacity=".85"/>`;
  s += `<path d="${path(pmv,X,Y,W,H,lo,hi,idx)}" fill="none" stroke="#3fa06a" stroke-width="2.1"/>`;
  s += `<text x="${X}" y="${Y-2}" fill="#6d7783" font-size="10">PMV (0 = neutral, ±0.7 = acceptable)</text>`;
  // CO2 shares the panel on its own scale underneath.
  const cSeen = numOf(co2.slice(0, Math.max(1, idx+1)));
  const cY=Y+H+30, cH=44, cHi=Math.max(1200, ...(cSeen.length?cSeen:[1200]))*1.05;
  const limit = 1000;
  s += `<path d="${path(co2,X,cY,W,cH,400,cHi,idx)}" fill="none" stroke="#7fc4d8" stroke-width="1.6"/>`;
  const yLim = cY+cH-cH*((limit-400)/((cHi-400)||1));
  s += `<line x1="${X}" y1="${yLim.toFixed(1)}" x2="${X+W}" y2="${yLim.toFixed(1)}" stroke="#a05252" stroke-dasharray="4 3"/>`;
  s += `<text x="${X}" y="${cY-4}" fill="#6d7783" font-size="10">indoor CO₂ (ppm) — dashed line is the ${limit}ppm limit</text>`;
  s += playhead(X,Y,W,H,idx);
  $('comfortChart').innerHTML = s;
}

/* ---------- side panels ---------- */
function drawNow(){
  const A=DATA.agent, i=Math.min(idx, n()-1); if(i<0) return;
  $('nTime').textContent = fmtT(A.time[i]);
  $('nOut').textContent  = (A.outdoor_temp[i]??0).toFixed(1)+'°C';
  $('nZone').textContent = (A.mean_zone_temp[i]??0).toFixed(1)+'°C';
  const occ = (A.occupancy[i]??0);
  $('nOcc').textContent  = occ>0.1 ? Math.round(occ*100)+'%' : 'empty';
  const csp = A.cooling_setpoint[i];
  $('nCool').textContent = (csp==null) ? '—' : csp.toFixed(1)+'°C';   // blank before first action
  const pmv=(A.max_abs_pmv[i]??0);
  $('nPmv').textContent = (pmv>=0?'+':'')+pmv.toFixed(2);
  $('nPmv').className = 'v ' + (Math.abs(pmv)<=0.7?'good':'bad');
  const kwh=(A.cumulative_kwh[i]??0); $('nKwh').textContent = kwh.toFixed(1)+' kWh';
  if(haveBase() && DATA.baseline.cumulative_kwh[i]!=null){
    const bk=DATA.baseline.cumulative_kwh[i];
    const pct = bk>0 ? (bk-kwh)/bk*100 : 0;
    $('nSave').textContent = (pct>=0?'−':'+')+Math.abs(pct).toFixed(1)+'%';
    $('nSave').className = 'v ' + (pct>=0?'good':'bad');
  } else $('nSave').textContent='—';

  // Required feedback metrics (air quality, carbon, price) plus the new lighting lever.
  const co2=(A.peak_co2&&A.peak_co2[i]);
  if(co2!=null){ $('nCo2').textContent=Math.round(co2)+' ppm';
    $('nCo2').className='v '+(co2<=1000?'good':'bad'); }
  const carb=(A.carbon&&A.carbon[i]);
  if(carb!=null) $('nCarbon').textContent=Math.round(carb)+' g/kWh';
  const lt=(A.light_level&&A.light_level[i]);
  if(lt!=null) $('nLight').textContent=Math.round(lt*100)+'%';
  const pr=(A.price&&A.price[i]);
  if(pr!=null){
    const tier = pr>=0.35?'peak':(pr>=0.14?'mid':'off-peak');
    $('nPrice').textContent='$'+pr.toFixed(2)+' '+tier;
    $('nPrice').className='v '+(tier==='peak'?'bad':'neutral');
  }
}

function drawFeed(){
  const t = DATA.agent.time[Math.min(idx,n()-1)]; if(!t) return;
  const cut = new Date(t).getTime();
  const evs = [];
  (DATA.decisions||[]).forEach(d=>{ if(new Date(d.time).getTime()<=cut) evs.push({k:'d',ts:new Date(d.time).getTime(),d}); });
  (DATA.reflections||[]).forEach(r=>{ const ts=new Date(r.day+'T23:59:00').getTime();
    if(ts<=cut) evs.push({k:'r',ts,d:r}); });
  evs.sort((a,b)=>b.ts-a.ts);
  // Cap the decision stream, but never let it crowd out the nightly self-critiques —
  // they are rarer and more interesting, so they always stay in the feed.
  let shown = 0;
  const top = evs.filter(e => e.k === 'r' ? true : ++shown <= 34);
  if(!top.length){ $('feed').innerHTML='<div class="empty">No decisions yet at this point in the run.</div>'; return; }
  $('feed').innerHTML = top.map(e=>{
    if(e.k==='r'){
      const r=e.d, esc = s => (s||'').replace(/</g,'&lt;');
      const KNOB = {
        deep_setback_cool_c: "how far it lets an empty building warm up",
        deep_setback_heat_c: "how cool it lets an empty building get",
        setpoint_reset_margin_c: "how close to the warm comfort edge it runs",
        precool_margin_c: "how hard it pre-cools before expensive hours"
      };
      const what = KNOB[r.parameter] || r.parameter;
      const dir = r.applied_value > r.old_value ? "increased" : "reduced";
      const capped = r.was_clamped
        ? ` <span class="tag safety">🛡 it wanted ${r.requested_value}°C — capped at the safe limit</span>` : '';
      return `<div class="ev reflect"><div class="t"><span>🌙 End of ${r.day} — the agent reviewed its own day</span></div>
        <div class="strat">It ${dir} ${what}</div>
        <div class="r">Changed from ${r.old_value}°C to ${r.applied_value}°C for tomorrow.${capped}</div>
        <div class="note">agent’s own words: “${esc(r.justification)}”</div></div>`;
    }
    const d=e.d, h=humanise(d.reason, d.source);
    const esc = s => (s||'').replace(/</g,'&lt;');
    // When the guard corrected the AI, say so — otherwise the card contradicts itself
    // ("nobody is here" next to "kept comfortable for the people inside"). This is the
    // safety layer visibly catching a wrong call, which is worth showing, not hiding.
    const corrected = (d.safety||[]).some(s =>
      s === 'occupied_comfort_cap' || s === 'comfort_override');
    if(corrected){
      h.text = `The AI proposed “${h.title.toLowerCase()}”, but people were in the building — `
             + `so the safety system stepped in and held it at a comfortable `
             + `${d.cooling_setpoint}°C instead.`;
      h.title = 'Safety system corrected the AI';
    }
    const safety=(d.safety||[]).map(s=>
      `<span class="tag safety">🛡 ${esc(SAFETY_PLAIN[s] || s)}</span>`).join(' ');
    // The agent's own words are kept, but demoted to a footnote so the plain-English
    // summary leads. Judges still see the model genuinely reasoned, not a canned script.
    const note = h.note ? `<div class="note">agent’s own words: “${esc(h.note)}”</div>` : '';
    return `<div class="ev ${d.source}"><div class="t"><span>${fmtT(d.time)}</span>
      <span>${d.latency_s?'decided in '+d.latency_s+'s':''}</span></div>
      <div class="strat">${h.title}</div>
      <div class="r">${esc(h.text)}</div>
      <div>${d.cooling_setpoint!=null?`<span class="tag">cooling to ${d.cooling_setpoint}°C</span>`:''}
      ${d.economizer?'<span class="tag">using cool outdoor air</span>':''} ${safety}</div>
      ${note}</div>`;
  }).join('');
}

function drawKPIs(){
  // Always computed up to the playhead, never end-of-run totals. A replay that showed
  // the final -21.9% from frame one contradicted both the animating charts and the
  // "vs baseline" tile below, so the page displayed two different savings at once.
  const A=DATA.agent, B=DATA.baseline, st=DATA.stats||{}, sv=DATA.savings||{};
  const i=Math.min(idx, n()-1);
  const set=(id,v,s2,good)=>{ const e=$(id); e.textContent=v;
    e.className='v '+(good===undefined?'neutral':(good?'good':'bad')); $(id+'S').textContent=s2; };
  const val=(o,k)=> (o && o[k] && o[k][i]!=null) ? o[k][i] : null;
  const pct=(b,a)=> (b>0) ? (b-a)/b*100 : 0;
  const fin = (i >= n()-1);
  const tag = fin ? '' : ' so far';

  if(i>=0 && haveBase()){
    const pairs=[['kEnergy','cumulative_kwh','kWh',0],['kCost','cumulative_cost','',2],
                 ['kCarbon','cumulative_co2_kg','kg',0]];
    pairs.forEach(([id,key,unit,dp])=>{
      const b=val(B,key), a=val(A,key);
      if(b==null||a==null){ set(id,'—','—'); return; }
      const p=pct(b,a);
      const money = (id==='kCost');
      const fmt=v=> (money?'$':'')+v.toFixed(dp)+(unit?' '+unit:'');
      set(id,(p>=0?'−':'+')+Math.abs(p).toFixed(1)+'%', `${fmt(b)} → ${fmt(a)}${tag}`, p>=0);
    });
    // Peak demand: highest single-step draw seen so far, on both runs.
    const dtH=0.25, mx=arr=>{ const v=(arr||[]).slice(0,i+1)
        .filter(x=>x!==null&&x!==undefined&&isFinite(x)); return v.length?Math.max(...v):0; };
    const bp=mx(B.step_kwh)/dtH, ap=mx(A.step_kwh)/dtH;
    if(bp>0){ const p=pct(bp,ap);
      set('kPeak',(p>=0?'−':'+')+Math.abs(p).toFixed(1)+'%',
          `${bp.toFixed(1)} → ${ap.toFixed(1)} kW${tag}`, p>=0); }
    // Comfort: share of occupied zone-steps within limits, accumulated to here.
    // Zone-weighted, exactly as telemetry.py scores it: violating zone-steps over
    // occupied zone-steps. Counting a step as "bad" if any single zone slipped gave a
    // far harsher number that contradicted the run summary.
    const NZ = DATA.n_zones || 1;
    const rate=(o)=>{ let occ=0,bad=0;
      for(let k=0;k<=i;k++){ if((o.occupancy&&o.occupancy[k]||0)>0.1){ occ+=NZ;
        bad += (o.comfort_violations&&o.comfort_violations[k])||0; } }
      return occ? (1-bad/occ)*100 : null; };
    const ar=rate(A), br=rate(B);
    if(ar!=null){
      set('kComfort', ar.toFixed(1)+'%',
          (br!=null? `baseline ${br.toFixed(1)}%` : 'occupied comfort-OK')+tag,
          br==null || ar>=br-2);
    }
  } else if(i>=0){
    // No baseline to compare against (a bare live run): show absolute running totals.
    const ak=val(A,'cumulative_kwh')||0, ac=val(A,'cumulative_cost')||0,
          ag=val(A,'cumulative_co2_kg')||0;
    set('kEnergy', ak.toFixed(1)+' kWh', 'consumed'+tag);
    set('kCost', '$'+ac.toFixed(2), 'energy cost'+tag);
    set('kCarbon', ag.toFixed(1)+' kg', 'CO₂'+tag);
    const dtH=0.25, steps=(A.step_kwh||[]).slice(0,i+1)
      .filter(v=>v!==null&&v!==undefined&&isFinite(v));
    set('kPeak', (steps.length?Math.max(...steps)/dtH:0).toFixed(1)+' kW', 'peak demand'+tag);
    const NZ2 = DATA.n_zones || 1;
    let occ=0,bad=0;
    for(let k=0;k<=i;k++){ if((A.occupancy[k]||0)>0.1){ occ+=NZ2;
      bad += (A.comfort_violations[k]||0); } }
    const r2=occ?(1-bad/occ)*100:100;
    set('kComfort', r2.toFixed(1)+'%', `${Math.round(occ/NZ2)} occupied steps`+tag, r2>=90);
  }

  const dec = st.decisions||{};
  $('stepsBadge').textContent = `${i+1} / ${DATA.n_steps||n()} steps`;
  if(dec.fallback!=null){
    $('fbBadge').textContent = `${dec.fallback} fallbacks · ${(dec.llm||0)+(dec.cache||0)} agent decisions`;
    $('fbBadge').className = 'badge ' + (dec.fallback===0?'ok':'');
  } else $('fbBadge').style.display='none';
  const ch=DATA.chaos||{};
  if(ch.total){ const b=$('chaosBadge'); b.style.display=''; b.textContent=`chaos ${ch.passed}/${ch.total} faults survived`; }
  const mb=$('modeBadge');
  if(LIVE){ mb.className='badge live'; mb.innerHTML = DATA.running? '<span class="dot"></span>LIVE' : 'run finished'; }
  else mb.textContent='replay · '+(DATA.summary?.backend||'energyplus');
}

function render(){
  if(!DATA || !n()) return;
  idx = Math.max(0, Math.min(idx, n()-1));
  $('scrub').max = Math.max(0, n()-1); $('scrub').value = idx;
  $('clock').textContent = fmtT(DATA.agent.time[idx]);
  drawEnergy(); drawTemp(); drawEnduse(); drawComfort(); drawNow(); drawFeed(); drawKPIs();
}

/* ---------- playback ---------- */
function setFollow(on){
  following = on;
  // The button only makes sense while a live run is still producing data.
  const live = LIVE && DATA && DATA.running;
  $('followBtn').style.display = (live && !on) ? '' : 'none';
}
function step(){
  if(idx >= n()-1){
    // At the live edge with more data still coming, idle instead of stopping.
    if(LIVE && DATA.running) return;
    pause(); return;
  }
  idx++; render();
}
function play(){
  if(playing) return;
  // Pressing play while parked at the end should replay from the start, not sit there
  // doing nothing — which is what a finished live run always leaves you looking at.
  const atEnd = idx >= n()-1;
  const stillStreaming = LIVE && DATA && DATA.running;
  if(atEnd && !stillStreaming) idx = 0;
  playing=true; setFollow(false);
  $('play').textContent='❚❚ Pause';
  timer=setInterval(step, Math.max(16, 260/parseInt($('speed').value)));
}
function pause(){ playing=false; $('play').textContent='▶ Play'; clearInterval(timer); timer=null; }
$('play').onclick=()=> playing?pause():play();
$('restart').onclick=()=>{ pause(); setFollow(false); idx=0; render(); };
$('scrub').oninput=e=>{ pause(); setFollow(false); idx=parseInt(e.target.value); render(); };
$('speed').onchange=()=>{ if(playing){ pause(); play(); } };
$('followBtn').onclick=()=>{ setFollow(true); idx=Math.max(0,n()-1); render(); };

/* ---------- live polling ---------- */
async function poll(){
  let alive = true;
  try{
    const r = await fetch('/api/state', {cache:'no-store'});
    const s = await r.json();
    DATA = s;
    if(following) idx = Math.max(0, n()-1);   // stick to the live edge
    alive = !!s.running;
    if(!alive) setFollow(false);   // run over: it is a normal replay from here on
    render();
  }catch(e){ /* server gone: leave the last frame up rather than blanking the demo */ }
  // Once the run has finished the state stops changing, so stop hammering the endpoint.
  if(alive) setTimeout(poll, 1200);
}

if(LIVE){ setFollow(true); poll(); }
else { render(); }
</script></body></html>
"""
