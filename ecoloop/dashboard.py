"""Quantitative Savings Dashboard (deliverable #3).

Reads the baseline and AI timeseries + summaries and produces:
  * a multi-panel PNG (cumulative energy, comfort/temperatures, control behaviour, savings)
  * a self-contained HTML dashboard with headline KPIs and a methodology note

The headline it must defend: percentage reduction in total kWh **while maintaining
thermal comfort boundaries**.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def _pct(base: float, ai: float) -> float:
    return round((base - ai) / base * 100.0, 1) if base else 0.0


def _load(results_dir: Path, tag: str):
    df = pd.read_csv(results_dir / f"{tag}_timeseries.csv", parse_dates=["time"])
    with open(results_dir / f"{tag}_summary.json", encoding="utf-8") as fh:
        summ = json.load(fh)
    return df, summ


def compute_savings(sb: dict, sa: dict) -> dict[str, Any]:
    return {
        "baseline_kwh": sb["total_kwh"], "ai_kwh": sa["total_kwh"],
        "kwh_pct": _pct(sb["total_kwh"], sa["total_kwh"]),
        "baseline_cost": sb["total_cost_usd"], "ai_cost": sa["total_cost_usd"],
        "cost_pct": _pct(sb["total_cost_usd"], sa["total_cost_usd"]),
        "baseline_co2": sb["total_co2_kg"], "ai_co2": sa["total_co2_kg"],
        "co2_pct": _pct(sb["total_co2_kg"], sa["total_co2_kg"]),
        "baseline_peak_kw": sb["peak_demand_kw"], "ai_peak_kw": sa["peak_demand_kw"],
        "peak_pct": _pct(sb["peak_demand_kw"], sa["peak_demand_kw"]),
        "baseline_comfort_ok": sb["comfort_ok_rate"], "ai_comfort_ok": sa["comfort_ok_rate"],
        "baseline_mean_pmv": sb["mean_pmv_occupied"], "ai_mean_pmv": sa["mean_pmv_occupied"],
        "comfort_maintained": sa["comfort_ok_rate"] >= sb["comfort_ok_rate"] - 0.02,
    }


def _plot(cfg: dict, dfb, dfa, sb, sa, sv, png_path: Path) -> None:
    occ_lo = cfg["comfort"]["occ_low_c"]
    occ_hi = cfg["comfort"]["occ_high_c"]
    peak_hours = set(cfg["signals"]["peak_hours"])
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))

    # 1. Cumulative energy
    a = ax[0, 0]
    a.plot(dfb.time, dfb.cumulative_kwh, color="#b0413e", lw=2,
           label=f"Baseline BMS — {sb['total_kwh']:.0f} kWh")
    a.plot(dfa.time, dfa.cumulative_kwh, color="#2a7f62", lw=2,
           label=f"Eco-Loop AI — {sa['total_kwh']:.0f} kWh")
    a.fill_between(dfa.time, dfa.cumulative_kwh, dfb.cumulative_kwh,
                   color="#2a7f62", alpha=0.12)
    a.set_title(f"Cumulative electricity  →  {sv['kwh_pct']}% saved")
    a.set_ylabel("kWh"); a.legend(loc="upper left")

    # 2. Comfort / temperatures
    a = ax[0, 1]
    a.axhspan(occ_lo, occ_hi, color="#2a7f62", alpha=0.10, label="occupied comfort band")
    a.plot(dfa.time, dfa.outdoor_temp, color="#d08a3e", lw=1, alpha=0.7, label="outdoor")
    a.plot(dfb.time, dfb.mean_zone_temp, color="#b0413e", lw=1.2, label="baseline zone T")
    a.plot(dfa.time, dfa.mean_zone_temp, color="#2a7f62", lw=1.2, label="AI zone T")
    a.set_title(f"Zone temperature vs comfort band  (AI comfort OK {sa['comfort_ok_rate']*100:.1f}%)")
    a.set_ylabel("°C"); a.legend(loc="upper left", fontsize=7)

    # 3. Control behaviour vs price
    a = ax[1, 0]
    a.plot(dfb.time, dfb.cooling_setpoint, color="#b0413e", lw=1.2, drawstyle="steps-post",
           label="baseline cooling SP")
    a.plot(dfa.time, dfa.cooling_setpoint, color="#2a7f62", lw=1.6, drawstyle="steps-post",
           label="AI cooling SP")
    for _, r in dfa.iterrows():
        if r.time.hour in peak_hours:
            a.axvspan(r.time, r.time + pd.Timedelta(minutes=cfg["run"]["timestep_minutes"]),
                      color="#c62828", alpha=0.05)
    a.set_title("AI setpoint strategy (red = price peak → coast / pre-cool)")
    a.set_ylabel("cooling setpoint °C"); a.legend(loc="upper left", fontsize=7)

    # 4. Savings bars
    a = ax[1, 1]
    labels = ["Energy\nkWh", "Cost\n$", "Carbon\nkg", "Peak\nkW"]
    pcts = [sv["kwh_pct"], sv["cost_pct"], sv["co2_pct"], sv["peak_pct"]]
    colors = ["#2a7f62" if p >= 0 else "#b0413e" for p in pcts]
    bars = a.bar(labels, pcts, color=colors)
    a.axhline(0, color="#333", lw=0.8)
    a.set_title("Reduction vs baseline (%)"); a.set_ylabel("% reduction")
    for b, p in zip(bars, pcts):
        a.text(b.get_x() + b.get_width() / 2, p + (0.4 if p >= 0 else -1.2),
               f"{p:+.1f}%", ha="center", fontweight="bold")

    for row in ax:
        for a in row:
            for lbl in a.get_xticklabels():
                lbl.set_rotation(0)
                lbl.set_fontsize(7)
    fig.suptitle("Eco-Loop Building Agents — Baseline vs Autonomous AI Closed-Loop",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def _html(cfg, sb, sa, sv, agent_stats, png_b64, html_path: Path) -> None:
    def card(title, value, sub, good=True):
        color = "#2a7f62" if good else "#b0413e"
        return (f'<div class="card"><div class="t">{title}</div>'
                f'<div class="v" style="color:{color}">{value}</div>'
                f'<div class="s">{sub}</div></div>')

    comfort_good = sv["comfort_maintained"]
    stats_rows = ""
    if agent_stats:
        for k, v in agent_stats.items():
            stats_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Eco-Loop Savings Dashboard</title><style>
body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#0f1115;color:#e8e8e8}}
.wrap{{max-width:1080px;margin:0 auto;padding:28px}}
h1{{font-size:22px;margin:0 0 4px}} .muted{{color:#9aa0a6;font-size:13px;margin-bottom:20px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px}}
.card{{background:#181b21;border:1px solid #262b33;border-radius:12px;padding:14px}}
.card .t{{font-size:12px;color:#9aa0a6}} .card .v{{font-size:26px;font-weight:700;margin:4px 0}}
.card .s{{font-size:11px;color:#7d838b}}
img{{width:100%;border-radius:12px;border:1px solid #262b33;background:#fff}}
table{{width:100%;border-collapse:collapse;margin-top:18px;font-size:13px}}
td,th{{border:1px solid #262b33;padding:7px 10px;text-align:left}} th{{background:#181b21}}
.banner{{padding:12px 16px;border-radius:10px;margin:16px 0;font-weight:600}}
.ok{{background:#12331f;border:1px solid #1f6b3f;color:#7fe0a0}}
.bad{{background:#33161a;border:1px solid #7a2a30;color:#f0a0a6}}
</style></head><body><div class="wrap">
<h1>Eco-Loop Building Agents — Quantitative Savings Dashboard</h1>
<div class="muted">Backend: <b>{sa['backend']}</b> &nbsp;|&nbsp; Horizon: {sa['steps']} steps
&nbsp;|&nbsp; Autonomous LLM closed-loop vs conventional scheduled BMS</div>
<div class="cards">
{card("Energy saved", f"{sv['kwh_pct']:+.1f}%", f"{sv['baseline_kwh']:.0f} → {sv['ai_kwh']:.0f} kWh", sv['kwh_pct']>=0)}
{card("Cost saved", f"{sv['cost_pct']:+.1f}%", f"${sv['baseline_cost']:.0f} → ${sv['ai_cost']:.0f}", sv['cost_pct']>=0)}
{card("Carbon saved", f"{sv['co2_pct']:+.1f}%", f"{sv['baseline_co2']:.0f} → {sv['ai_co2']:.0f} kg", sv['co2_pct']>=0)}
{card("Peak demand", f"{sv['peak_pct']:+.1f}%", f"{sv['baseline_peak_kw']:.1f} → {sv['ai_peak_kw']:.1f} kW", sv['peak_pct']>=0)}
</div>
<div class="banner {'ok' if comfort_good else 'bad'}">
{'✓ Thermal comfort maintained' if comfort_good else '✗ Comfort degraded — review'}:
occupied comfort-OK rate {sv['baseline_comfort_ok']*100:.1f}% (baseline) →
{sv['ai_comfort_ok']*100:.1f}% (AI); mean PMV {sv['ai_mean_pmv']:+.2f}.
</div>
<img src="data:image/png;base64,{png_b64}"/>
<table><tr><th>Metric</th><th>Baseline BMS</th><th>Eco-Loop AI</th></tr>
<tr><td>Total energy (kWh)</td><td>{sb['total_kwh']}</td><td>{sa['total_kwh']}</td></tr>
<tr><td>Total cost ($)</td><td>{sb['total_cost_usd']}</td><td>{sa['total_cost_usd']}</td></tr>
<tr><td>Total carbon (kg CO₂)</td><td>{sb['total_co2_kg']}</td><td>{sa['total_co2_kg']}</td></tr>
<tr><td>Peak demand (kW)</td><td>{sb['peak_demand_kw']}</td><td>{sa['peak_demand_kw']}</td></tr>
<tr><td>Comfort-OK rate (occupied)</td><td>{sb['comfort_ok_rate']*100:.1f}%</td><td>{sa['comfort_ok_rate']*100:.1f}%</td></tr>
<tr><td>Mean PMV (occupied)</td><td>{sb['mean_pmv_occupied']}</td><td>{sa['mean_pmv_occupied']}</td></tr>
</table>
{('<h3>Agent runtime</h3><table><tr><th>Stat</th><th>Value</th></tr>' + stats_rows + '</table>') if agent_stats else ''}
<p class="muted" style="margin-top:18px">Comfort is scored over occupied zone-steps using
Fanger PMV (ISO 7730) with a CO₂ cap. Savings compare identical weather/occupancy/tariff
inputs; only the control policy differs.</p>
</div></body></html>"""
    html_path.write_text(html, encoding="utf-8")


def make_dashboard(cfg: dict, results_dir: str | Path,
                   baseline_tag: str = "baseline", agent_tag: str = "agent",
                   agent_stats: dict | None = None,
                   out_prefix: str = "dashboard") -> dict[str, Any]:
    results_dir = Path(results_dir)
    dfb, sb = _load(results_dir, baseline_tag)
    dfa, sa = _load(results_dir, agent_tag)
    if len(dfa) != len(dfb):
        # Headline numbers (sv, below) still come from each run's own full summary, so
        # they stay accurate; only the side-by-side timeseries plot needs equal length.
        n = min(len(dfa), len(dfb))
        print(f"[dashboard] NOTE: baseline ({len(dfb)} rows) and agent ({len(dfa)} rows) "
              f"timeseries differ in length — plotting the first {n} rows of each.")
        dfa = dfa.iloc[:n].reset_index(drop=True)
        dfb = dfb.iloc[:n].reset_index(drop=True)
    sv = compute_savings(sb, sa)
    png_path = results_dir / f"{out_prefix}.png"
    html_path = results_dir / f"{out_prefix}.html"
    _plot(cfg, dfb, dfa, sb, sa, sv, png_path)
    png_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    _html(cfg, sb, sa, sv, agent_stats, png_b64, html_path)
    with open(results_dir / f"{out_prefix}_savings.json", "w", encoding="utf-8") as fh:
        json.dump(sv, fh, indent=2)
    return {"png": str(png_path), "html": str(html_path), "savings": sv}
