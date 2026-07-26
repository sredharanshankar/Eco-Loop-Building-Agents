# Eco-Loop Building Agents — Master Build & Enhancement Plan

> **How to use this document.** Paste it back to a capable coding agent as the brief for the
> next iteration. It states the mission, the exact scoring rubric, the current architecture
> (so the agent doesn't re-derive it), and a prioritized roadmap of enhancements that make the
> solution *more sophisticated* while directly moving each evaluation number. Work top-down:
> the highest-leverage, criterion-aligned items come first.

---

## 1. Mission (Honeywell "Eco-Loop Building Agents")

Build a **live, operational closed-loop control pipeline** where an **open-source LLM** (via
**MCP / agentic tool-calling**) autonomously controls an **EnergyPlus** building simulation in
real time: stream sensor feedback → LLM reasons → inject setpoints/ECMs back into EnergyPlus →
prove **quantifiable kWh savings while holding thermal comfort**.

## 2. Scoring rubric — optimize explicitly for these weights

| # | Criterion | Weight | What actually moves the score | Where we attack it |
|---|-----------|:-----:|-------------------------------|--------------------|
| 1 | **System Integration** | **30%** | Loop runs a long horizon **without crashing**; robust to bad LLM output, timeouts, sim errors | Safety guard + heuristic fallback + threaded EP bridge; §5.1 |
| 2 | **Energy Efficiency Realized** | **25%** | **Net kWh reduction** vs baseline | Setpoint reset, deep setback, MPC pre-cooling; §5.2 |
| 3 | **Thermal Comfort & Constraints** | **20%** | Saved energy **without** hurting comfort (PMV/CO₂) | PMV-scored comfort, proactive comfort cap; §5.3 |
| 4 | **Agentic Autonomy & Code Elegance** | **15%** | Real tool-calling, MCP, **self-correction** loops | MCP server, strategy tools, error-triage self-heal; §5.4 |
| 5 | **Presentation & Documentation** | **10%** | Clear architecture, data viz, delivery | Dashboard, arch doc, deck, 3-min video; §5.5 |

**Rule of thumb:** never trade comfort for energy (criterion 3 gates criterion 2), and never let
the loop crash (criterion 1 dominates). A robust 12–18% saving with 100% comfort beats a fragile
30% saving that overheats occupants or throws.

## 3. Current architecture (already built — extend, don't rewrite)

```
run.py  (CLI: baseline | agent | compare | dashboard | demo)
mcp_server.py  (dependency-free MCP stdio server: parse_building_model, summarize_simulation_log,
                evaluate_comfort, grid_signals, run_closed_loop)
ecoloop/
  config.py         one merged config dict (comfort bands, tariff, model, cadence, EP paths)
  weather.py        synthetic exogenous drivers for the RC backend (deterministic summer week)
  signals.py        time-of-use price + marginal grid carbon
  comfort.py        Fanger PMV/PPD (ISO 7730) + comfort bands
  backends/
    base.py               SimBackend ABC + Observation/Action  (backend-agnostic contract)
    rc_backend.py         fast physics RC twin (dev/CI/fallback): multi-zone lumped-capacitance,
                          ideal-loads HVAC, economizer free-cooling, CO2 mass balance
    energyplus_backend.py REAL EnergyPlus via pyenergyplus, PUSH→PULL threaded bridge; actuates
                          "Zone Temperature Control" setpoints, reads ideal-loads energy → kWh
    idf_tools.py          parse zones/units, retarget run period, EPW forecast, .err triage
  agent/
    llm_client.py    OpenAI-compatible (Ollama) client; timeout/retry; model kept resident
    tools.py         MCP-equivalent ToolRegistry; terminal set_control takes an ECM STRATEGY enum
    prompts.py       compact system prompt + situational brief + log summarizer
    cache.py         decision cache (hour/occupancy/tier/outdoor band) — latency management
    supervisor.py    sense→reason→act loop, single-call fast path, heuristic fallback, self-correct
  control/
    baseline.py      rigid scheduled BMS (the incumbent to beat)
    safety.py        hard envelope + proactive occupied comfort cap + reactive recovery
    loop.py          the closed-loop runner (supervisory cadence: decide hourly, hold between)
  telemetry.py       per-step CSV + run summary JSON (kWh, cost, carbon, peak, comfort rate)
  dashboard.py       matplotlib PNG + self-contained HTML savings dashboard
```

**Key working facts (verified):** EnergyPlus v26.1.0 Python API loads on Python 3.14; the
ideal-loads 5-zone model responds to setpoint actuation (cool 22→26 °C cuts cooling ~44%);
qwen2.5:3b tool-calls reliably when the model is **pinned in RAM** (`keep_alive:-1`; otherwise a
40 s reload dominates on a 7.7 GB machine); the same controller drives both backends unchanged.

## 4. Deliverables checklist (map to the submission)

- [x] Unified source (EP API wrapper + agent orchestration + comms bus)
- [x] Building models: `models/baseline.idf` + runtime-generated `results/ep/in.idf`
- [x] Quantitative savings dashboard (`results/dashboard.html`, proves % kWh + comfort)
- [x] System Architecture Document (`docs/ARCHITECTURE.md`)
- [ ] 3-min PoC video (record `python run.py demo --backend energyplus`)
- [ ] Presentation (fill the provided template from `presentation/outline.md`)

---

## 5. Enhancement roadmap (do these to become "more sophisticated")

### 5.1 System Integration (30%) — make it bulletproof over long horizons
1. **Watchdog + structured recovery.** Wrap each step; on any EP/LLM/exception, log, fall back to
   the last safe action, and continue. Add a per-run `integrity_report` (steps, exceptions caught,
   fallback count, LLM timeout count) to the summary — evidence of robustness.
2. **Annual / multi-week horizon runs.** Prove it survives a full cooling season, not 3 days. Add
   `--weeks` and a headless CI job (`compare --no-llm`) that asserts no crash + savings > 0.
3. **Self-healing from EnergyPlus errors.** On a fatal `.err`, feed `summarize_simulation_log` to
   the LLM, have it propose an IDF fix via a `patch_idf` tool, re-run — a true "execute tasks
   without human code modification" loop. (Hooks already exist: `idf_tools`, `summarize_err`.)
4. **Async LLM (non-blocking).** Run inference in a worker thread; the sim advances on the last
   action until the new decision is ready. Removes the LLM from the critical path entirely.

### 5.2 Energy Efficiency (25%) — deeper, model-based savings
1. **LLM-supervised MPC.** Let the LLM set a *policy* (comfort band + carbon/price aggressiveness);
   a short receding-horizon optimizer (use the RC twin as the predictive model) computes the
   optimal setpoint trajectory over the next N hours. LLM = strategy, MPC = numbers. Big, robust
   savings and a strong "agentic + optimal" story.
2. **Thermal-mass pre-cooling that nets kWh-down**, not just load-shift: exploit high COP at cooler
   morning outdoor temps + genuine economizer hours (choose an EPW/climate with cool nights, e.g.
   swap Tampa for a high-diurnal-swing site so night flush pays back).
3. **Richer ECM set:** supply-air-temperature reset, demand-controlled ventilation (tie to the CO₂
   model), lighting/plug-load shed, optimal start/stop. Each becomes a new `strategy` enum value +
   actuator.
4. **Report end-use breakdown** (cooling/fans/pumps) and savings per ECM so the 25% is defensible.

### 5.3 Thermal Comfort (20%) — prove balance, not sacrifice
1. **Adaptive comfort (ASHRAE 55 adaptive / EN 16798)** in addition to Fanger PMV — widens the
   comfort band with outdoor temperature, unlocking more savings *legitimately*.
2. **Per-zone comfort** (each zone its own PMV target, occupancy, CO₂) instead of building-wide.
3. **Comfort-constrained objective:** expose a hard PMV band to the MPC; plot the PMV distribution
   (violation histogram) baseline vs AI to show the AI is *inside* the band more often.
4. Enable EnergyPlus native CO₂ (`ZoneAirContaminantBalance`) and Fanger comfort outputs instead of
   the current nominal CO₂ estimate.

### 5.4 Agentic Autonomy & Elegance (15%)
1. **Multi-agent roles:** a *Perception* agent (summarizes telemetry), a *Strategy* agent (picks
   ECMs), a *Safety/Critic* agent (vetoes unsafe actions) — orchestrated over MCP. Shows creative
   tool-calling and self-correction.
2. **Tool-use transparency:** log every tool call + argument + result to a trace viewer; surface
   "the agent called grid_forecast, saw a peak, chose peak_coast."
3. **Model-agnostic proof:** run the same loop on 2–3 OSS models (qwen2.5, llama3.2, mistral) and
   report a small ablation (savings, latency, tool-call validity rate).
4. **Reflection loop:** at day boundaries, the agent reviews yesterday's kWh + comfort and adjusts
   its own policy parameters (stored in config) — visible self-improvement.

### 5.5 Presentation & Documentation (10%)
1. Interactive dashboard (Plotly/HTML) with a time slider showing live state → decision → injection.
2. Architecture diagram (the feedback loop) as an SVG in the deck and README.
3. Record the 3-min video from `demo` mode (verbose loop prints each LLM decision + injection).
4. One-command reproducibility (`python run.py compare --backend energyplus`) documented up top.

---

## 6. Concrete next-iteration task list (ordered)

1. Add `--weeks`, watchdog, and `integrity_report`; run a 4-week EP season headless and confirm no
   crash + positive savings. *(System Integration)*
2. Implement `patch_idf` tool + the EnergyPlus-error self-healing loop with a deliberately broken
   IDF as a demo. *(Autonomy + Integration)*
3. Add the RC-twin receding-horizon MPC under LLM policy control; compare kWh vs the current
   heuristic/LLM. *(Energy)*
4. Add adaptive comfort + per-zone PMV; regenerate the dashboard with a PMV-distribution panel.
   *(Comfort)*
5. Switch to a high-diurnal EPW; enable native EnergyPlus CO₂ + Fanger outputs. *(Energy + Comfort)*
6. Add the Perception/Strategy/Critic multi-agent orchestration over MCP + a tool-trace viewer.
   *(Autonomy)*
7. Model ablation across 2–3 OSS LLMs; table of savings/latency/validity. *(Autonomy)*
8. Upgrade the dashboard to interactive HTML; export the architecture SVG; script the demo video.
   *(Presentation)*

## 7. Definition of done / guardrails

- The closed loop completes a **multi-week EnergyPlus** horizon with **zero unhandled exceptions**.
- Reported **kWh reduction ≥ 12%** with occupied **comfort rate ≥ baseline** (PMV within band).
- Every headline number is **reproducible** from one command and backed by the CSVs in `results/`.
- The LLM genuinely drives decisions (report `llm_driven_fraction`), and the loop **never stalls**
  when the LLM is slow/unavailable (fallback proven by killing Ollama mid-run).
- No fabricated numbers: comfort is scored from simulated PMV; savings compare identical
  weather/occupancy/tariff with only the control policy differing.
