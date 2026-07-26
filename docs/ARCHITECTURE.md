# Eco-Loop Building Agents — System Architecture

Physical-AI proof-of-concept: an open-source LLM autonomously runs a live EnergyPlus building in
a **closed control loop**, streaming sensor feedback, reasoning about it, and injecting control
actions back into the running simulation to cut energy while holding thermal comfort.

## 1. The closed loop

```
        ┌──────────────────────── FEEDBACK (EnergyPlus → AI) ─────────────────────────┐
        │  zone temps · outdoor temp · HVAC energy · PMV comfort · CO₂ · price/carbon   │
        ▼                                                                               │
  ┌───────────┐   Observation    ┌──────────────┐  strategy   ┌───────────────┐        │
  │ EnergyPlus │ ───────────────► │  LLM Agent    │ ─────────► │ Safety Guard   │        │
  │ (or RC twin)│                 │ (Ollama, MCP  │  set_control│ (hard envelope)│        │
  │  digital    │ ◄─────────────── │  tool-calling)│ ◄────────── │ + comfort cap │        │
  └───────────┘  setpoints (act)  └──────────────┘             └───────────────┘        │
        │            FORWARD INJECTION (AI → EnergyPlus): Zone Temperature Control        │
        └────────────────────────────── every timestep ─────────────────────────────────┘
```

- **Feedback:** each EnergyPlus zone-timestep streams zone air temperatures, outdoor conditions,
  ideal-loads HVAC energy, and (computed) PMV/CO₂.
- **Reasoning:** the LLM evaluates the state and 6-hour weather/grid forecasts against comfort
  targets, peak-demand and carbon signals.
- **Control:** the LLM selects an **Energy Conservation Measure** (ECM) via the `set_control` tool.
- **Forward injection:** the chosen setpoints are written to EnergyPlus's `Zone Temperature
  Control` actuators *before the zone heat balance*, so they drive the very next timestep.

## 2. Backend abstraction — one loop, two engines

Everything above the `SimBackend` interface (agent, safety, telemetry, dashboard) is
engine-agnostic. Two backends implement it:

| Backend | Purpose | Fidelity | Speed |
|---------|---------|----------|-------|
| `energyplus_backend` | **The real deliverable** — pyenergyplus runtime API | High (EnergyPlus v26) | ~1–2 s / sim-day |
| `rc_backend` | dev / CI / graceful fallback | Physics RC twin | ~ms / sim-day |

Switching is a single flag: `--backend energyplus`. The RC twin also serves as the *predictive
model* for future MPC work and lets us iterate on the agent without paying EnergyPlus runtime.

### 2.1 EnergyPlus integration (push → pull bridge)
EnergyPlus is **push-driven** (it calls a registered callback each timestep); our controller is
**pull-driven** (`reset`/`step`). We bridge them with a **background EnergyPlus thread and two
one-slot queues**, turning EnergyPlus into a gym-style environment. The callback (registered on
`begin_zone_timestep_before_init_heat_balance`) reads sensors, publishes an `Observation`, blocks
for the controller's `Action`, and writes the setpoint actuators — a clean per-timestep handshake.
Warm-up and sizing timesteps are gated out via `api_data_fully_ready` + `warmup_flag`. HVAC energy
comes from the ideal-loads *Total Cooling/Heating Energy* variables converted to electricity via
heat-pump COPs.

## 3. Tool-calling architecture (the MCP surface)

The agent senses and acts **only** through tools — the standardized, inspectable shape MCP
defines. The same tool registry is exposed two ways:

- **`mcp_server.py`** — a real, dependency-free **MCP stdio server** (JSON-RPC 2.0: `initialize`,
  `tools/list`, `tools/call`) exposing `parse_building_model`, `summarize_simulation_log`,
  `evaluate_comfort`, `grid_signals`, and `run_closed_loop`. Any MCP client (Claude Desktop, an
  IDE, our agent) can drive the building.
- **In-process `ToolRegistry`** — the identical tools called directly inside the hot control loop
  to avoid stdio latency.

**Terminal action tool — `set_control(strategy, economizer, reason)`.** Instead of asking a small
model for raw setpoint numbers (error-prone), it selects **one ECM strategy** from an enum —
`setpoint_reset · precool · peak_coast · precondition · deep_setback` — which a deterministic map
turns into safe, valid setpoints. Choosing a labelled measure is reliable for a 3B model, keeps
outputs short (fast), and guarantees the numbers are always in-range.

Sensing/look-ahead tools (`get_comfort_status`, `get_weather_forecast`, `get_grid_forecast`) and a
what-if tool (`evaluate_setpoint`) are used to build the decision context and are available over
MCP for interactive/advanced use.

## 4. Prompt engineering

- **Role + policy system prompt** framing the agent as a supervisory controller with five ECMs and
  crisp selection rules ("if unoccupied and empty >2 h you MUST deep-setback").
- **Front-loaded, compact context.** One user turn carries a one-line state summary, a 3-hour grid
  outlook, and a **derived FACTS line** (`occupied_now`, `occupancy_within_2h`, `price_tier_now`,
  `peak_within_2h`, `free_cooling_possible`). Pre-computing these booleans (from the forecast tools)
  turns the LLM's job into a reliable classification rather than fragile arithmetic, and keeps the
  prompt small.
- **Structured tool output** (a single JSON tool call) instead of free-form text — parsed
  robustly, with a lenient fallback parser for small-model quirks.
- **Determinism:** low temperature, capped `max_tokens`, and a bounded token budget.

## 5. Prompt-latency management (critical on modest hardware)

An OSS 3B model on a 7.7 GB CPU-only laptop is the slowest link. Four compounding techniques keep
the loop practical:

1. **Supervisory cadence.** The LLM decides once per **hour**; the chosen setpoints are held while
   the physics advance every 15 min. The slow brain sits above a fast local controller — exactly
   how real supervisory MPC layers work.
2. **Decision cache.** Situations are bucketed by `(hour, occupancy, price tier, outdoor band)`.
   A repeated bucket **reuses** the prior decision with zero inference, collapsing a multi-day
   horizon to a few dozen model calls (`llm_driven_fraction`, `hit_rate` are reported).
3. **Single-call fast path.** Context is front-loaded so the model commits in **one** tool call
   (short strategy output), not a multi-turn chain.
4. **Resident model.** The model is pinned in RAM (`keep_alive:-1`); without this, RAM pressure
   evicts it and each call pays a ~40 s reload. Warm calls are ~1–15 s.

If a call still times out or errors, the loop **never stalls** — see §6.

## 6. Robustness & self-correction (System Integration)

- **Deterministic safety guard** clamps every action to a hard envelope, enforces a minimum
  dead-band, and applies a **proactive occupied comfort cap** (cooling setpoint may never exceed
  the comfort ceiling while occupied) plus a **reactive recovery** if occupants are already
  uncomfortable. The LLM optimizes; the guard guarantees safety — so a hallucinated or wrong
  strategy can never overheat the building or crash the run.
- **Heuristic fallback controller** (itself a competent ECM policy) takes over on any LLM
  timeout/error, so the closed loop completes the full horizon regardless of the model.
- **Self-correction:** invalid tool arguments are returned to the model as a tool error for a
  bounded retry; EnergyPlus `.err` logs are summarized and can be fed back for error triage.
- **Process isolation.** EnergyPlus runs in a dedicated child process (§2.1), so even a hard
  native crash of the simulation engine cannot corrupt or kill the controller/agent process.

### 6.1 Fault-injection ("chaos") harness — robustness we can *demonstrate*

`chaos_test.py` deliberately breaks the loop and asserts it survives, so System Integration is
evidence rather than assertion. Every scenario models a failure that genuinely occurs (several
actually occurred during this project's development):

| Scenario | Injected fault | Result |
|---|---|---|
| `llm_unreachable` | Ollama endpoint down for the entire run | loop completes on the deterministic fallback; comfort-OK **100%** |
| `llm_timeout_storm` | every LLM call times out | loop completes on fallback; comfort-OK **100%** |
| `malformed_tool_args` | junk / missing / wrong-typed tool arguments (6 cases) | all 6 returned as structured tool errors, **0 exceptions** |
| `adversarial_actions` | controller emits NaN, ±inf, inverted, absurd, `None` setpoints every step | **0 envelope violations**; guard clamped every step |
| `policy_fuzz` | 400 adversarial self-critique proposals (NaN/inf/±1e9/strings) | **0 bound escapes, 0 exceptions** |
| `ep_worker_killed` | EnergyPlus worker process **hard-killed** mid-run (step 40) | controller survived; loop ended cleanly via `StopIteration` |

Run it with `python chaos_test.py` (fast, RC backend) or `python chaos_test.py --with-ep` to
include the EnergyPlus worker-kill. Results are written to `results/chaos_report.json`.

The harness paid for itself immediately: it found a **real crash bug** — `SafetyGuard.enforce()`
raised `TypeError` on a `None` setpoint and would have silently propagated NaN/inf (since
`min`/`max` don't filter them). A malformed LLM response could have triggered that in production.
The guard now sanitizes every value to a finite float (falling back to a mid-band setpoint and
flagging `nonfinite_setpoint_replaced`) before clamping.

### 6.2 Nightly bounded self-critique (the agent tunes its own policy)

At each simulated **day boundary** the loop runner (`control/loop.py::_DayTracker`) closes out the
day's energy and comfort and calls `AgentSupervisor.on_new_day()`. The agent is then shown its own
result — kWh used, occupied comfort-OK rate, violation count, mean PMV — plus its current policy,
and may adjust **at most one** parameter for tomorrow via the `propose_policy_tweak` tool:

```
DAY 2026-07-21 RESULT: energy=172.8kWh, occupied comfort-OK=100.0% (0 violations of 245), mean PMV=-0.13
CURRENT POLICY: setpoint_reset_margin_c=0.00, precool_margin_c=1.50, deep_setback_cool_c=28.00, ...
```

This is genuine closed-loop self-improvement, but it is **bounded by construction**:

- Only four named knobs are adjustable (`agent/policy.py`), each with a hard `[lo, hi]` range;
  a proposal outside the range is **clamped, never rejected or applied unbounded** (a request of
  `999.0` becomes `30.0` and is logged with `was_clamped: true`).
- The knobs only change *which setpoints a strategy proposes*. The **per-step `SafetyGuard`
  remains the final authority** on every actuated setpoint, so no tweak can breach comfort.
- A failed or malformed reflection is swallowed and logged — self-critique can never stop control.
- Accepting a tweak invalidates the decision cache, since cached decisions are stale under a
  changed policy.

Every tweak (requested value, applied value, bounds, whether it was clamped, and the agent's own
justification) is written to `results/agent_reflections.json`; the count and final policy appear in
`agent_runstats.json`. Disable with `agent.reflection: false` in config.

**Observed on the 7-day run.** The agent escalated its unoccupied setback ceiling across successive
nights — requesting **29.5 → 30.5 → 32.5 °C** — and the bound held it at **30.0 °C** every time
after the first. This is the mechanism working as intended on real small-model output: the agent
genuinely adapts, and the envelope makes an over-eager proposal harmless.

Two honest limitations of a 3B model here, both visible in `agent_reflections.json`:
- Its *stated* justifications drifted ("to ensure comfort during occupied hours" while it was in
  fact trading unoccupied comfort for savings). The **action** was bounded and energy-beneficial;
  the narration was not always faithful to it.
- Once a knob saturated, it kept re-proposing the same clamped change. The reflection brief now
  reports each knob's allowed range and flags `[AT MAX]`/`[AT MIN]` so it redirects to another
  parameter instead of repeating a no-op.

## 7. Handling lengthy simulation logs

EnergyPlus `.err`/`.audit` logs run to thousands of lines and would blow the context window.
`summarize_log` (a) strips everything except **severe/warning/fatal** lines, (b) **de-duplicates**
repeated messages, (c) **caps** the count, and (d) prepends a one-line tally. Only this compact
digest is ever shown to the model — cheap, in-context, and enough for error triage and
self-healing. The same function backs the MCP `summarize_simulation_log` tool.

## 8. Telemetry, comparison & the savings dashboard

Every step is logged to CSV (per-zone temps/PMV/CO₂, energy, cost, carbon, applied setpoints,
decision source, latency). A run summary captures total kWh, cost, carbon, peak demand, and the
comfort-OK rate over occupied zone-steps. `dashboard.py` renders a multi-panel PNG (cumulative
energy, zone-vs-comfort-band, control strategy vs price peaks, and a savings bar chart) and a
self-contained HTML dashboard with headline KPIs — the quantitative proof of **% kWh saved while
comfort is maintained**. Baseline and AI runs use identical weather/occupancy/tariff inputs; only
the control policy differs, so the delta is attributable purely to the agent.

## 8.1 Presentation dashboards (replay + live)

`ecoloop/live_dashboard.py` provides two surfaces over one UI, both dependency-free
(stdlib `http.server`, hand-rolled SVG, no CDN — they work offline on a judge's machine):

- **Interactive replay** (`python run.py live-dashboard`) embeds a finished run into a single
  self-contained HTML file. Press play and the week replays: the savings gap widens, zone
  temperatures track the comfort band, and the agent's **real logged reasoning** streams into a
  feed alongside nightly self-critiques and safety-guard interventions. It is a recording, so it
  cannot stall or fail during judging — this is the submitted artifact.
- **Live streaming** (`--live` on `agent`/`compare`/`demo`) runs a small HTTP server fed by a
  thread-safe `LiveState` that the control loop pushes each timestep into. The same page then
  polls `/api/state`. Every write is wrapped so a dashboard fault can never disturb control, and
  a standalone `agent --live` holds the final state on screen after the run ends rather than
  exiting and blanking the demo.

**Comparison integrity.** The live view overlays the baseline as a "ghost" for contrast, but a
leftover baseline from a *different* backend or start date would silently produce a nonsense
comparison (an RC agent charted against an EnergyPlus baseline appears to use twice the energy).
`_baseline_if_compatible` therefore **fails closed**: the ghost is dropped unless
`baseline_summary.json` exists *and* its backend and start time match the current run. Showing no
ghost is always preferable to showing a wrong one.

## 9. Reproduce

```bash
# one command: baseline vs autonomous agent on real EnergyPlus + dashboard
# (the submitted result is the 7-day horizon)
python run.py compare --backend energyplus --days 7
# fast physics twin (no EnergyPlus needed), instant:
python run.py compare --backend rc
# short, verbose, for the demo video:
python run.py demo --backend energyplus

# live dashboard while the loop runs (open http://127.0.0.1:8765):
python run.py agent --backend energyplus --days 1 --live

# rebuild the interactive replay dashboard from an existing run:
python run.py live-dashboard
```
