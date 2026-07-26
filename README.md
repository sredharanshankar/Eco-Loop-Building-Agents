# Eco-Loop Building Agents 

**An autonomous closed-loop building controller: an open-source LLM (via MCP-style tool-calling)
drives a live EnergyPlus simulation in real time — reading sensor feedback, reasoning about
weather/grid/comfort, and injecting control setpoints back into the running model to cut energy
while holding thermal comfort.**

Built for the Honeywell *Eco-Loop Building Agents* hackathon.

---

## What it does

```
EnergyPlus  ──stream sensors──►  LLM agent  ──choose ECM──►  Safety guard  ──inject setpoints──►  EnergyPlus
 (digital twin)   temps/energy/     (Ollama,     strategy       (comfort +        Zone Temp Control     (next timestep)
                  PMV/CO₂/grid)     MCP tools)                   envelope)          actuators
```

- **Real EnergyPlus** (`pyenergyplus` runtime API) as the physical sandbox — the same loop also
  runs on a fast physics **RC digital twin** for development and as a fallback.
- **Open-source LLM** (`qwen2.5:3b` on Ollama) as the brain, using **agentic tools / a real MCP
  server** to sense and act — no human in the loop.
- **Proven savings**: a baseline rigid BMS vs the autonomous agent on identical conditions, with a
  dashboard that quantifies **% kWh reduction while comfort is maintained**.

## Results — 7-day horizon on real EnergyPlus, fully LLM-driven

672 timesteps, Tampa TMY3 summer week, 5-zone office. Identical weather/occupancy/tariff for both
runs; **only the control policy differs**.

| Metric | Baseline BMS | Eco-Loop AI | Δ |
|--------|:-----------:|:-----------:|:--:|
| **HVAC energy** | 333.9 kWh | **255.7 kWh** | **−23.4%** |
| Energy cost | $91.90 | $73.65 | −19.9% |
| Grid carbon | 81.9 kg CO₂ | 66.3 kg CO₂ | −19.1% |
| Peak demand | 7.0 kW | 6.1 kW | −13.0% |
| **Occupied comfort-OK** | 88.6% | **89.9%** | **improved** |

Reliability over the full horizon: **672/672 steps, zero crashes, 168 LLM-driven decisions,
0 fallbacks**, plus **6 nightly self-critiques**. The AI saves energy *and* is slightly more
comfortable than the incumbent, because the baseline wastefully overcools (≈23.5 °C, PMV ≈ −0.6)
while the agent runs the warm comfort edge (≈26 °C, PMV ≈ +0.2).

Run it yourself — the numbers and dashboard regenerate from one command (below). Full write-up in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quick start

### Prerequisites
1. **Python 3.10+** and deps: `pip install -r requirements.txt`
2. **EnergyPlus v24+** (v26 tested) — the backend auto-adds its `pyenergyplus` API to the path.
   Set the install dir in `ecoloop/config.py` → `energyplus.install_dir` if not `C:\EnergyPlusV26-1-0`.
3. **Ollama** with a tool-capable model:
   ```bash
   ollama serve
   ollama pull qwen2.5:3b
   ```
   On low-RAM machines the model is auto-pinned (`keep_alive:-1`) to avoid reload stalls.

### Run

```bash
# Baseline vs autonomous agent on REAL EnergyPlus, then build the dashboard:
python run.py compare --backend energyplus --days 3

# Instant physics-twin run (no EnergyPlus required):
python run.py compare --backend rc

# Heuristic policy only (no LLM) — fast, deterministic, great for CI:
python run.py compare --backend energyplus --no-llm

# Short, verbose loop for the demo video (prints each LLM decision + injection):
python run.py demo --backend energyplus

# Fault-injection harness — deliberately break the loop and prove it survives:
python chaos_test.py --with-ep

# Interactive replay dashboard (open results/dashboard_live.html, press play):
python run.py live-dashboard

# LIVE dashboard — watch the agent control EnergyPlus in real time at localhost:8765:
python run.py agent --backend energyplus --days 1 --live
```

### The dashboards

| File / command | What it is |
|---|---|
| `results/dashboard_live.html` | **Interactive replay.** Self-contained — judges open it and press play to watch the whole week unfold: charts animate, KPIs update, and the agent's *real logged reasoning* streams in a feed. No server, no install, cannot stall. |
| `results/dashboard.png` / `.html` | Static multi-panel summary (good for slides and the report). |
| `--live` flag | **Live streaming.** Starts a local dashboard that updates while the loop runs, so you can watch EnergyPlus feed the agent and the agent inject setpoints back, in real time. Used for the demo video. |

Both share one UI and need **zero extra dependencies** (Python's stdlib HTTP server, hand-rolled
SVG charts, no CDN) so they render offline where external scripts are blocked.

Outputs land in `results/`: `dashboard.html` (open it), `dashboard.png`, `*_timeseries.csv`,
`*_summary.json`, `agent_runstats.json`, and the runtime-generated `results/ep/in.idf`.

### The MCP server

`mcp_server.py` is a real Model Context Protocol server (stdio, JSON-RPC 2.0). Register it with any
MCP client:

```json
{ "mcpServers": { "eco-loop": { "command": "python", "args": ["mcp_server.py"] } } }
```

Tools: `parse_building_model`, `summarize_simulation_log`, `evaluate_comfort`, `grid_signals`,
`run_closed_loop`.

## How it works (short version)

- **Backend-agnostic loop.** A `SimBackend` interface lets the identical agent/safety/telemetry
  stack drive either EnergyPlus (push→pull threaded bridge) or the RC twin.
- **Agent picks ECMs, not raw numbers.** `set_control` takes a *strategy* enum (setpoint_reset,
  precool, peak_coast, precondition, deep_setback) → reliable for a small model; a deterministic
  map yields safe setpoints.
- **Latency management.** Hourly supervisory cadence + decision caching + single-call prompts +
  a RAM-resident model keep the loop real-time on a CPU laptop.
- **Never crashes, never overheats.** A deterministic safety guard clamps every action and caps
  occupied cooling at the comfort ceiling; a heuristic fallback covers any LLM timeout; EnergyPlus
  runs in an isolated child process so even a native engine crash can't take down the controller.
- **Proven under fault injection.** `chaos_test.py` kills the LLM, feeds the agent malformed and
  adversarial output, fuzzes its self-critique, and hard-kills the EnergyPlus process mid-run —
  **6/6 scenarios survive** with comfort intact (`results/chaos_report.json`).
- **It critiques itself nightly.** At each simulated day boundary the agent reviews its own energy
  and comfort outcome and may retune **one bounded** policy knob for tomorrow — logged to
  `results/agent_reflections.json`, and still subject to the per-step safety guard.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the tool-calling architecture, prompt
engineering, latency strategy, and log handling; see [`PLAN.md`](PLAN.md) for the enhancement
roadmap.

## Repository layout

```
run.py              CLI: baseline | agent | compare | dashboard | demo
chaos_test.py       fault-injection harness (LLM faults, adversarial actions, EP process kill)
mcp_server.py       real MCP stdio server exposing the tools
ecoloop/            the package (config, comfort, signals, backends, agent, control, telemetry, dashboard)
models/baseline.idf base EnergyPlus building
docs/ARCHITECTURE.md   system architecture document
PLAN.md             master build & enhancement plan
presentation/       slide outline
results/            generated dashboards, CSVs, summaries
```
