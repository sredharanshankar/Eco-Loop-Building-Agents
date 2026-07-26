"""The LLM supervisory controller — orchestrates sense -> reason -> act each hour.

Flow per decision:
  1. Cache check (latency management): identical situation -> reuse decision, no inference.
  2. Front-load a compact situational brief (state + 6h weather + 6h grid) into one user turn.
  3. Run a bounded tool-calling loop: the model may query/what-if, and must finish by
     calling set_control (forward injection).
  4. Every model output passes through the deterministic SafetyGuard in the loop.
  5. If the model errors, times out, or never commits, fall back to a deterministic smart
     ECM controller so the closed loop NEVER stalls.

This design makes the agent genuinely LLM-driven while remaining robust over long
horizons and cheap enough to run in near real-time on a 3B CPU model.
"""

from __future__ import annotations

import json
from typing import Any

from ..backends.base import Action, Observation, SimBackend
from ..comfort import ComfortSpec
from ..signals import TariffCarbon
from .cache import DecisionCache
from .llm_client import LLMClient
from .policy import AdjustablePolicy
from .prompts import (REFLECTION_PROMPT, SYSTEM_PROMPT, build_reflection_text,
                      parse_tool_arguments)
from .tools import TERMINAL_TOOL, ToolContext, ToolRegistry

MAX_TOOL_TURNS = 2   # 1 decision call + at most 1 self-correction


class AgentSupervisor:
    name = "agent"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tariff = TariffCarbon(cfg)
        self.comfort = ComfortSpec.from_config(cfg)
        self.llm = LLMClient(cfg)
        self.disable_llm = cfg["agent"].get("disable_llm", False)
        self.cache = DecisionCache(self.tariff, enabled=cfg["agent"].get("use_cache", True))
        b = cfg["baseline"]
        self.setback_heat = b["setback_heat_sp"]
        self.setback_cool = b["setback_cool_sp"]
        # Self-critique: bounded policy the agent may retune at each day boundary.
        self.policy = AdjustablePolicy.from_config(cfg)
        self.reflect_enabled = cfg["agent"].get("reflection", True)
        self.reflections: list[dict[str, Any]] = []
        # run stats
        self.llm_calls = 0
        self.total_latency = 0.0
        self.decisions = {"llm": 0, "cache": 0, "fallback": 0}
        self.self_corrections = 0
        self.errors: list[str] = []

    def reset(self) -> None:
        self.cache = DecisionCache(self.tariff, enabled=self.cfg["agent"].get("use_cache", True))
        self.policy = AdjustablePolicy.from_config(self.cfg)
        self.reflections = []
        self.llm_calls = 0
        self.total_latency = 0.0
        self.decisions = {"llm": 0, "cache": 0, "fallback": 0}
        self.self_corrections = 0
        self.errors = []

    # -- main entry -----------------------------------------------------------
    def decide(self, obs: Observation, backend: SimBackend) -> tuple[Action, dict[str, Any]]:
        """Uniform controller interface used by the loop runner."""
        return self.act(obs, backend)

    def act(self, obs: Observation, backend: SimBackend) -> tuple[Action, dict[str, Any]]:
        occupied = obs.occupancy > 0.10
        ctx = ToolContext(obs, backend, self.tariff, self.comfort, self.cfg, occupied,
                          policy=self.policy)
        registry = ToolRegistry(ctx)

        cached = self.cache.get(obs, occupied)
        if cached is not None:
            # Re-derive setpoints so a cache hit always reflects the CURRENT policy,
            # including any tweak the nightly self-critique has since applied.
            _, action = registry.dispatch(TERMINAL_TOOL, cached)
            if action is not None:
                action.reason += " [cached]"
                self.decisions["cache"] += 1
                return action, {"source": "cache"}

        if self.disable_llm:
            action, meta, args = None, {}, None
        else:
            action, meta, args = self._ask_llm(registry, obs, occupied)
        if action is None:
            action = self._heuristic(obs, backend, occupied)
            self.decisions["fallback"] += 1
            meta = {**meta, "source": "fallback"}
        else:
            self.decisions["llm"] += 1
            meta = {**meta, "source": "llm"}
            self.cache.put(obs, occupied, args)
        return action, meta

    # -- nightly self-critique ------------------------------------------------
    def on_new_day(self, obs: Observation, backend: SimBackend, day_stats: dict) -> None:
        """Called by the loop runner at each simulated day boundary.

        The agent reviews the day it just finished and may retune ONE bounded policy
        knob for tomorrow. Failures here are swallowed: reflection is an enhancement,
        never a reason for the control loop to stop.
        """
        if not self.reflect_enabled or self.disable_llm:
            return
        try:
            self._reflect(obs, backend, day_stats)
        except Exception as exc:  # noqa: BLE001 - never let self-critique break the run
            self.errors.append(f"reflection failed: {type(exc).__name__}: {exc}")

    def _reflect(self, obs: Observation, backend: SimBackend, day_stats: dict) -> None:
        ctx = ToolContext(obs, backend, self.tariff, self.comfort, self.cfg,
                          obs.occupancy > 0.10, policy=self.policy)
        registry = ToolRegistry(ctx)
        day = day_stats.get("day", obs.time.date().isoformat())
        user = build_reflection_text(
            day, day_stats.get("kwh", 0.0), day_stats.get("violations", 0),
            day_stats.get("occupied_zone_steps", 0), day_stats.get("mean_pmv", 0.0),
            self.policy.describe_with_bounds(),
        )
        messages = [
            {"role": "system", "content": REFLECTION_PROMPT},
            {"role": "user", "content": user},
        ]
        res = self.llm.chat(messages, tools=registry.reflect_schema(), tool_choice="auto")
        self.llm_calls += 1
        self.total_latency += res.latency_s
        if not res.ok:
            self.errors.append(f"reflection call failed: {res.error}")
            return
        tcs = getattr(res.message, "tool_calls", None) or []
        for tc in tcs:
            args = parse_tool_arguments(tc.function.arguments)
            entry, _ = registry.dispatch(tc.function.name, args)
            entry = {**entry, "day_stats": day_stats, "latency_s": round(res.latency_s, 2)}
            self.reflections.append(entry)
            # No cache invalidation needed: the cache holds the chosen STRATEGY, and a
            # hit re-derives setpoints from the live policy, so a retune is picked up
            # automatically. Clearing it here (an earlier design) forced a full day of
            # re-inference every night and drove the hit rate to zero.
            break   # at most one tweak per day

    # -- LLM tool-calling loop ------------------------------------------------
    def _ask_llm(self, registry: ToolRegistry, obs: Observation,
                 occupied: bool) -> tuple[Action | None, dict[str, Any], dict | None]:
        # Compact, front-loaded context so the model decides in ONE short call — the core
        # latency move. The derived FACTS line already encodes what drives the strategy, so
        # we skip the verbose per-zone + hourly forecast dump (still computed for the FACTS).
        lo, hi = self.comfort.band(occupied)
        worst_pmv = max(obs.zone_pmv.values(), key=abs) if obs.zone_pmv else 0.0
        grid = registry.dispatch("get_grid_forecast", {"hours": 3})[0]
        next3 = ", ".join(f"{g['time'][-5:]}={g['tier']}" for g in grid["forecast"])
        user = (
            f"STATE: occupied={occupied} outdoor={obs.outdoor_temp:.0f}C "
            f"mean_zone={obs.mean_zone_temp:.1f}C worst_PMV={worst_pmv:+.2f} "
            f"peak_CO2={obs.peak_co2:.0f}ppm energy_so_far={obs.cumulative_kwh:.0f}kWh "
            f"comfort_band=[{lo},{hi}]C\n"
            f"GRID next 3h: {next3}\n"
            f"{self._facts(obs, registry.ctx.backend, occupied)}\n"
            f"Pick ONE strategy and call set_control."
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        tools = registry.action_schema()          # only the terminal tool -> one round-trip
        committed: Action | None = None
        committed_args: dict | None = None
        latency = 0.0
        calls = 0
        corrections = 0

        for _ in range(MAX_TOOL_TURNS):            # normally 1; a 2nd only self-corrects
            res = self.llm.chat(messages, tools=tools, tool_choice="auto")
            self.llm_calls += 1
            calls += 1
            latency += res.latency_s
            self.total_latency += res.latency_s
            if not res.ok:
                self.errors.append(res.error)
                break
            msg = res.message
            tcs = getattr(msg, "tool_calls", None) or []
            if not tcs:
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({"role": "user",
                                 "content": "Call set_control now with a strategy from the list."})
                continue
            messages.append(self._assistant_msg(msg, tcs))
            for tc in tcs:
                args = parse_tool_arguments(tc.function.arguments)
                result, action = registry.dispatch(tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result)})
                if action is not None:
                    committed, committed_args = action, args
                elif "error" in result:
                    # The tool rejected the choice; the error goes back to the model and
                    # the next turn is a genuine self-correction rather than a retry.
                    corrections += 1
            if committed is not None:
                break
        meta = {"latency_s": round(latency, 2), "tool_calls": calls}
        if corrections:
            meta["self_corrections"] = corrections
            self.self_corrections += corrections
        return committed, meta, committed_args

    def _facts(self, obs: Observation, backend: SimBackend, occupied: bool) -> str:
        sph = max(1, 3600 // backend.timestep_seconds)
        fc = backend.forecast(2 * sph + 1)
        occ_soon = any(e.occ > 0.1 for e in fc)
        tier = self.tariff.tier(obs.time)
        peak_soon = tier != "peak" and any(self.tariff.tier(e.time) == "peak" for e in fc)
        free_cool = obs.outdoor_temp < obs.mean_zone_temp
        carbon = self.tariff.carbon(obs.time)
        dirty = carbon > self.tariff.carbon_base + 80
        # occupied_now leads and is stated in words as well as a boolean: a 3B model was
        # observed skimming past the flag and answering "unoccupied" during work hours.
        state = "PEOPLE ARE IN THE BUILDING" if occupied else "THE BUILDING IS EMPTY"
        return (f"FACTS: {state}. occupied_now={occupied}, "
                f"occupancy_within_2h={occ_soon}, "
                f"price_tier_now={tier}, peak_within_2h={peak_soon}, "
                f"free_cooling_possible={free_cool}, "
                f"grid_carbon={carbon:.0f}gCO2/kWh, high_carbon_hour={dirty}")

    @staticmethod
    def _assistant_msg(msg: Any, tcs: list) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                        if isinstance(tc.function.arguments, str)
                        else json.dumps(tc.function.arguments),
                    },
                }
                for tc in tcs
            ],
        }

    # -- deterministic fallback (also a strong ECM controller) ----------------
    def _heuristic(self, obs: Observation, backend: SimBackend, occupied: bool) -> Action:
        from .tools import ToolRegistry
        occ_lo, occ_hi = self.comfort.occ_low_c, self.comfort.occ_high_c
        tier = self.tariff.tier(obs.time)
        dirty = self.tariff.carbon(obs.time) > self.tariff.carbon_base + 80
        shed = tier == "peak" or dirty
        # Reasons are prefixed with the strategy name (same shape as the LLM path) so the
        # dashboard can translate them into plain English for a non-technical audience.
        if occupied:
            # Cooling-setpoint reset: run at the comfort ceiling. The baseline overcools
            # to 24 C (PMV -0.5, wasteful); holding the ceiling cuts cooling energy AND
            # sits closer to thermal neutrality, so comfort improves while energy drops.
            strat = "peak_coast" if tier == "peak" else "setpoint_reset"
            cool = occ_hi
            heat = occ_lo - 1.0
            why = ("power is at its most expensive, so coasting and trimming load"
                   if tier == "peak" else
                   "holding the warm edge of comfort instead of overcooling")
            # Free-cooling assist only when outdoor air is genuinely cooler (net-saving).
            econ = obs.outdoor_temp < obs.mean_zone_temp - 1.0
        else:
            strat = "deep_setback"
            cool = self.setback_cool
            heat = self.setback_heat
            # Night flush only when it clearly pays back: cool night + still-warm mass.
            econ = obs.outdoor_temp < 21.0 and obs.mean_zone_temp > 25.0
            why = ("building is empty, so lights, equipment and fresh air are dialled down"
                   + (" while cool night air flushes out the heat" if econ else ""))
        light, plug, vent = ToolRegistry._strategy_loads(strat, shed)
        return Action(round(heat, 1), round(cool, 1), econ, f"{strat}: {why}",
                      light_level=light, plug_level=plug, vent_level=vent)

    # -- reporting ------------------------------------------------------------
    def run_stats(self) -> dict[str, Any]:
        total = sum(self.decisions.values())
        return {
            "llm_calls": self.llm_calls,
            "avg_latency_s": round(self.total_latency / self.llm_calls, 2) if self.llm_calls else 0.0,
            "decisions": self.decisions,
            "llm_driven_fraction": round(
                (self.decisions["llm"] + self.decisions["cache"]) / total, 3) if total else 0.0,
            **self.cache.stats,
            "self_corrections": self.self_corrections,
            "reflections": len(self.reflections),
            "final_policy": self.policy.describe(),
            "recent_errors": self.errors[-3:],
        }
