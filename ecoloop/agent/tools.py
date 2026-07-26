"""Agentic tool registry (the MCP-equivalent surface).

These are the only ways the LLM senses and acts on the building — small, deterministic,
JSON in/out functions, exactly the shape MCP standardises. The same registry is exposed
over a real MCP server (``mcp_server.py``) and used in-process by the supervisor.

Design for a small local model: the terminal ``set_control`` tool takes an ECM *strategy*
(a label) rather than raw setpoint numbers. Choosing one of five well-defined measures is
reliable for a 3B model; the deterministic strategy->setpoint mapping then guarantees the
numbers are always valid and comfort-safe. This is both the reliability and the latency
win (short outputs, one call).

  sense   -> get_comfort_status
  look    -> get_weather_forecast, get_grid_forecast
  reason  -> evaluate_setpoint   (what-if, used by the MCP server / advanced mode)
  act     -> set_control         (terminal: forward-injects the chosen ECM into the sim)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..backends.base import Action, Observation, SimBackend
from ..comfort import ComfortSpec
from ..signals import TariffCarbon
from .policy import AdjustablePolicy

TERMINAL_TOOL = "set_control"
REFLECT_TOOL = "propose_policy_tweak"

STRATEGIES = ["setpoint_reset", "precool", "peak_coast", "precondition", "deep_setback"]


@dataclass
class ToolContext:
    obs: Observation
    backend: SimBackend
    tariff: TariffCarbon
    comfort: ComfortSpec
    cfg: dict
    occupied: bool
    policy: AdjustablePolicy | None = None


def _set_control_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": TERMINAL_TOOL,
            "description": "Commit ONE Energy Conservation Measure for the next hour. This "
                           "forward-injects the corresponding setpoints into the running "
                           "EnergyPlus simulation. Call exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": STRATEGIES,
                        "description": "setpoint_reset=occupied, run warm at comfort edge (default "
                                       "occupied, max saving); precool=occupied, peak within ~2h; "
                                       "peak_coast=occupied during a price peak; precondition="
                                       "unoccupied but occupancy resumes within ~2h; deep_setback="
                                       "unoccupied and empty for a while (max saving).",
                    },
                    "economizer": {
                        "type": "boolean",
                        "description": "Free cooling: true ONLY when outdoor air is cooler than indoors.",
                    },
                    "shed_nonessential": {
                        "type": "boolean",
                        "description": "Trim non-essential lighting and equipment power. Use when "
                                       "electricity is expensive (peak tier) or the grid is running "
                                       "dirty (high gCO2). Safe: occupants keep usable light.",
                    },
                    "reason": {"type": "string", "description": "One-line justification."},
                },
                "required": ["strategy", "reason"],
            },
        },
    }


def _reflect_schema(knob_names: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": REFLECT_TOOL,
            "description": "End-of-day self-critique: adjust ONE of your own control-policy "
                           "parameters for tomorrow, based on yesterday's energy and comfort "
                           "outcome. The value is clamped to a safe range automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter": {"type": "string", "enum": knob_names,
                                  "description": "Which policy parameter to adjust."},
                    "new_value": {"type": "number", "description": "The new value (degrees C)."},
                    "justification": {"type": "string",
                                      "description": "One line: what in yesterday's result "
                                                     "motivates this change."},
                },
                "required": ["parameter", "new_value", "justification"],
            },
        },
    }


class ToolRegistry:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx

    def update(self, ctx: ToolContext) -> None:
        self.ctx = ctx

    def schemas(self) -> list[dict[str, Any]]:
        """Full tool surface (used by the MCP server)."""
        return [
            self._sense_schema("get_comfort_status",
                               "Current live building state: per-zone temperature, PMV comfort, "
                               "CO2, occupancy, and which zones are outside comfort."),
            self._fc_schema("get_weather_forecast",
                            "Upcoming outdoor temperature (C) and solar level for the next N hours."),
            self._fc_schema("get_grid_forecast",
                            "Upcoming electricity price ($/kWh), tier, and carbon (gCO2/kWh)."),
            {
                "type": "function",
                "function": {
                    "name": "evaluate_setpoint",
                    "description": "What-if: predicted PMV comfort at candidate setpoints and whether "
                                   "they stay within comfort, plus a relative cooling-energy index.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cooling_setpoint": {"type": "number"},
                            "heating_setpoint": {"type": "number"},
                        },
                        "required": ["cooling_setpoint", "heating_setpoint"],
                    },
                },
            },
            _set_control_schema(),
        ]

    def action_schema(self) -> list[dict[str, Any]]:
        """The single terminal tool used in the latency-critical control loop."""
        return [_set_control_schema()]

    def reflect_schema(self) -> list[dict[str, Any]]:
        """The single tool offered during the end-of-day self-critique."""
        policy = self.ctx.policy or AdjustablePolicy.from_config(self.ctx.cfg)
        return [_reflect_schema(policy.knob_names())]

    @staticmethod
    def _sense_schema(name: str, desc: str) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": name, "description": desc, "parameters": {"type": "object", "properties": {}}}}

    @staticmethod
    def _fc_schema(name: str, desc: str) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object",
                           "properties": {"hours": {"type": "integer", "default": 6}}}}}

    # -- dispatch -------------------------------------------------------------
    def dispatch(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], Action | None]:
        try:
            if name == "get_comfort_status":
                return self._comfort_status(), None
            if name == "get_weather_forecast":
                return self._weather_forecast(int(args.get("hours", 6))), None
            if name == "get_grid_forecast":
                return self._grid_forecast(int(args.get("hours", 6))), None
            if name == "evaluate_setpoint":
                return self._evaluate(float(args["cooling_setpoint"]),
                                      float(args["heating_setpoint"])), None
            if name == REFLECT_TOOL:
                policy = self.ctx.policy
                if policy is None:
                    return {"error": "no adjustable policy in this context"}, None
                entry = policy.propose(
                    str(args.get("parameter", "")).strip(),
                    args.get("new_value"),
                    self.ctx.obs.time.date().isoformat(),
                    str(args.get("justification", ""))[:200],
                )
                return entry, None
            if name == TERMINAL_TOOL:
                strat = str(args.get("strategy", "")).strip().lower()
                if strat not in STRATEGIES:
                    return {"error": f"unknown strategy '{strat}'. Choose one of {STRATEGIES}."}, None
                # Occupancy is a hard fact, not a judgement call. A small model will happily
                # answer "deep_setback: unoccupied" at 2pm on a Monday; rejecting it here
                # forces a self-correcting retry instead of leaving a decision on record
                # whose stated reason is plainly false.
                occupied_only = {"setpoint_reset", "precool", "peak_coast"}
                empty_only = {"deep_setback", "precondition"}
                if self.ctx.occupied and strat in empty_only:
                    return {"error": f"'{strat}' assumes an empty building, but the building IS "
                                     f"OCCUPIED right now (occupancy "
                                     f"{self.ctx.obs.occupancy:.0%}). Choose one of "
                                     f"{sorted(occupied_only)} instead."}, None
                if not self.ctx.occupied and strat in occupied_only:
                    return {"error": f"'{strat}' assumes people are present, but the building is "
                                     f"EMPTY right now. Choose one of {sorted(empty_only)} "
                                     f"instead."}, None
                cool, heat = self._strategy_setpoints(strat)
                econ = bool(args.get("economizer", False))
                shed = bool(args.get("shed_nonessential", False))
                light, plug, vent = self._strategy_loads(strat, shed)
                reason = f"{strat}: {str(args.get('reason', ''))[:160]}"
                action = Action(round(heat, 1), round(cool, 1), econ, reason,
                                light_level=light, plug_level=plug, vent_level=vent)
                return {"status": "accepted", "strategy": strat,
                        "applied": action.as_dict()}, action
            return {"error": f"unknown tool '{name}'"}, None
        except (KeyError, TypeError, ValueError) as exc:
            return {"error": f"invalid arguments for {name}: {exc}"}, None

    def _strategy_setpoints(self, strat: str) -> tuple[float, float]:
        c = self.ctx.comfort
        lo, hi = c.occ_low_c, c.occ_high_c
        # Margins come from the bounded AdjustablePolicy so the nightly self-critique can
        # tune them within safe limits; defaults reproduce the original fixed behaviour.
        p = self.ctx.policy or AdjustablePolicy.from_config(self.ctx.cfg)
        reset_cool = hi - p.get("setpoint_reset_margin_c")
        table = {
            "setpoint_reset": (reset_cool, lo - 1.0),
            "precool": (hi - p.get("precool_margin_c"), lo - 1.0),
            "peak_coast": (hi, lo - 1.0),
            "precondition": (hi, lo - 2.0),
            # deep_setback goes deeper than the incumbent's weak setback for extra savings
            # when the building is empty (clamped by the safety envelope during occupancy).
            "deep_setback": (p.get("deep_setback_cool_c"), p.get("deep_setback_heat_c")),
        }
        return table.get(strat, (reset_cool, lo - 1.0))

    @staticmethod
    def _strategy_loads(strat: str, shed: bool) -> tuple[float, float, float]:
        """(light, plug, vent) levels for a strategy, as fractions of the schedule.

        An empty building needs almost no light, little equipment and minimal fresh air —
        that is where the non-HVAC savings come from, and it compounds because those loads
        also stop heating the space. The safety guard still has the final say, so an
        occupied zone can never actually go dark or unventilated.
        """
        table = {
            #                     light  plug  vent
            "setpoint_reset":    (1.00, 1.00, 1.00),
            "precool":           (1.00, 1.00, 1.00),
            "peak_coast":        (0.75, 0.90, 1.00),   # trim load through the price peak
            "precondition":      (0.30, 0.50, 0.60),   # nobody in yet, ramping up
            "deep_setback":      (0.05, 0.30, 0.15),   # empty: lights off, minimum air
        }
        light, plug, vent = table.get(strat, (1.0, 1.0, 1.0))
        if shed:
            # Deliberately modest: enough to matter on the meter, not enough to notice.
            light = min(light, 0.75)
            plug = min(plug, 0.85)
        return light, plug, vent

    # -- sensing implementations ----------------------------------------------
    def _comfort_status(self) -> dict[str, Any]:
        o = self.ctx.obs
        c = self.ctx.comfort
        lo, hi = c.band(self.ctx.occupied)
        zones = [{
            "zone": z, "temp_c": round(o.zone_temps[z], 1), "pmv": round(o.zone_pmv[z], 2),
            "co2_ppm": round(o.zone_co2[z]),
            "comfortable": abs(o.zone_pmv[z]) <= c.pmv_limit and o.zone_co2[z] <= c.co2_limit_ppm,
        } for z in o.zone_temps]
        return {
            "occupied": self.ctx.occupied, "occupancy_fraction": round(o.occupancy, 2),
            "outdoor_temp_c": round(o.outdoor_temp, 1), "comfort_band_c": [lo, hi],
            "pmv_limit": c.pmv_limit, "co2_limit_ppm": c.co2_limit_ppm,
            "energy_kwh_so_far": round(o.cumulative_kwh, 1), "zones": zones,
        }

    def _hourly(self, hours: int):
        hours = max(1, min(12, hours))
        steps_per_hour = max(1, 3600 // self.ctx.backend.timestep_seconds)
        fc = self.ctx.backend.forecast(hours * steps_per_hour + 1)
        return fc[::steps_per_hour][:hours]

    def _weather_forecast(self, hours: int) -> dict[str, Any]:
        return {"forecast": [{
            "time": e.time.strftime("%a %H:%M"), "outdoor_temp_c": round(e.t_out, 1),
            "solar_level": round(e.solar_ghi / 850.0, 2), "occupancy": round(e.occ, 2),
        } for e in self._hourly(hours)]}

    def _grid_forecast(self, hours: int) -> dict[str, Any]:
        return {"forecast": [{
            "time": e.time.strftime("%a %H:%M"),
            "price_per_kwh": round(self.ctx.tariff.price(e.time), 3),
            "tier": self.ctx.tariff.tier(e.time),
            "carbon_g_per_kwh": round(self.ctx.tariff.carbon(e.time)),
        } for e in self._hourly(hours)]}

    def _evaluate(self, cool_sp: float, heat_sp: float) -> dict[str, Any]:
        c = self.ctx.comfort
        pmv_cool = c.pmv(cool_sp)
        lo, hi = c.band(self.ctx.occupied)
        within = abs(pmv_cool) <= c.pmv_limit and cool_sp <= hi + 0.5 and cool_sp - heat_sp >= 1.0
        return {
            "pmv_at_cooling_setpoint": round(pmv_cool, 2),
            "pmv_at_heating_setpoint": round(c.pmv(heat_sp), 2),
            "within_comfort": bool(within),
            "relative_cooling_energy_index": round(max(0.0, self.ctx.obs.outdoor_temp - cool_sp), 1),
        }
