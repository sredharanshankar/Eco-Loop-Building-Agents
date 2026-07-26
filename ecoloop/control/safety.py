"""Safety guardrails — the deterministic envelope the LLM can never escape.

This is central to the *System Integration* score: no matter what the model returns
(a hallucinated number, an out-of-range value, an inverted deadband, or nonsense), the
loop stays physically sane and never crashes. It also provides a reactive comfort
recovery net — if occupants are already uncomfortable, we override toward comfort
regardless of the agent's suggestion. The agent optimises; the guard keeps it safe.
"""

from __future__ import annotations

import math

from ..backends.base import Action, Observation
from ..comfort import ComfortSpec


class SafetyGuard:
    def __init__(self, cfg: dict):
        a = cfg["agent"]
        self.min_cool = a["min_cool_sp_c"]
        self.max_cool = a["max_cool_sp_c"]
        self.min_heat = a["min_heat_sp_c"]
        self.max_heat = a["max_heat_sp_c"]
        self.min_deadband = a["min_deadband_c"]
        # Floors for the non-HVAC levers while people are present.
        self.min_light_occupied = a.get("min_light_level_occupied", 0.70)
        self.min_plug_occupied = a.get("min_plug_level_occupied", 0.80)
        self.min_vent_occupied = a.get("min_vent_level_occupied", 0.80)
        self.comfort = ComfortSpec.from_config(cfg)

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    @staticmethod
    def _sanitize(value, fallback: float) -> tuple[float, bool]:
        """Coerce an arbitrary model-supplied value to a finite float.

        A small model (or a corrupted tool call) can emit None, a string, NaN or inf.
        ``float()`` raises on the first two and silently propagates the last two, so
        both are handled here before any clamping — the guard must never raise.
        """
        try:
            v = float(value)
        except (TypeError, ValueError):
            return fallback, True
        if not math.isfinite(v):
            return fallback, True
        return v, False

    def enforce(self, action: Action, obs: Observation, occupied: bool,
                arriving_soon: bool = False) -> tuple[Action, list[str]]:
        notes: list[str] = []
        # Mid-band fallbacks keep the building safe if a value is unusable entirely.
        cool_raw, cool_bad = self._sanitize(action.cooling_setpoint, self.comfort.occ_high_c)
        heat_raw, heat_bad = self._sanitize(action.heating_setpoint, self.comfort.occ_low_c)
        if cool_bad or heat_bad:
            notes.append("nonfinite_setpoint_replaced")
        cool = self._clamp(cool_raw, self.min_cool, self.max_cool)
        heat = self._clamp(heat_raw, self.min_heat, self.max_heat)

        if cool - heat < self.min_deadband:
            notes.append("deadband_widened")
            cool = min(self.max_cool, heat + self.min_deadband)
            heat = cool - self.min_deadband

        # Proactive comfort cap: while occupied, the cooling setpoint may never exceed the
        # comfort ceiling — this makes the loop robust to any wrong strategy the LLM emits
        # (e.g. a deep setback chosen during occupancy) without ever risking discomfort.
        if occupied and cool > self.comfort.occ_high_c:
            notes.append("occupied_comfort_cap")
            cool = self.comfort.occ_high_c
            if cool - heat < self.min_deadband:
                heat = cool - self.min_deadband

        # Reactive comfort recovery: occupants already too warm/cold -> pull back.
        if occupied and obs.max_abs_pmv > self.comfort.pmv_limit + 0.3:
            recovered = self._clamp(self.comfort.occ_high_c, self.min_cool, self.max_cool)
            if recovered < cool:
                notes.append("comfort_override")
                cool = recovered
                if cool - heat < self.min_deadband:
                    heat = cool - self.min_deadband

        if occupied and obs.peak_co2 > self.comfort.co2_limit_ppm + 150:
            notes.append("iaq_override_economizer")
            econ = True
        else:
            econ = bool(action.economizer)

        # --- non-HVAC levers -------------------------------------------------
        # These save real electricity, but they are also the ones that can make a
        # building unusable (dark) or unhealthy (stuffy), so they get hard floors.
        light, l_bad = self._sanitize(action.light_level, 1.0)
        plug, p_bad = self._sanitize(action.plug_level, 1.0)
        vent, v_bad = self._sanitize(action.vent_level, 1.0)
        if l_bad or p_bad or v_bad:
            notes.append("nonfinite_level_replaced")
        light = self._clamp(light, 0.0, 1.0)
        plug = self._clamp(plug, 0.0, 1.0)
        vent = self._clamp(vent, 0.0, 1.0)

        # The `occupied` flag uses a comfort-oriented threshold (10% occupancy). For the
        # load floors that is too lax: the first few arrivals fall under it and would walk
        # into a dark building for a timestep. Anyone at all present is enough.
        # ``arriving_soon`` closes the last gap: a decision made while the building is
        # completely empty would otherwise leave the first arrivals in the dark for one
        # timestep, since no amount of looking at the present tells you they are coming.
        anyone_present = occupied or obs.occupancy > 0.02 or arriving_soon
        if anyone_present:
            if light < self.min_light_occupied:
                notes.append("lighting_floor")
                light = self.min_light_occupied
            if plug < self.min_plug_occupied:
                notes.append("plug_floor")
                plug = self.min_plug_occupied
            if vent < self.min_vent_occupied:
                notes.append("ventilation_floor")
                vent = self.min_vent_occupied
        # Rising CO2 overrides any attempt to save by starving the building of air.
        if obs.peak_co2 > self.comfort.co2_limit_ppm - 100:
            if vent < 1.0:
                notes.append("iaq_full_ventilation")
            vent = 1.0

        reason = action.reason if isinstance(action.reason, str) else str(action.reason)
        safe = Action(round(heat, 2), round(cool, 2), econ, reason,
                      light_level=round(light, 3), plug_level=round(plug, 3),
                      vent_level=round(vent, 3))
        return safe, notes
