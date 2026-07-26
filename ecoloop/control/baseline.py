"""Baseline controller — a conventional rule-based BMS.

Fixed occupied-hours setpoints with a deep night/weekend setback and no awareness of
price, carbon, weather forecast, or comfort headroom. This is the incumbent the
autonomous agent must beat; it is the reference for every savings number.
"""

from __future__ import annotations

from ..backends.base import Action, Observation


class BaselineController:
    name = "baseline"

    def __init__(self, cfg: dict):
        b = cfg["baseline"]
        self.occ_heat = b["occ_heat_sp"]
        self.occ_cool = b["occ_cool_sp"]
        self.setback_heat = b["setback_heat_sp"]
        self.setback_cool = b["setback_cool_sp"]
        self.start = b["occ_start_h"]
        self.end = b["occ_end_h"]

    def act(self, obs: Observation) -> Action:
        dt = obs.time
        occupied = dt.weekday() < 5 and self.start <= dt.hour < self.end
        if occupied:
            return Action(self.occ_heat, self.occ_cool, economizer=False,
                          reason="scheduled occupied setpoints")
        return Action(self.setback_heat, self.setback_cool, economizer=False,
                      reason="scheduled night/weekend setback")

    def decide(self, obs: Observation, backend=None) -> tuple[Action, dict]:
        """Uniform controller interface used by the loop runner."""
        return self.act(obs), {"source": "baseline"}

    def reset(self) -> None:
        pass
