"""Exogenous drivers for the RC fallback backend.

Provides a deterministic, reproducible summer week (with an optional heat-wave day)
so results are comparable between the baseline and the AI run. The EnergyPlus backend
uses a real EPW file instead; both expose the same :class:`Exogenous` record so the
rest of the system (agent tools, forecasts) is backend-agnostic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Exogenous:
    index: int
    time: datetime
    t_out: float        # outdoor dry-bulb [C]
    solar_ghi: float    # global horizontal irradiance [W/m2]
    occ: float          # occupancy fraction [0..1]


def _occupancy(dt: datetime) -> float:
    """Typical small-office occupancy schedule with a lunch dip."""
    if dt.weekday() >= 5:        # weekend
        return 0.05
    h = dt.hour + dt.minute / 60.0
    if h < 6.5 or h >= 19.0:
        return 0.0
    if h < 8.0:
        return (h - 6.5) / 1.5
    if h < 12.0:
        return 1.0
    if h < 13.0:
        return 0.6
    if h < 17.0:
        return 1.0
    return max(0.0, 1.0 - (h - 17.0) / 2.0)


class SyntheticWeather:
    """Precomputes the full exogenous horizon for speed and determinism."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        run = cfg["run"]
        w = cfg["weather"]
        self.step_seconds = int(run["timestep_minutes"] * 60)
        self.n_steps = int(run["days"] * 24 * 3600 / self.step_seconds)
        self.start = datetime.fromisoformat(run["start_date"])
        self._t_out: list[float] = []
        self._solar: list[float] = []
        self._occ: list[float] = []
        rng = _LCG(run["seed"])
        for i in range(self.n_steps + 96):   # +1 day of lookahead for forecasts
            dt = self.start + timedelta(seconds=i * self.step_seconds)
            h = dt.hour + dt.minute / 60.0
            day = (dt - self.start).days
            # Diurnal temperature with a warm bias on the heat-wave day.
            base = w["mean_c"] + w["amp_c"] * math.sin(2 * math.pi * (h - w["phase_h"]) / 24.0)
            if day == w["heatwave_day"]:
                base += w["heatwave_bonus_c"]
            base += 1.2 * math.sin(2 * math.pi * day / 4.0)      # slow day-to-day drift
            base += (rng.next() - 0.5) * 0.8                     # small noise
            # Daylight bell for solar.
            frac = math.sin(math.pi * (h - 6.0) / 13.0)
            solar = w["solar_peak_w_per_m2"] * max(0.0, frac) if 6.0 <= h <= 19.0 else 0.0
            self._t_out.append(base)
            self._solar.append(solar)
            self._occ.append(_occupancy(dt))

    def at(self, i: int) -> Exogenous:
        i = max(0, min(i, len(self._t_out) - 1))
        dt = self.start + timedelta(seconds=i * self.step_seconds)
        return Exogenous(i, dt, self._t_out[i], self._solar[i], self._occ[i])

    def forecast(self, i: int, n: int) -> list[Exogenous]:
        return [self.at(j) for j in range(i, i + n)]


class _LCG:
    """Tiny seeded PRNG so runs are byte-for-byte reproducible without touching
    numpy's global state."""

    def __init__(self, seed: int):
        self.state = (seed * 2654435761 + 12345) & 0xFFFFFFFF

    def next(self) -> float:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state / 0x7FFFFFFF
