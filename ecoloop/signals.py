"""Grid signals: time-of-use electricity price and marginal carbon intensity.

These are the levers that make "smart" control pay off. A flat-schedule BMS ignores
them; the agent uses them to shift and shed load — pre-cooling on cheap, low-carbon
midday power and coasting through the expensive, dirty evening peak. Backend-agnostic:
depends only on wall-clock time.
"""

from __future__ import annotations

import math
from datetime import datetime


class TariffCarbon:
    def __init__(self, cfg: dict):
        s = cfg["signals"]
        self.price_offpeak = s["price_offpeak"]
        self.price_mid = s["price_mid"]
        self.price_peak = s["price_peak"]
        self.peak_hours = set(s["peak_hours"])
        self.mid_hours = set(s["mid_hours"])
        self.carbon_base = s["carbon_base"]
        self.carbon_peak_add = s["carbon_peak_add"]
        self.carbon_peak_hours = set(s["carbon_peak_hours"])
        self.carbon_solar_dip = s["carbon_solar_dip"]

    def price(self, dt: datetime) -> float:
        """$/kWh."""
        h = dt.hour
        if h in self.peak_hours:
            return self.price_peak
        if h in self.mid_hours:
            return self.price_mid
        return self.price_offpeak

    def carbon(self, dt: datetime) -> float:
        """Marginal grid carbon intensity, gCO2/kWh."""
        h = dt.hour + dt.minute / 60.0
        solar_norm = max(0.0, math.sin(math.pi * (h - 6.0) / 13.0)) if 6.0 <= h <= 19.0 else 0.0
        c = self.carbon_base
        if int(h) in self.carbon_peak_hours:
            c += self.carbon_peak_add
        c -= self.carbon_solar_dip * solar_norm
        return max(60.0, c)

    def tier(self, dt: datetime) -> str:
        h = dt.hour
        if h in self.peak_hours:
            return "peak"
        if h in self.mid_hours:
            return "mid"
        return "offpeak"
