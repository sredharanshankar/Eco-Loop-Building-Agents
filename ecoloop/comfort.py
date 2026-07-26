"""Thermal comfort — Fanger PMV/PPD per ISO 7730.

Comfort is a first-class constraint, not an afterthought: the agent is scored on
whether it saves energy *without* pushing occupants out of comfort. We expose the
predicted mean vote (PMV) and predicted percentage dissatisfied (PPD) so the loop
can reason about "is this setpoint still acceptable?" rather than just tracking a
raw temperature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def pmv_ppd(ta: float, tr: float, vel: float, rh: float,
            met: float, clo: float, wme: float = 0.0) -> tuple[float, float]:
    """Return (PMV, PPD) for the given conditions (ISO 7730 / ASHRAE 55).

    ta  : air temperature [C]
    tr  : mean radiant temperature [C]
    vel : relative air velocity [m/s]
    rh  : relative humidity [%]
    met : metabolic rate [met]
    clo : clothing insulation [clo]
    """
    pa = rh * 10.0 * math.exp(16.6536 - 4030.183 / (ta + 235.0))
    icl = 0.155 * clo
    m = met * 58.15
    w = wme * 58.15
    mw = m - w
    fcl = 1.0 + 1.29 * icl if icl <= 0.078 else 1.05 + 0.645 * icl
    hcf = 12.1 * math.sqrt(max(vel, 1e-6))
    taa = ta + 273.0
    tra = tr + 273.0
    tcla = taa + (35.5 - ta) / (3.5 * icl + 0.1)

    p1 = icl * fcl
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * taa
    p5 = 308.7 - 0.028 * mw + p2 * (tra / 100.0) ** 4
    xn = tcla / 100.0
    xf = xn
    eps = 0.00015
    for _ in range(150):
        xf = (xf + xn) / 2.0
        hcn = 2.38 * abs(100.0 * xf - taa) ** 0.25
        hc = hcf if hcf > hcn else hcn
        xn = (p5 + p4 * hc - p2 * xf ** 4) / (100.0 + p3 * hc)
        if abs(xn - xf) <= eps:
            break
    tcl = 100.0 * xn - 273.0

    hl1 = 3.05 * 0.001 * (5733.0 - 6.99 * mw - pa)
    hl2 = 0.42 * (mw - 58.15) if mw > 58.15 else 0.0
    hl3 = 1.7 * 0.00001 * m * (5867.0 - pa)
    hl4 = 0.0014 * m * (34.0 - ta)
    hl5 = 3.96 * fcl * (xn ** 4 - (tra / 100.0) ** 4)
    hl6 = fcl * hc * (tcl - ta)

    ts = 0.303 * math.exp(-0.036 * m) + 0.028
    pmv = ts * (mw - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    pmv = max(-3.5, min(3.5, pmv))
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pmv ** 4 - 0.2179 * pmv ** 2)
    return pmv, ppd


@dataclass
class ComfortSpec:
    met: float
    clo: float
    vel: float
    rh: float
    occ_low_c: float
    occ_high_c: float
    unocc_low_c: float
    unocc_high_c: float
    pmv_limit: float
    co2_limit_ppm: float

    @classmethod
    def from_config(cls, cfg: dict) -> "ComfortSpec":
        c = cfg["comfort"]
        return cls(
            met=c["met"], clo=c["clo"], vel=c["vel"], rh=c["rh"],
            occ_low_c=c["occ_low_c"], occ_high_c=c["occ_high_c"],
            unocc_low_c=c["unocc_low_c"], unocc_high_c=c["unocc_high_c"],
            pmv_limit=c["pmv_limit"], co2_limit_ppm=c["co2_limit_ppm"],
        )

    def pmv(self, ta: float, tr: float | None = None) -> float:
        return pmv_ppd(ta, tr if tr is not None else ta, self.vel, self.rh,
                       self.met, self.clo)[0]

    def band(self, occupied: bool) -> tuple[float, float]:
        if occupied:
            return self.occ_low_c, self.occ_high_c
        return self.unocc_low_c, self.unocc_high_c

    def is_violation(self, ta: float, occupied: bool, co2_ppm: float = 0.0) -> bool:
        """True if this zone is uncomfortable *while occupied*."""
        if not occupied:
            return False
        if abs(self.pmv(ta)) > self.pmv_limit:
            return True
        if co2_ppm > self.co2_limit_ppm:
            return True
        return False
