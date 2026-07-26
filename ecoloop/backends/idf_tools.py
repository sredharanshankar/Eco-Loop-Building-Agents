"""IDF and EPW helpers used by the EnergyPlus backend and exposed as agent tools.

These are the "parse files / modify the model without human code changes" capabilities
the brief asks for: read the controlled zones out of an .idf, retarget its run period,
set the timestep, read a weather forecast out of the .epw, and triage a runtime .err log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..weather import Exogenous, _occupancy


def set_run_period(text: str, bm: int, bd: int, em: int, ed: int) -> str:
    """Retarget the first RunPeriod to [bm/bd .. em/ed]."""
    def repl(m: re.Match) -> str:
        b = m.group(0)
        b = re.sub(r'-?\d+,(\s*!- Begin Month)', f'{bm},\\1', b)
        b = re.sub(r'-?\d+,(\s*!- Begin Day of Month)', f'{bd},\\1', b)
        b = re.sub(r'-?\d+,(\s*!- End Month)', f'{em},\\1', b)
        b = re.sub(r'-?\d+,(\s*!- End Day of Month)', f'{ed},\\1', b)
        return b
    return re.sub(r'RunPeriod,.*?;', repl, text, count=1, flags=re.DOTALL)


def set_timestep(text: str, n: int) -> str:
    return re.sub(r'Timestep,\s*\d+;', f'Timestep,{n};', text, count=1)


def keep_first_run_period(text: str) -> str:
    """Some example files ship several RunPeriods; keep only the first."""
    blocks = list(re.finditer(r'RunPeriod,.*?;', text, flags=re.DOTALL))
    for b in reversed(blocks[1:]):
        text = text[:b.start()] + text[b.end():]
    return text


def parse_internal_loads(text: str) -> dict[str, list[dict]]:
    """Lighting and plug-load objects with their schedule name and design wattage.

    The ``Lights``/``ElectricEquipment`` actuators take an ABSOLUTE power in watts, not a
    fraction. Overriding with a flat fraction would erase the schedule's shape — asking
    for "100%" at 3am would blast the lights at full design power, far worse than the
    baseline. So we capture the design level here and, at runtime, multiply it by the
    (un-actuated, still-readable) schedule value and the agent's requested level.

    Only the explicit ``LightingLevel``/``EquipmentLevel`` form is supported; anything
    else (watts per area/person) is skipped rather than guessed at.
    """
    out: dict[str, list[dict]] = {"lights": [], "equips": []}
    kinds = {"lights": ("lights", "lightinglevel"),
             "equips": ("electricequipment", "equipmentlevel")}
    for f in _iter_objects(text):
        for key, (obj_type, method) in kinds.items():
            if f[0].lower() != obj_type or len(f) < 6:
                continue
            if f[4].strip().lower() != method:
                continue
            try:
                design_w = float(f[5])
            except ValueError:
                continue
            out[key].append({"name": f[1], "schedule": f[3], "design_w": design_w})
    return out


def parse_ideal_loads_units(text: str) -> list[str]:
    """Names of ZoneHVAC:IdealLoadsAirSystem objects (energy is reported per unit)."""
    names: list[str] = []
    for f in _iter_objects(text):
        if f[0].lower() == "zonehvac:idealloadsairsystem" and len(f) >= 2:
            if f[1] not in names:
                names.append(f[1])
    return names


def _iter_objects(text: str):
    """Yield each IDF object as a list of stripped fields [type, field1, field2, ...].

    Strips ``!``/``!-`` comments first so fields are read, not their descriptions.
    """
    clean = re.sub(r'!.*', '', text)
    for obj in clean.split(";"):
        fields = [f.strip() for f in obj.split(",")]
        fields = [f for f in fields if f]
        if fields:
            yield fields


def parse_controlled_zones(text: str) -> list[str]:
    """Zone names that have a ZoneControl:Thermostat (i.e. the conditioned zones)."""
    zones: list[str] = []
    for f in _iter_objects(text):
        if f[0].lower() == "zonecontrol:thermostat" and len(f) >= 3:
            z = f[2]
            if z not in zones:
                zones.append(z)
    return zones


def summarize_err(path: str, max_lines: int = 20) -> str:
    """Compress an EnergyPlus .err file to its severe/warning/fatal lines."""
    try:
        text = open(path, encoding="utf-8", errors="ignore").read()
    except OSError as exc:
        return f"could not read err file: {exc}"
    from ..agent.prompts import summarize_log
    return summarize_log(text, max_lines)


@dataclass
class EPWForecast:
    """Hourly outdoor dry-bulb and global horizontal irradiance parsed from an EPW."""
    by_key: dict[tuple[int, int, int], tuple[float, float]]
    year: int

    def at(self, dt: datetime) -> Exogenous:
        t, g = self.by_key.get((dt.month, dt.day, dt.hour), (30.0, 0.0))
        return Exogenous(0, dt, t, g, _occupancy(dt))

    def forecast(self, start: datetime, n: int, step_seconds: int) -> list[Exogenous]:
        out = []
        for k in range(n):
            dt = start + timedelta(seconds=k * step_seconds)
            out.append(self.at(dt))
        return out


def parse_epw(path: str, year: int) -> EPWForecast:
    by_key: dict[tuple[int, int, int], tuple[float, float]] = {}
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for _ in range(8):            # skip EPW header
            fh.readline()
        for line in fh:
            parts = line.split(",")
            if len(parts) < 15:
                continue
            try:
                month = int(parts[1]); day = int(parts[2])
                hour = int(parts[3]) - 1          # EPW hour is 1..24 (hour-ending)
                tdry = float(parts[6]); ghi = float(parts[13])
            except (ValueError, IndexError):
                continue
            by_key[(month, day, hour % 24)] = (tdry, ghi)
    return EPWForecast(by_key, year)
