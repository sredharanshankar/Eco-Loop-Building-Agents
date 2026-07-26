"""Central configuration.

Defaults live in code so the project runs with zero external files. A ``config.json``
at the repo root (or a path passed on the CLI) is deep-merged over the defaults so
every tunable — comfort bands, tariffs, model name, control cadence — lives in one
place. This is also what lets the agent "modify parameters automatically": it edits a
copy of the config / IDF at runtime and re-runs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "run": {
        "days": 3,
        "timestep_minutes": 15,
        "start_date": "2026-07-20",   # a hot mid-summer week
        "seed": 42,
    },
    "building": {
        "zones": ["Core", "North", "East", "South", "West"],
        "zone_area_m2": [180.0, 70.0, 70.0, 70.0, 70.0],
        "orientation_deg": [0.0, 0.0, 90.0, 180.0, 270.0],  # N/E/S/W solar phase
        "UA_env_w_per_k": [90.0, 190.0, 190.0, 200.0, 190.0],  # core is buffered
        "C_zone_j_per_k": [5.0e6, 3.0e6, 3.0e6, 3.0e6, 3.0e6],
        "window_frac": [0.0, 0.30, 0.35, 0.45, 0.35],
        "internal_peak_w": [3500.0, 1800.0, 1800.0, 1800.0, 1800.0],
        "hvac_capacity_kw": 9.0,
        "cop_cool": 3.2,
        "cop_heat": 3.6,
        "fan_kw": 0.4,
        "vent_ach": 0.6,            # base ventilation air changes / hour
        "co2_gen_per_person_m3s": 4.0e-6,
        "people_peak": [12, 5, 5, 5, 5],
    },
    "comfort": {
        "met": 1.1, "clo": 0.5, "vel": 0.10, "rh": 50.0,
        "occ_low_c": 22.0, "occ_high_c": 26.0, "pmv_limit": 0.7,
        "unocc_low_c": 15.0, "unocc_high_c": 30.0,
        "co2_limit_ppm": 1000.0,
    },
    "baseline": {
        # Typical rigid BMS: overcools to 23 C during occupancy (wasteful AND slightly
        # cold, PMV about -0.75) with only a weak setback. This is the incumbent to beat.
        "occ_heat_sp": 21.5, "occ_cool_sp": 23.5,
        "setback_heat_sp": 18.0, "setback_cool_sp": 26.0,
        "occ_start_h": 8, "occ_end_h": 18,
    },
    "signals": {
        # Time-of-use tariff and grid carbon intensity drive the agent's ECMs.
        "price_offpeak": 0.09, "price_mid": 0.16, "price_peak": 0.42,   # $/kWh
        "peak_hours": [14, 15, 16, 17, 18],
        "mid_hours": [7, 8, 9, 10, 11, 12, 13, 19, 20, 21],
        "carbon_base": 210.0, "carbon_peak_add": 320.0,   # gCO2/kWh
        "carbon_peak_hours": [16, 17, 18, 19, 20],
        "carbon_solar_dip": 55.0,   # midday solar lowers grid carbon
    },
    "weather": {
        "mean_c": 29.0, "amp_c": 7.5, "phase_h": 15.0,   # peak ~3pm
        "heatwave_day": 1, "heatwave_bonus_c": 5.0,
        "solar_peak_w_per_m2": 850.0,
    },
    "agent": {
        "provider": "ollama",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "qwen2.5:3b",
        "control_interval_minutes": 60,   # supervisory cadence (latency management)
        "timeout_s": 60,
        "temperature": 0.15,
        "num_ctx": 4096,
        "use_cache": True,
        "max_retries": 2,
        # Nightly bounded self-critique: at each simulated day boundary the agent reviews
        # its own energy/comfort outcome and may retune ONE bounded policy knob.
        "reflection": True,
        # Safety envelope the agent may never leave (hard guardrails).
        "min_cool_sp_c": 21.5, "max_cool_sp_c": 28.0,
        "min_heat_sp_c": 18.0, "max_heat_sp_c": 23.0,
        "min_deadband_c": 1.5,
        # Floors for the non-HVAC levers while the building is occupied: people must
        # never be left in the dark, without their equipment, or short of fresh air.
        "min_light_level_occupied": 0.70,
        "min_plug_level_occupied": 0.80,
        "min_vent_level_occupied": 0.80,
    },
    "paths": {
        "results_dir": "results",
        "models_dir": "models",
    },
    "energyplus": {
        "install_dir": "C:\\EnergyPlusV26-1-0",
        "idf": "models/baseline.idf",
        "epw": "C:\\EnergyPlusV26-1-0\\WeatherData\\USA_FL_Tampa.Intl.AP.722110_TMY3.epw",
        "output_dir": "results/ep",
        "timestep_per_hour": 4,
        # EnergyPlus splits HVAC electricity across these end-use meters; we sum them.
        "hvac_meters": ["Cooling:Electricity", "Heating:Electricity", "Fans:Electricity",
                        "Pumps:Electricity", "HeatRejection:Electricity"],
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the merged configuration dict."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as fh:
                cfg = _deep_merge(cfg, json.load(fh))
    else:
        default_path = Path("config.json")
        if default_path.exists():
            with open(default_path, "r", encoding="utf-8") as fh:
                cfg = _deep_merge(cfg, json.load(fh))
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
