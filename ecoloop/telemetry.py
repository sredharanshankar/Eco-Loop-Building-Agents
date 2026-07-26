"""Structured telemetry: per-step timeseries + a run-level summary.

Everything the dashboard and the evaluation need comes from here — total kWh, cost,
carbon, peak demand, and the comfort accounting (violation rate over occupied zone-
steps). Written as CSV (timeseries) and JSON (summary) so results are inspectable and
diffable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .backends.base import Observation


class Recorder:
    def __init__(self, tag: str, cfg: dict, backend_name: str, controller_name: str):
        self.tag = tag
        self.cfg = cfg
        self.backend_name = backend_name
        self.controller_name = controller_name
        self.rows: list[dict[str, Any]] = []
        self.n_zones = len(cfg["building"]["zones"])
        self._dt_h = cfg["run"]["timestep_minutes"] / 60.0
        # accumulators
        self.occupied_zone_steps = 0
        self.violation_zone_steps = 0
        self.pmv_sum_occ = 0.0
        self.pmv_n_occ = 0
        self.max_abs_pmv = 0.0
        self.peak_step_kwh = 0.0
        self.decision_sources: dict[str, int] = {}
        self.decision_log: list[dict[str, Any]] = []
        self.last: Observation | None = None

    def record(self, obs: Observation, meta: dict[str, Any]) -> None:
        row = obs.to_row()
        row["source"] = meta.get("source", "")
        row["latency_s"] = meta.get("latency_s", "")
        self.rows.append(row)
        self.last = obs

        occupied = obs.occupancy > 0.10
        if occupied:
            self.occupied_zone_steps += self.n_zones
            self.violation_zone_steps += obs.comfort_violations
            for p in obs.zone_pmv.values():
                self.pmv_sum_occ += p
                self.pmv_n_occ += 1
            self.max_abs_pmv = max(self.max_abs_pmv, obs.max_abs_pmv)
        self.peak_step_kwh = max(self.peak_step_kwh, obs.step_kwh)

        src = meta.get("source", "")
        if src:
            self.decision_sources[src] = self.decision_sources.get(src, 0) + 1
        if src in ("llm", "fallback"):
            self.decision_log.append({
                "time": obs.time.isoformat(),
                "source": src,
                "latency_s": meta.get("latency_s"),
                "cooling_setpoint": obs.applied_action.get("cooling_setpoint"),
                "heating_setpoint": obs.applied_action.get("heating_setpoint"),
                "economizer": obs.applied_action.get("economizer"),
                "reason": obs.applied_action.get("reason", ""),
                "safety": meta.get("safety"),
            })

    def summary(self) -> dict[str, Any]:
        last = self.last
        occ_ok = 1.0 - (self.violation_zone_steps / self.occupied_zone_steps) \
            if self.occupied_zone_steps else 1.0
        peak_kw = self.peak_step_kwh / self._dt_h if self._dt_h else 0.0
        return {
            "tag": self.tag,
            "backend": self.backend_name,
            "controller": self.controller_name,
            "steps": len(self.rows),
            "total_kwh": round(last.cumulative_kwh, 2) if last else 0.0,
            "total_cost_usd": round(last.cumulative_cost, 2) if last else 0.0,
            "total_co2_kg": round(last.cumulative_co2_kg, 2) if last else 0.0,
            "peak_demand_kw": round(peak_kw, 2),
            "occupied_zone_steps": self.occupied_zone_steps,
            "comfort_violation_zone_steps": self.violation_zone_steps,
            "comfort_ok_rate": round(occ_ok, 4),
            "mean_pmv_occupied": round(self.pmv_sum_occ / self.pmv_n_occ, 3) if self.pmv_n_occ else 0.0,
            "max_abs_pmv_occupied": round(self.max_abs_pmv, 3),
            "decision_sources": self.decision_sources,
        }

    def save(self, results_dir: str | Path) -> dict[str, str]:
        d = Path(results_dir)
        d.mkdir(parents=True, exist_ok=True)
        ts_path = d / f"{self.tag}_timeseries.csv"
        sum_path = d / f"{self.tag}_summary.json"
        dec_path = d / f"{self.tag}_decisions.json"
        if self.rows:
            fields = list(self.rows[0].keys())
            with open(ts_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(self.rows)
        with open(sum_path, "w", encoding="utf-8") as fh:
            json.dump(self.summary(), fh, indent=2)
        with open(dec_path, "w", encoding="utf-8") as fh:
            json.dump(self.decision_log, fh, indent=2)
        return {"timeseries": str(ts_path), "summary": str(sum_path), "decisions": str(dec_path)}
