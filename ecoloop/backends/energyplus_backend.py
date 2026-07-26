"""High-fidelity EnergyPlus backend via the pyenergyplus runtime API.

EnergyPlus is push-driven (it calls a callback each timestep); our controller is
pull-driven (it calls ``step``). We bridge the two by running EnergyPlus in a dedicated
child **process** (``ep_worker.py``) and exchanging newline-delimited JSON over its
stdin/stdout, turning EnergyPlus into a gym-style ``reset``/``step`` environment. The
*identical* controller, safety guard, telemetry and dashboard then run unchanged on real
EnergyPlus — flipping ``--backend energyplus`` is the only difference.

Process (not thread) isolation is deliberate: an earlier in-process background-thread
bridge was found to crash intermittently near the end of a run on Windows (native SEH
0xe06d7363 raised from deep inside ``run_energyplus`` when invoked off the main thread).
Every probe that called EnergyPlus directly on a process's main thread never crashed
across many runs. Running the engine as a child process's main thread avoids the failure
mode entirely, and a native crash there can never corrupt or kill the controller process.

Feedback  : Zone Mean Air Temperature, Site Outdoor Dry-Bulb, ideal-loads HVAC energy.
Injection : the built-in "Zone Temperature Control" Heating/Cooling Setpoint actuators.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from ..comfort import ComfortSpec
from ..signals import TariffCarbon
from ..weather import Exogenous, _occupancy
from . import idf_tools
from .base import Action, Observation, SimBackend


class EnergyPlusBackend(SimBackend):
    name = "energyplus"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        ep = cfg["energyplus"]
        self.install_dir = ep["install_dir"]

        run = cfg["run"]
        self.year = datetime.fromisoformat(run["start_date"]).year
        self._start = datetime.fromisoformat(run["start_date"])
        self._days = run["days"]
        self._tsph = ep["timestep_per_hour"]
        self._dt = 3600 // self._tsph
        self._n_steps = self._days * 24 * self._tsph

        self.tariff = TariffCarbon(cfg)
        self.comfort = ComfortSpec.from_config(cfg)
        self.epw = ep["epw"]
        b = cfg["building"]
        self.cop_cool = b["cop_cool"]
        self.cop_heat = b["cop_heat"]
        self.fan_w = b["fan_kw"] * 1000.0

        # Prepare a run-scoped IDF: single summer run period at our timestep.
        from datetime import timedelta
        base_text = Path(ep["idf"]).read_text(encoding="utf-8", errors="ignore")
        end = self._start + timedelta(days=self._days - 1)
        text = idf_tools.keep_first_run_period(base_text)
        text = idf_tools.set_run_period(text, self._start.month, self._start.day,
                                        end.month, end.day)
        text = idf_tools.set_timestep(text, self._tsph)
        self.zones = idf_tools.parse_controlled_zones(text)
        self.units = idf_tools.parse_ideal_loads_units(text)
        loads = idf_tools.parse_internal_loads(text)
        self.lights = loads["lights"]
        self.equips = loads["equips"]
        self.out_dir = Path(ep["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_idf = self.out_dir / "in.idf"
        self.run_idf.write_text(text, encoding="utf-8")
        self.epw_fc = idf_tools.parse_epw(self.epw, self.year)

        # subprocess bridge state
        self._proc: subprocess.Popen | None = None
        self._stderr_fh = None
        self._i = 0
        self._last_action: dict = {}
        self._current_dt = self._start
        self.cum_kwh = self.cum_cost = self.cum_co2 = 0.0
        self.cum_hvac = self.cum_lights = self.cum_plugs = 0.0
        self._err_path = self.out_dir / "eplusout.err"
        self._worker_log_path = self.out_dir / "worker_stderr.log"

    # -- SimBackend interface -------------------------------------------------
    @property
    def zone_names(self) -> list[str]:
        return self.zones

    @property
    def n_steps(self) -> int:
        return self._n_steps

    @property
    def timestep_seconds(self) -> int:
        return self._dt

    def _dt_from_msg(self, msg: dict) -> datetime:
        return datetime(self.year, msg["month"], msg["day"],
                        max(0, min(23, msg["hour"])), max(0, min(59, int(msg["minute"]) % 60)))

    def _build_obs(self, temps: dict, t_out: float, step_kwh: float,
                   applied_action: dict, kwh_hvac: float = 0.0,
                   kwh_lights: float = 0.0, kwh_plugs: float = 0.0) -> Observation:
        dt = self._current_dt
        occ = _occupancy(dt)
        occupied = occ > 0.10
        pmv = {z: self.comfort.pmv(temps[z]) for z in self.zones}
        co2 = {z: 420.0 + occ * 350.0 for z in self.zones}   # nominal IAQ indicator
        violations = sum(1 for z in self.zones
                         if self.comfort.is_violation(temps[z], occupied, co2[z]))
        price = self.tariff.price(dt)
        carbon = self.tariff.carbon(dt)
        step_cost = step_kwh * price
        step_co2 = step_kwh * carbon / 1000.0
        self._i += 1
        self.cum_kwh += step_kwh
        self.cum_cost += step_cost
        self.cum_co2 += step_co2
        self.cum_hvac += kwh_hvac
        self.cum_lights += kwh_lights
        self.cum_plugs += kwh_plugs
        return Observation(
            index=self._i, time=dt, outdoor_temp=t_out, occupancy=occ,
            zone_temps=temps, zone_pmv=pmv, zone_co2=co2,
            step_kwh=step_kwh, cumulative_kwh=self.cum_kwh,
            price=price, carbon=carbon, step_cost=step_cost, cumulative_cost=self.cum_cost,
            step_co2_kg=step_co2, cumulative_co2_kg=self.cum_co2,
            comfort_violations=violations, applied_action=applied_action,
            step_kwh_hvac=kwh_hvac, step_kwh_lights=kwh_lights, step_kwh_plugs=kwh_plugs,
            cumulative_kwh_hvac=self.cum_hvac, cumulative_kwh_lights=self.cum_lights,
            cumulative_kwh_plugs=self.cum_plugs,
        )

    def _spawn(self) -> None:
        spec = {
            "install_dir": self.install_dir, "idf": str(self.run_idf), "epw": self.epw,
            "out_dir": str(self.out_dir), "zones": self.zones, "units": self.units,
            "lights": self.lights, "equips": self.equips,
        }
        worker = Path(__file__).with_name("ep_worker.py")
        self._stderr_fh = open(self._worker_log_path, "w", encoding="utf-8")
        popen_kwargs = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self._proc = subprocess.Popen(
            [sys.executable, str(worker), json.dumps(spec)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr_fh,
            text=True, bufsize=1, **popen_kwargs,
        )

    def _read_obs(self) -> Observation | None:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            return None
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            return None
        if msg.get("event") == "end":
            return None
        self._current_dt = self._dt_from_msg(msg)
        cool_j = msg["cool_j"]
        heat_j = msg["heat_j"]
        # Ideal-loads report a thermal load; convert to electricity with the heat-pump
        # COPs and add fan power for any zone actively conditioning this step.
        elec_j = sum(cool_j) / self.cop_cool + sum(heat_j) / self.cop_heat
        n_active = sum(1 for c, h in zip(cool_j, heat_j) if c > 1.0 or h > 1.0)
        kwh_hvac = elec_j / 3.6e6 + self.fan_w / 1000.0 * (self._dt / 3600.0) * n_active
        # Lighting and plug loads are metered directly by EnergyPlus.
        kwh_lights = msg.get("light_j", 0.0) / 3.6e6
        kwh_plugs = msg.get("plug_j", 0.0) / 3.6e6
        return self._build_obs(msg["zones"], msg["t_out"],
                               kwh_hvac + kwh_lights + kwh_plugs, self._last_action,
                               kwh_hvac, kwh_lights, kwh_plugs)

    def reset(self) -> Observation:
        self._spawn()
        obs = self._read_obs()
        if obs is None:
            raise RuntimeError(
                f"EnergyPlus produced no timesteps. See {self._err_path} and "
                f"{self._worker_log_path}")
        return obs

    def step(self, action: Action) -> Observation:
        if self._proc is None or self._proc.stdin is None:
            raise StopIteration
        try:
            self._proc.stdin.write(json.dumps({
                "cool_sp": action.cooling_setpoint, "heat_sp": action.heating_setpoint,
                "light": action.light_level, "plug": action.plug_level,
                "vent": action.vent_level,
            }) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            raise StopIteration from None
        self._last_action = action.as_dict()
        obs = self._read_obs()
        if obs is None:
            if self._i < self._n_steps:
                sys.stderr.write(
                    f"[energyplus_backend] WARNING: run ended early at step "
                    f"{self._i}/{self._n_steps}. See {self._worker_log_path}\n")
            raise StopIteration
        return obs

    def current_exogenous(self) -> Exogenous:
        return self.epw_fc.at(self._current_dt)

    def forecast(self, n: int) -> list[Exogenous]:
        return self.epw_fc.forecast(self._current_dt, n, self._dt)

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._stderr_fh:
            try:
                self._stderr_fh.close()
            except OSError:
                pass
        self._proc = None

    def err_summary(self) -> str:
        return idf_tools.summarize_err(str(self._err_path))
