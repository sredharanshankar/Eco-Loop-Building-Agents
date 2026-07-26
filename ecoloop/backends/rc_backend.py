"""RC (resistance-capacitance) thermal backend.

A fast, deterministic, physics-based digital twin used for development, CI, and as a
graceful fallback when EnergyPlus is unavailable. Each zone is a lumped-capacitance
node driven by conduction, orientation-aware solar gains, occupancy-driven internal
gains and an ideal-loads heat pump. It deliberately mirrors the EnergyPlus data model
(zone air temperatures, setpoint schedules, electricity meter, CO2) so the agent and
controller behave identically on both backends.
"""

from __future__ import annotations

import math

from ..comfort import ComfortSpec
from ..signals import TariffCarbon
from ..weather import Exogenous, SyntheticWeather
from .base import Action, Observation, SimBackend

SHGC = 0.40
WALL_HEIGHT = 3.0
DIFFUSE_FRAC = 0.30
CO2_OUTDOOR = 420.0
CEILING_H = 3.0
RHO_CP = 1.2 * 1005.0          # volumetric heat capacity of air [J/(m3.K)]
ACH_FREE = 6.0                 # boosted air changes/hour when economizer free-cooling


class RCBackend(SimBackend):
    name = "rc"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        b = cfg["building"]
        self.zones = list(b["zones"])
        self.area = dict(zip(self.zones, b["zone_area_m2"]))
        self.orient = dict(zip(self.zones, b["orientation_deg"]))
        self.UA = dict(zip(self.zones, b["UA_env_w_per_k"]))
        self.C = dict(zip(self.zones, b["C_zone_j_per_k"]))
        self.wfrac = dict(zip(self.zones, b["window_frac"]))
        self.int_peak = dict(zip(self.zones, b["internal_peak_w"]))
        self.people = dict(zip(self.zones, b["people_peak"]))
        self.cop_cool = b["cop_cool"]
        self.cop_heat = b["cop_heat"]
        self.fan_w = b["fan_kw"] * 1000.0
        self.econ_fan_w = b["fan_kw"] * 1000.0 * 0.25   # low-power night-flush/relief fan
        self.vent_ach = b["vent_ach"]
        self.co2_gen = b["co2_gen_per_person_m3s"]
        self.cap = {z: b["hvac_capacity_kw"] * 1000.0 * max(1.0, self.area[z] / 70.0)
                    for z in self.zones}

        self.weather = SyntheticWeather(cfg)
        self.tariff = TariffCarbon(cfg)
        self.comfort = ComfortSpec.from_config(cfg)
        self._n_steps = self.weather.n_steps
        self._dt = self.weather.step_seconds

        self.i = 0
        self.temps: dict[str, float] = {}
        self.co2: dict[str, float] = {}
        self.cum_kwh = self.cum_cost = self.cum_co2 = 0.0
        self.cum_hvac = self.cum_lights = self.cum_plugs = 0.0
        self._split = (0.0, 0.0, 0.0)   # (hvac, lights, plugs) kWh for the last interval

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

    def reset(self) -> Observation:
        self.i = 0
        self.temps = {z: 24.0 for z in self.zones}
        self.co2 = {z: 450.0 for z in self.zones}
        self.cum_kwh = self.cum_cost = self.cum_co2 = 0.0
        self.cum_hvac = self.cum_lights = self.cum_plugs = 0.0
        self._split = (0.0, 0.0, 0.0)
        exo = self.weather.at(0)
        return self._observe(exo, exo, step_kwh=0.0, action=None)

    def step(self, action: Action) -> Observation:
        if self.i >= self._n_steps:
            raise StopIteration
        exo = self.weather.at(self.i)
        step_kwh = self._simulate_interval(action, exo)
        self.i += 1
        exo_next = self.weather.at(self.i)
        return self._observe(exo, exo_next, step_kwh, action)

    def current_exogenous(self) -> Exogenous:
        return self.weather.at(self.i)

    def forecast(self, n: int) -> list[Exogenous]:
        return self.weather.forecast(self.i, n)

    # -- physics --------------------------------------------------------------
    def _solar_on_facade(self, zone: str, exo: Exogenous) -> float:
        """Approximate solar heat gain through a zone's windows [W]."""
        if exo.solar_ghi <= 0.0 or self.wfrac[zone] <= 0.0:
            return 0.0
        h = exo.time.hour + exo.time.minute / 60.0
        sun_az = 180.0 + (h - 12.0) * 15.0            # deg from north
        direct = max(0.0, math.cos(math.radians(sun_az - self.orient[zone])))
        wall_area = math.sqrt(self.area[zone]) * WALL_HEIGHT
        window_area = wall_area * self.wfrac[zone]
        return exo.solar_ghi * window_area * SHGC * (DIFFUSE_FRAC + 0.70 * direct)

    def _simulate_interval(self, action: Action, exo: Exogenous) -> float:
        dt_sub = min(300, self._dt)
        n_sub = max(1, self._dt // dt_sub)
        heat_sp = action.heating_setpoint
        cool_sp = action.cooling_setpoint
        econ = action.economizer
        step_kwh = 0.0

        # Lighting and plug loads: the agent can trim them, which saves their electricity
        # directly AND removes the heat they dump into the zone, cutting cooling too.
        lvl_light = max(0.0, min(1.0, action.light_level))
        lvl_plug = max(0.0, min(1.0, action.plug_level))
        step_kwh_lights = step_kwh_plugs = 0.0

        for zone in self.zones:
            T = self.temps[zone]
            C = self.C[zone]
            UA = self.UA[zone]
            cap = self.cap[zone]
            vol = self.area[zone] * CEILING_H
            ua_free = ACH_FREE * vol / 3600.0 * RHO_CP       # W/K of free-cooling airflow
            q_solar = self._solar_on_facade(zone, exo)
            # Internal gains split evenly between lighting and equipment, each scalable.
            scheduled_int = (0.15 + 0.85 * exo.occ) * self.int_peak[zone]
            w_lights = 0.5 * scheduled_int * lvl_light
            w_plugs = 0.5 * scheduled_int * lvl_plug
            q_int = w_lights + w_plugs
            hours = self._dt / 3600.0
            step_kwh_lights += w_lights / 1000.0 * hours
            step_kwh_plugs += w_plugs / 1000.0 * hours
            for _ in range(n_sub):
                q_base = UA * (exo.t_out - T) + q_solar + q_int
                # Economizer: ventilative free cooling with cool outdoor air (fan-only),
                # capped so it never drives the zone below the heating setpoint.
                q_econ = 0.0
                if econ and exo.t_out < T - 0.5:
                    q_econ = ua_free * (exo.t_out - T)
                    t_after = T + (q_base + q_econ) / C * dt_sub
                    if t_after < heat_sp:
                        q_econ = min(0.0, C * (heat_sp - T) / dt_sub - q_base)
                econ_active = q_econ < -1.0
                q_env = q_base + q_econ
                t_free = T + q_env / C * dt_sub

                q_hvac = 0.0
                comp_w = 0.0
                if t_free > cool_sp:                         # mechanical cooling
                    q_hvac = max(C * (cool_sp - t_free) / dt_sub, -cap)
                    comp_w = (-q_hvac) / self.cop_cool
                elif t_free < heat_sp:                       # heating
                    q_hvac = min(C * (heat_sp - t_free) / dt_sub, cap)
                    comp_w = q_hvac / self.cop_heat
                if q_hvac != 0.0:
                    fan = self.fan_w                      # full AHU supply fan
                elif econ_active:
                    fan = self.econ_fan_w                 # low-power free-cooling path
                else:
                    fan = 0.0
                T = T + (q_env + q_hvac) / C * dt_sub
                step_kwh += (comp_w + fan) / 1000.0 * (dt_sub / 3600.0)
            self.temps[zone] = T

            # CO2 mass balance over the full timestep. Throttling ventilation saves the
            # energy of conditioning outdoor air, but lets CO2 build up — the safety
            # guard is what stops that becoming an air-quality problem.
            lvl_vent = max(0.1, min(1.0, action.vent_level))
            ach = self.vent_ach * lvl_vent * (3.0 if econ else 1.0)
            vol = self.area[zone] * CEILING_H
            q_vent = ach * vol / 3600.0                      # m3/s
            gen_ppm_s = self.people[zone] * exo.occ * self.co2_gen / vol * 1e6
            decay = math.exp(-q_vent / vol * self._dt)
            steady = CO2_OUTDOOR + gen_ppm_s * vol / max(q_vent, 1e-6)
            self.co2[zone] = steady + (self.co2[zone] - steady) * decay

        self._split = (step_kwh, step_kwh_lights, step_kwh_plugs)
        return step_kwh + step_kwh_lights + step_kwh_plugs

    # -- observation ----------------------------------------------------------
    def _observe(self, exo: Exogenous, exo_stamp: Exogenous,
                 step_kwh: float, action: Action | None) -> Observation:
        occupied = exo.occ > 0.10
        pmv = {z: self.comfort.pmv(self.temps[z]) for z in self.zones}
        violations = sum(
            1 for z in self.zones
            if self.comfort.is_violation(self.temps[z], occupied, self.co2[z])
        )
        price = self.tariff.price(exo.time)
        carbon = self.tariff.carbon(exo.time)
        step_cost = step_kwh * price
        step_co2 = step_kwh * carbon / 1000.0
        self.cum_kwh += step_kwh
        self.cum_cost += step_cost
        self.cum_co2 += step_co2
        hvac, lights, plugs = self._split
        self.cum_hvac += hvac
        self.cum_lights += lights
        self.cum_plugs += plugs
        return Observation(
            index=self.i,
            time=exo_stamp.time,
            outdoor_temp=exo.t_out,
            occupancy=exo.occ,
            zone_temps=dict(self.temps),
            zone_pmv=pmv,
            zone_co2=dict(self.co2),
            step_kwh=step_kwh,
            cumulative_kwh=self.cum_kwh,
            price=price,
            carbon=carbon,
            step_cost=step_cost,
            cumulative_cost=self.cum_cost,
            step_co2_kg=step_co2,
            cumulative_co2_kg=self.cum_co2,
            comfort_violations=violations,
            applied_action=action.as_dict() if action else {},
            step_kwh_hvac=hvac, step_kwh_lights=lights, step_kwh_plugs=plugs,
            cumulative_kwh_hvac=self.cum_hvac, cumulative_kwh_lights=self.cum_lights,
            cumulative_kwh_plugs=self.cum_plugs,
        )
