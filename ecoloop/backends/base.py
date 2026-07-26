"""The backend contract shared by the EnergyPlus and RC simulators.

The whole point of this abstraction is that the *agent, controller, telemetry, and
dashboard never know or care which simulator is running*. That is what lets us
develop and iterate in seconds on the RC model, then flip a single flag to drive the
identical loop through high-fidelity EnergyPlus.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..weather import Exogenous


@dataclass
class Action:
    """A supervisory control command, mapped directly onto EnergyPlus actuators.

    Temperature is only one lever. Lighting and plug loads are a large slice of a
    commercial building's electricity *and* they dump heat into the space, so trimming
    them saves twice — directly, and again in the cooling needed to remove that heat.
    Ventilation is the third: fresh air is expensive to condition, so it is worth
    matching to actual occupancy rather than running flat out.

      heating/cooling_setpoint -> Zone Temperature Control
      light_level              -> Lights | Electricity Rate       (fraction of design)
      plug_level               -> ElectricEquipment | Electricity Rate
      vent_level               -> Ideal Loads | Outdoor Air Mass Flow Rate
    """
    heating_setpoint: float
    cooling_setpoint: float
    economizer: bool = False
    reason: str = ""
    light_level: float = 1.0    # 0..1 fraction of scheduled lighting power
    plug_level: float = 1.0     # 0..1 fraction of scheduled equipment power
    vent_level: float = 1.0     # 0..1 fraction of design outdoor-air flow

    def as_dict(self) -> dict[str, Any]:
        return {
            "heating_setpoint": round(self.heating_setpoint, 2),
            "cooling_setpoint": round(self.cooling_setpoint, 2),
            "economizer": self.economizer,
            "reason": self.reason,
            "light_level": round(self.light_level, 3),
            "plug_level": round(self.plug_level, 3),
            "vent_level": round(self.vent_level, 3),
        }


@dataclass
class Observation:
    """One timestep of streamed feedback (EnergyPlus -> AI)."""
    index: int
    time: datetime
    outdoor_temp: float
    occupancy: float
    zone_temps: dict[str, float]
    zone_pmv: dict[str, float]
    zone_co2: dict[str, float]
    step_kwh: float
    cumulative_kwh: float
    price: float
    carbon: float
    step_cost: float
    cumulative_cost: float
    step_co2_kg: float
    cumulative_co2_kg: float
    comfort_violations: int
    applied_action: dict[str, Any] = field(default_factory=dict)
    # End-use split of step_kwh, so savings can be attributed rather than just totalled.
    step_kwh_hvac: float = 0.0
    step_kwh_lights: float = 0.0
    step_kwh_plugs: float = 0.0
    cumulative_kwh_hvac: float = 0.0
    cumulative_kwh_lights: float = 0.0
    cumulative_kwh_plugs: float = 0.0

    @property
    def mean_zone_temp(self) -> float:
        return sum(self.zone_temps.values()) / max(1, len(self.zone_temps))

    @property
    def max_abs_pmv(self) -> float:
        return max((abs(v) for v in self.zone_pmv.values()), default=0.0)

    @property
    def peak_co2(self) -> float:
        return max(self.zone_co2.values(), default=0.0)

    def to_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "index": self.index,
            "time": self.time.isoformat(),
            "outdoor_temp": round(self.outdoor_temp, 3),
            "occupancy": round(self.occupancy, 3),
            "step_kwh": round(self.step_kwh, 5),
            "cumulative_kwh": round(self.cumulative_kwh, 4),
            "price": round(self.price, 4),
            "carbon": round(self.carbon, 1),
            "step_cost": round(self.step_cost, 5),
            "cumulative_cost": round(self.cumulative_cost, 4),
            "step_co2_kg": round(self.step_co2_kg, 5),
            "cumulative_co2_kg": round(self.cumulative_co2_kg, 4),
            "comfort_violations": self.comfort_violations,
            "heating_setpoint": self.applied_action.get("heating_setpoint"),
            "cooling_setpoint": self.applied_action.get("cooling_setpoint"),
            "economizer": int(bool(self.applied_action.get("economizer", False))),
            "light_level": self.applied_action.get("light_level"),
            "plug_level": self.applied_action.get("plug_level"),
            "vent_level": self.applied_action.get("vent_level"),
            "mean_zone_temp": round(self.mean_zone_temp, 3),
            "max_abs_pmv": round(self.max_abs_pmv, 3),
            "peak_co2": round(self.peak_co2, 1),
            "step_kwh_hvac": round(self.step_kwh_hvac, 5),
            "step_kwh_lights": round(self.step_kwh_lights, 5),
            "step_kwh_plugs": round(self.step_kwh_plugs, 5),
            "cumulative_kwh_hvac": round(self.cumulative_kwh_hvac, 4),
            "cumulative_kwh_lights": round(self.cumulative_kwh_lights, 4),
            "cumulative_kwh_plugs": round(self.cumulative_kwh_plugs, 4),
        }
        for z, t in self.zone_temps.items():
            row[f"temp_{z}"] = round(t, 3)
        for z, p in self.zone_pmv.items():
            row[f"pmv_{z}"] = round(p, 3)
        for z, c in self.zone_co2.items():
            row[f"co2_{z}"] = round(c, 1)
        return row


class SimBackend(ABC):
    """Common interface for every simulator."""

    name: str = "abstract"

    @property
    @abstractmethod
    def zone_names(self) -> list[str]: ...

    @property
    @abstractmethod
    def n_steps(self) -> int: ...

    @property
    @abstractmethod
    def timestep_seconds(self) -> int: ...

    @abstractmethod
    def reset(self) -> Observation:
        """Initialise the simulation and return the first observation."""

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Apply the supervisory command for the current step, advance one timestep,
        and return the resulting feedback. Raises StopIteration when the horizon ends."""

    @abstractmethod
    def current_exogenous(self) -> Exogenous:
        """Weather/occupancy at the current step (for the agent's situational report)."""

    @abstractmethod
    def forecast(self, n: int) -> list[Exogenous]:
        """Upcoming exogenous conditions, for the agent's look-ahead tools."""

    def close(self) -> None:
        """Release any simulator resources (no-op by default)."""
