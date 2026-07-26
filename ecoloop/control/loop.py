"""The closed-loop coordinator — the beating heart of Eco-Loop.

    reset  ->  [ feedback: obs ]  ->  decide (baseline schedule OR LLM supervisor)
            ->  safety guard  ->  forward-inject action  ->  step sim  ->  repeat

Implements the supervisory cadence that makes an LLM-in-the-loop practical: the
controller is consulted every ``decide_every`` steps and the chosen setpoints are held
between consultations, while the physics advances every timestep. This is exactly how a
real supervisory MPC layer sits on top of a fast local BMS.
"""

from __future__ import annotations

from typing import Any, Callable

from ..backends.base import SimBackend
from ..telemetry import Recorder
from .safety import SafetyGuard


def run_loop(
    backend: SimBackend,
    controller: Any,
    recorder: Recorder,
    decide_every: int = 1,
    safety: SafetyGuard | None = None,
    verbose: bool = False,
    progress_cb: Callable[[int, int], None] | None = None,
    live: Any | None = None,
) -> dict[str, Any]:
    """Drive a full closed-loop simulation and return the run summary."""
    obs = backend.reset()
    recorder.record(obs, {"source": "init"})
    last_action = None
    step = 0
    day = _DayTracker(obs)

    while True:
        occupied = obs.occupancy > 0.10
        if last_action is None or step % decide_every == 0:
            action, meta = controller.decide(obs, backend)
        else:
            action, meta = last_action, {"source": "held"}

        if safety is not None:
            action, notes = safety.enforce(action, obs, occupied,
                                           arriving_soon=_arriving_soon(backend))
            if notes:
                meta = {**meta, "safety": notes}
        last_action = action

        if verbose and meta.get("source") in ("llm", "fallback"):
            aa = action.as_dict()
            tag = meta["source"].upper()
            extra = f" [{meta['latency_s']}s]" if meta.get("latency_s") else ""
            print(f"  {obs.time:%a %H:%M} {tag}{extra} -> cool={aa['cooling_setpoint']} "
                  f"heat={aa['heating_setpoint']} econ={aa['economizer']} :: {aa['reason']}")

        try:
            obs = backend.step(action)
        except StopIteration:
            break

        recorder.record(obs, meta)
        if live is not None:
            live.push(obs, meta)
        step += 1
        if progress_cb and step % 20 == 0:
            progress_cb(step, backend.n_steps)

        # Day boundary: let a self-critiquing controller review the day it just finished.
        finished = day.update(obs)
        if finished is not None and hasattr(controller, "on_new_day"):
            if verbose:
                print(f"  --- day {finished['day']} done: {finished['kwh']:.1f} kWh, "
                      f"{finished['violations']} comfort violations -> self-critique ---")
            before = len(getattr(controller, "reflections", []))
            controller.on_new_day(obs, backend, finished)
            if live is not None:
                for entry in getattr(controller, "reflections", [])[before:]:
                    live.push_reflection(entry)

    return recorder.summary()


def _arriving_soon(backend: SimBackend, minutes: int = 30) -> bool:
    """True if anyone shows up within the next half hour.

    Used only by the safety guard, to bring lighting and ventilation up *before* the
    first people walk in rather than a timestep after.
    """
    try:
        n = max(1, (minutes * 60) // backend.timestep_seconds)
        return any(e.occ > 0.02 for e in backend.forecast(n + 1))
    except Exception:  # noqa: BLE001 - a forecast problem must not break control
        return False


class _DayTracker:
    """Accumulates per-simulated-day energy and comfort, and detects day rollovers."""

    def __init__(self, obs) -> None:
        self._date = obs.time.date()
        self._kwh0 = obs.cumulative_kwh
        self._violations = 0
        self._pmv_sum = 0.0
        self._pmv_n = 0
        self._occ_zone_steps = 0

    def update(self, obs) -> dict | None:
        """Fold in this step; return the finished day's stats when the date rolls over."""
        if obs.occupancy > 0.10:
            self._violations += obs.comfort_violations
            self._occ_zone_steps += len(obs.zone_temps)
            for p in obs.zone_pmv.values():
                self._pmv_sum += p
                self._pmv_n += 1

        if obs.time.date() == self._date:
            return None

        stats = {
            "day": self._date.isoformat(),
            "kwh": obs.cumulative_kwh - self._kwh0,
            "violations": self._violations,
            "occupied_zone_steps": self._occ_zone_steps,
            "mean_pmv": self._pmv_sum / self._pmv_n if self._pmv_n else 0.0,
        }
        self._date = obs.time.date()
        self._kwh0 = obs.cumulative_kwh
        self._violations = 0
        self._pmv_sum = 0.0
        self._pmv_n = 0
        self._occ_zone_steps = 0
        return stats
