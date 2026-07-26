"""Bounded, adjustable ECM policy — the knobs the nightly self-critique loop may tune.

The strategy->setpoint mapping in ``tools.py`` used to be hardcoded constants. Pulling
the handful of tunable margins into this small, explicitly-bounded object is what lets
the agent's nightly reflection (``supervisor.py::_reflect``) safely adjust its own
behaviour: every knob carries a hard [lo, hi], so a proposed tweak can never leave the
range that was already comfort/energy sane by construction — a bad or adversarial
proposal is silently clamped, never rejected outright, never applied unbounded. The
PER-STEP SafetyGuard remains the final authority on every actuated setpoint regardless
of what the policy says; this only changes which setpoints the strategy table proposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

KNOB_NAMES = [
    "setpoint_reset_margin_c",
    "precool_margin_c",
    "deep_setback_cool_c",
    "deep_setback_heat_c",
]


@dataclass(frozen=True)
class Knob:
    value: float
    lo: float
    hi: float

    def __post_init__(self) -> None:
        # Never allow an inverted or out-of-range knob to exist: a lo > hi range makes
        # clamping meaningless and silently corrupts every later proposal.
        object.__setattr__(self, "hi", max(self.lo, self.hi))
        object.__setattr__(self, "value", max(self.lo, min(self.hi, self.value)))

    def clamped(self, new_value: float) -> "Knob":
        return replace(self, value=max(self.lo, min(self.hi, new_value)))


@dataclass
class AdjustablePolicy:
    """Named, bounded knobs read by ``ToolRegistry._strategy_setpoints``.

    Defaults (no-arg construction) reproduce the original hardcoded behaviour. Use
    :meth:`from_config` in production so bounds stay sensible relative to the actual
    configured baseline setback rather than fixed literals.
    """
    setpoint_reset_margin_c: Knob = field(default_factory=lambda: Knob(0.0, 0.0, 1.0))
    precool_margin_c: Knob = field(default_factory=lambda: Knob(1.5, 0.5, 3.0))
    deep_setback_cool_c: Knob = field(default_factory=lambda: Knob(28.0, 27.0, 30.0))
    deep_setback_heat_c: Knob = field(default_factory=lambda: Knob(16.0, 12.0, 17.0))
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict) -> "AdjustablePolicy":
        b = cfg["baseline"]
        a = cfg["agent"]
        # Knob ranges are clipped to the SafetyGuard's hard envelope. Outside it the guard
        # clamps anyway, so a wider range would let the agent "tune" into a region that can
        # never affect actuation — it would saturate and then waste nightly calls proposing
        # inert changes (observed on the 7-day run before this was tightened).
        cool_lo, cool_hi = a["min_cool_sp_c"], a["max_cool_sp_c"]
        heat_lo, heat_hi = a["min_heat_sp_c"], a["max_heat_sp_c"]
        # Deep setback wants the warmest cooling / coolest heating the envelope permits.
        setback_cool0 = min(max(b["setback_cool_sp"] + 2.0, 28.0), cool_hi)
        setback_heat0 = heat_lo
        return cls(
            setpoint_reset_margin_c=Knob(0.0, 0.0, 1.0),
            precool_margin_c=Knob(1.5, 0.5, 3.0),
            deep_setback_cool_c=Knob(setback_cool0, max(cool_lo, cool_hi - 2.0), cool_hi),
            deep_setback_heat_c=Knob(setback_heat0, heat_lo, min(heat_hi, heat_lo + 2.0)),
        )

    def get(self, name: str) -> float:
        return getattr(self, name).value

    def knob_names(self) -> list[str]:
        return list(KNOB_NAMES)

    def describe(self) -> str:
        return ", ".join(f"{n}={self.get(n):.2f}" for n in KNOB_NAMES)

    def describe_with_bounds(self) -> str:
        """Values plus their allowed range, flagging knobs already at a limit.

        Without this the agent cannot tell a knob is saturated and will keep proposing
        the same out-of-range change every night (observed: it asked for 30.5, then
        32.5, then 32.5 again while the knob sat pinned at its 30.0 ceiling).
        """
        parts = []
        for n in KNOB_NAMES:
            k: Knob = getattr(self, n)
            flag = ""
            if abs(k.value - k.hi) < 1e-9:
                flag = " [AT MAX — cannot increase further]"
            elif abs(k.value - k.lo) < 1e-9:
                flag = " [AT MIN — cannot decrease further]"
            parts.append(f"{n}={k.value:.2f} (allowed {k.lo:.1f}..{k.hi:.1f}){flag}")
        return "; ".join(parts)

    def propose(self, name: str, new_value: float, day: str, justification: str) -> dict[str, Any]:
        """Apply a bounded tweak to one knob; returns a log entry describing what happened."""
        if name not in KNOB_NAMES:
            return {"day": day, "parameter": name, "accepted": False,
                    "reason": f"unknown parameter '{name}', must be one of {KNOB_NAMES}",
                    "justification": justification}
        knob: Knob = getattr(self, name)
        try:
            clamped = knob.clamped(float(new_value))
        except (TypeError, ValueError):
            return {"day": day, "parameter": name, "accepted": False,
                    "reason": f"new_value {new_value!r} is not numeric",
                    "justification": justification}
        entry = {
            "day": day, "parameter": name,
            "requested_value": round(float(new_value), 3),
            "old_value": round(knob.value, 3),
            "applied_value": round(clamped.value, 3),
            "bounds": [knob.lo, knob.hi],
            "was_clamped": abs(clamped.value - float(new_value)) > 1e-9,
            "accepted": True,
            "justification": justification,
        }
        setattr(self, name, clamped)
        self.history.append(entry)
        return entry
