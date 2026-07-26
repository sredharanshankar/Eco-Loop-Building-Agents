"""Fault-injection / chaos harness — evidence for the System Integration criterion.

Rather than *claiming* the closed loop is robust, this deliberately breaks things and
asserts the loop still completes and stays comfort-safe. Every scenario below models a
failure that actually happens in practice (and several that actually happened during
this project's development):

  1. llm_unreachable      — the Ollama endpoint is down for the whole run.
  2. llm_timeout_storm    — the LLM answers, but every call times out.
  3. malformed_tool_args  — the model returns junk/invalid tool arguments.
  4. adversarial_actions  — the controller emits NaN/inf/inverted/absurd setpoints.
  5. ep_worker_killed     — the EnergyPlus worker process is killed mid-run.

Pass criteria per scenario: no unhandled exception, the run produces telemetry, and
occupied thermal comfort is never worse than the safety guard permits.

Run:  python chaos_test.py            (RC backend, fast, no EnergyPlus needed)
      python chaos_test.py --with-ep  (also runs the EnergyPlus worker-kill scenario)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ecoloop.agent.supervisor import AgentSupervisor
from ecoloop.backends.base import Action
from ecoloop.backends.rc_backend import RCBackend
from ecoloop.comfort import ComfortSpec
from ecoloop.config import load_config
from ecoloop.control.loop import run_loop
from ecoloop.control.safety import SafetyGuard
from ecoloop.telemetry import Recorder

RESULTS = Path("results")


def _cfg(days: int = 2) -> dict:
    cfg = load_config()
    cfg["run"]["days"] = days
    return cfg


def _decide_every(cfg: dict) -> int:
    return max(1, round(cfg["agent"]["control_interval_minutes"] / cfg["run"]["timestep_minutes"]))


def _comfort_check(cfg: dict, rec: Recorder) -> tuple[bool, str]:
    """Occupied comfort must never be catastrophically violated, even under fault."""
    summ = rec.summary()
    spec = ComfortSpec.from_config(cfg)
    ok_rate = summ["comfort_ok_rate"]
    max_pmv = summ["max_abs_pmv_occupied"]
    # The guard permits transient excursions (e.g. morning warm-up); it must never allow
    # a sustained runaway. 1.5 PMV is the "clearly unacceptable" threshold.
    if max_pmv > 1.5:
        return False, f"max occupied |PMV| {max_pmv} exceeded 1.5"
    if ok_rate < 0.70:
        return False, f"occupied comfort-OK rate {ok_rate:.2f} below 0.70"
    return True, f"comfort-OK {ok_rate*100:.1f}%, max |PMV| {max_pmv}"


# -- fault injectors ----------------------------------------------------------
class _BrokenLLM:
    """Stands in for LLMClient. mode: 'unreachable' | 'timeout' | 'malformed'."""

    def __init__(self, mode: str):
        self.mode = mode
        self.calls = 0

    def health(self) -> bool:
        return self.mode != "unreachable"

    def chat(self, messages, tools=None, tool_choice="auto"):
        from ecoloop.agent.llm_client import LLMResult
        self.calls += 1
        if self.mode in ("unreachable", "timeout"):
            return LLMResult(False, None, 0.01,
                             "APIConnectionError: injected fault" if self.mode == "unreachable"
                             else "APITimeoutError: injected fault")
        # malformed: a well-formed response carrying junk tool arguments
        return LLMResult(True, _JunkMessage(), 0.01)


class _JunkFunction:
    name = "set_control"
    arguments = '{"strategy": "obliterate_the_building", "new_value": '   # truncated junk


class _JunkToolCall:
    id = "call_junk"
    function = _JunkFunction()


class _JunkMessage:
    content = "here you go"
    tool_calls = [_JunkToolCall()]


class _AdversarialController:
    """Emits values designed to break naive downstream code, every single step."""

    name = "adversarial"
    _VALUES = [
        (float("nan"), float("nan")),
        (float("inf"), float("-inf")),
        (99.0, -99.0),          # absurd magnitudes
        (10.0, 40.0),           # inverted: heating far above cooling
        (22.0, 22.0),           # zero deadband
        (None, None),           # wrong type entirely
    ]
    # The non-HVAC levers are the ones that can make a building unusable, so they get
    # the same abuse: negative, over-unity, non-finite and wrong-typed values.
    _LEVELS = [
        (-5.0, -5.0, -5.0),
        (99.0, 99.0, 99.0),
        (float("nan"), float("inf"), float("-inf")),
        (0.0, 0.0, 0.0),        # pitch dark, no equipment, no fresh air
        (None, "off", None),    # wrong types entirely
    ]

    def __init__(self):
        self.i = 0

    def decide(self, obs, backend):
        heat, cool = self._VALUES[self.i % len(self._VALUES)]
        light, plug, vent = self._LEVELS[self.i % len(self._LEVELS)]
        self.i += 1
        try:
            action = Action(heating_setpoint=heat, cooling_setpoint=cool,
                            economizer=True, reason="adversarial injection",
                            light_level=light, plug_level=plug, vent_level=vent)
        except Exception:
            action = Action(21.0, 25.0, False, "adversarial fallback")
        return action, {"source": "fallback"}


# -- scenarios ----------------------------------------------------------------
def scenario_llm_fault(mode: str) -> dict:
    cfg = _cfg()
    backend = RCBackend(cfg)
    sup = AgentSupervisor(cfg)
    sup.llm = _BrokenLLM(mode)          # inject the fault
    rec = Recorder(f"chaos_{mode}", cfg, backend.name, "agent")
    run_loop(backend, sup, rec, decide_every=_decide_every(cfg), safety=SafetyGuard(cfg))
    backend.close()
    summ = rec.summary()
    ok, detail = _comfort_check(cfg, rec)
    stats = sup.run_stats()
    completed = summ["steps"] >= backend.n_steps
    return {
        "scenario": f"llm_{mode}",
        "passed": bool(ok and completed and summ["total_kwh"] > 0),
        "steps_completed": summ["steps"], "steps_expected": backend.n_steps,
        "fallback_decisions": stats["decisions"]["fallback"],
        "total_kwh": summ["total_kwh"], "comfort": detail,
        "note": "loop completed on the deterministic fallback controller",
    }


def scenario_adversarial_actions() -> dict:
    cfg = _cfg()
    backend = RCBackend(cfg)
    rec = Recorder("chaos_adversarial", cfg, backend.name, "adversarial")
    guard = SafetyGuard(cfg)
    run_loop(backend, _AdversarialController(), rec, decide_every=1, safety=guard)
    backend.close()
    summ = rec.summary()
    ok, detail = _comfort_check(cfg, rec)
    # Every actuated setpoint must have landed inside the hard envelope.
    a = cfg["agent"]
    bad = []
    for row in rec.rows:
        c, h = row.get("cooling_setpoint"), row.get("heating_setpoint")
        if c is None or h is None:
            continue
        if not (a["min_cool_sp_c"] - 0.01 <= c <= a["max_cool_sp_c"] + 0.01):
            bad.append(("cool", c))
        if not (a["min_heat_sp_c"] - 0.01 <= h <= a["max_heat_sp_c"] + 0.01):
            bad.append(("heat", h))
        if isinstance(c, float) and (math.isnan(c) or math.isinf(c)):
            bad.append(("cool_nonfinite", c))
        # Non-HVAC levers must stay in [0,1], and never strand occupants in a dark or
        # unventilated building however absurd the requested value was.
        occ = row.get("occupancy", 0) or 0
        for name, floor in (("light_level", a["min_light_level_occupied"]),
                            ("plug_level", a["min_plug_level_occupied"]),
                            ("vent_level", a["min_vent_level_occupied"])):
            v = row.get(name)
            if v is None or v == "":
                continue
            v = float(v)
            if not (0.0 <= v <= 1.0) or math.isnan(v):
                bad.append((name + "_range", v))
            if occ > 0.02 and v < floor - 0.01:
                bad.append((name + "_floor", v))
    return {
        "scenario": "adversarial_actions",
        "passed": bool(ok and not bad and summ["steps"] >= backend.n_steps),
        "steps_completed": summ["steps"], "steps_expected": backend.n_steps,
        "envelope_violations": len(bad), "examples": bad[:3],
        "comfort": detail,
        "note": "SafetyGuard clamped NaN/inf/inverted/absurd setpoints every step",
    }


def scenario_malformed_tool_args() -> dict:
    """The registry must reject junk arguments as a tool error, never raise."""
    from ecoloop.agent.tools import ToolContext, ToolRegistry
    from ecoloop.signals import TariffCarbon

    cfg = _cfg(days=1)
    backend = RCBackend(cfg)
    obs = backend.reset()
    ctx = ToolContext(obs, backend, TariffCarbon(cfg), ComfortSpec.from_config(cfg),
                      cfg, True)
    reg = ToolRegistry(ctx)
    cases = [
        ("set_control", {"strategy": "obliterate_the_building", "reason": "junk"}),
        ("set_control", {}),
        ("set_control", {"strategy": None, "reason": 12345}),
        ("evaluate_setpoint", {"cooling_setpoint": "not a number"}),
        ("nonexistent_tool", {"x": 1}),
        ("propose_policy_tweak", {"parameter": "nope", "new_value": 5}),
    ]
    errors_returned, raised = 0, 0
    for name, args in cases:
        try:
            result, action = reg.dispatch(name, args)
            if "error" in result or result.get("accepted") is False:
                errors_returned += 1
        except Exception:
            raised += 1
    backend.close()
    return {
        "scenario": "malformed_tool_args",
        "passed": raised == 0 and errors_returned == len(cases),
        "cases": len(cases), "handled_as_tool_error": errors_returned, "exceptions_raised": raised,
        "note": "invalid arguments return a structured tool error the model can self-correct on",
    }


def scenario_policy_fuzz() -> dict:
    """Fuzz the bounded policy: no input may push a knob outside its declared range."""
    from ecoloop.agent.policy import AdjustablePolicy

    cfg = _cfg(days=1)
    pol = AdjustablePolicy.from_config(cfg)
    rng = random.Random(7)
    escapes, raised = 0, 0
    payloads = [float("nan"), float("inf"), float("-inf"), 1e9, -1e9, 0.0, None, "hot"]
    for i in range(400):
        name = rng.choice(pol.knob_names() + ["bogus_knob"])
        val = rng.choice(payloads) if i % 4 == 0 else rng.uniform(-500, 500)
        try:
            pol.propose(name, val, "2026-07-20", "fuzz")
        except Exception:
            raised += 1
            continue
        for kn in pol.knob_names():
            knob = getattr(pol, kn)
            v = knob.value
            if not (isinstance(v, float) and math.isfinite(v) and knob.lo - 1e-9 <= v <= knob.hi + 1e-9):
                escapes += 1
    return {
        "scenario": "policy_fuzz",
        "passed": escapes == 0 and raised == 0,
        "iterations": 400, "bound_escapes": escapes, "exceptions_raised": raised,
        "final_policy": pol.describe(),
        "note": "self-critique knobs stayed inside their safe bounds under adversarial input",
    }


def scenario_ep_worker_killed(kill_after: int = 40) -> dict:
    """Kill the EnergyPlus worker mid-run; the controller process must survive.

    The kill is triggered deterministically after a fixed number of steps (NOT on a
    wall-clock timer) so the fault is guaranteed to actually fire — an earlier
    timer-based version raced the simulation and passed without ever killing anything.
    """
    from ecoloop.backends.energyplus_backend import EnergyPlusBackend

    cfg = _cfg(days=2)
    backend = EnergyPlusBackend(cfg)
    rec = Recorder("chaos_ep_kill", cfg, backend.name, "baseline")
    obs = backend.reset()
    rec.record(obs, {"source": "init"})
    guard = SafetyGuard(cfg)

    total_steps = backend.n_steps
    killed = False
    steps, crashed = 0, None
    try:
        while True:
            action, _ = guard.enforce(Action(21.0, 25.0, False, "chaos"), obs, True)
            try:
                obs = backend.step(action)
            except StopIteration:
                break
            rec.record(obs, {"source": "baseline"})
            steps += 1
            if steps == kill_after:
                proc = backend._proc
                if proc and proc.poll() is None:
                    proc.kill()          # hard-kill the native EnergyPlus process
                    killed = True
    except Exception as exc:  # noqa: BLE001 - this is exactly what must NOT happen
        crashed = f"{type(exc).__name__}: {exc}"
    finally:
        backend.close()

    # The fault must have actually fired AND stopped the run early, and the controller
    # process must have survived it without an unhandled exception.
    stopped_early = steps < total_steps
    return {
        "scenario": "ep_worker_killed",
        "passed": bool(crashed is None and killed and stopped_early),
        "worker_killed": killed,
        "steps_before_stop": steps, "steps_expected_if_healthy": total_steps,
        "stopped_early_as_expected": stopped_early,
        "controller_crashed": crashed,
        "note": f"worker hard-killed at step {kill_after}; parent survived and ended the "
                f"loop cleanly via StopIteration",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Eco-Loop chaos / fault-injection harness")
    ap.add_argument("--with-ep", action="store_true",
                    help="also run the EnergyPlus worker-kill scenario (slower)")
    args = ap.parse_args()

    scenarios = [
        ("llm_unreachable", lambda: scenario_llm_fault("unreachable")),
        ("llm_timeout_storm", lambda: scenario_llm_fault("timeout")),
        ("malformed_tool_args", scenario_malformed_tool_args),
        ("adversarial_actions", scenario_adversarial_actions),
        ("policy_fuzz", scenario_policy_fuzz),
    ]
    if args.with_ep:
        scenarios.append(("ep_worker_killed", scenario_ep_worker_killed))

    print("=" * 72)
    print("  ECO-LOOP CHAOS HARNESS — deliberately breaking the closed loop")
    print("=" * 72)
    results = []
    for name, fn in scenarios:
        print(f"\n[chaos] injecting fault: {name} ...")
        t0 = time.perf_counter()
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001 - a harness-level crash is itself a failure
            res = {"scenario": name, "passed": False,
                   "controller_crashed": f"{type(exc).__name__}: {exc}"}
        res["duration_s"] = round(time.perf_counter() - t0, 1)
        results.append(res)
        print(f"        {'PASS' if res['passed'] else 'FAIL'}  ({res['duration_s']}s)  "
              f"{res.get('note', res.get('controller_crashed', ''))}")

    passed = sum(1 for r in results if r["passed"])
    print("\n" + "=" * 72)
    print(f"  RESULT: {passed}/{len(results)} fault-injection scenarios survived")
    print("=" * 72)
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['scenario']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "chaos_report.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"passed": passed, "total": len(results), "scenarios": results}, fh, indent=2)
    print(f"\n  Report: {out}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
