"""Eco-Loop Building Agents — unified CLI entrypoint.

Commands:
  python run.py baseline   [--backend rc|energyplus]
  python run.py agent      [--backend ...] [--verbose] [--no-llm]
  python run.py compare    [--backend ...] [--verbose]      # baseline + agent + dashboard
  python run.py dashboard                                   # rebuild dashboard from results/
  python run.py demo       [--backend ...]                  # short, verbose, for the video

Common options: --config PATH  --days N  --interval MIN  --model NAME  --results DIR
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from ecoloop.config import load_config
from ecoloop.control.baseline import BaselineController
from ecoloop.control.loop import run_loop
from ecoloop.control.safety import SafetyGuard
from ecoloop.telemetry import Recorder


def make_backend(name: str, cfg: dict):
    if name == "rc":
        from ecoloop.backends.rc_backend import RCBackend
        return RCBackend(cfg)
    if name in ("energyplus", "ep", "eplus"):
        from ecoloop.backends.energyplus_backend import EnergyPlusBackend
        return EnergyPlusBackend(cfg)
    raise SystemExit(f"unknown backend '{name}' (use rc or energyplus)")


def _decide_every(cfg: dict) -> int:
    return max(1, round(cfg["agent"]["control_interval_minutes"] / cfg["run"]["timestep_minutes"]))


def _progress(step: int, total: int) -> None:
    pct = 100 * step / total if total else 0
    sys.stdout.write(f"\r    ...{step}/{total} steps ({pct:.0f}%)")
    sys.stdout.flush()


def run_baseline(cfg: dict, backend_name: str, results_dir: str) -> dict:
    print(f"[baseline] backend={backend_name} ...")
    backend = make_backend(backend_name, cfg)
    ctrl = BaselineController(cfg)
    rec = Recorder("baseline", cfg, backend.name, "baseline")
    t0 = time.perf_counter()
    summ = run_loop(backend, ctrl, rec, decide_every=1, safety=None, progress_cb=_progress)
    backend.close()
    rec.save(results_dir)
    print(f"\n[baseline] {summ['total_kwh']} kWh, ${summ['total_cost_usd']}, "
          f"{summ['total_co2_kg']} kg CO2, comfort-OK {summ['comfort_ok_rate']*100:.1f}% "
          f"({time.perf_counter()-t0:.1f}s)")
    return summ


def run_agent(cfg: dict, backend_name: str, results_dir: str,
              verbose: bool = False, no_llm: bool = False,
              live_port: int | None = None, hold_live: bool = False) -> tuple[dict, dict]:
    from ecoloop.agent.supervisor import AgentSupervisor
    if no_llm:
        cfg = {**cfg, "agent": {**cfg["agent"], "disable_llm": True}}
    mode = "heuristic-only" if no_llm else f"LLM={cfg['agent']['model']}"
    print(f"[agent] backend={backend_name}, {mode}, "
          f"control every {cfg['agent']['control_interval_minutes']}min ...")
    backend = make_backend(backend_name, cfg)
    sup = AgentSupervisor(cfg)
    if not no_llm and not sup.llm.health():
        print("[agent] WARNING: LLM endpoint not reachable — falling back to heuristic.")
    rec = Recorder("agent", cfg, backend.name, "agent")
    safety = SafetyGuard(cfg)

    live_state = None
    if live_port:
        from ecoloop.live_dashboard import LiveState, start_live_server
        live_state = LiveState(backend.name, backend.n_steps, results_dir,
                               n_zones=len(backend.zone_names))
        start_live_server(live_state, live_port)

    t0 = time.perf_counter()
    summ = run_loop(backend, sup, rec, decide_every=_decide_every(cfg),
                    safety=safety, verbose=verbose,
                    progress_cb=None if verbose else _progress, live=live_state)
    backend.close()
    stats = sup.run_stats()
    if live_state is not None:
        live_state.finish(stats)
    rec.save(results_dir)
    with open(Path(results_dir) / "agent_runstats.json", "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    with open(Path(results_dir) / "agent_reflections.json", "w", encoding="utf-8") as fh:
        json.dump(sup.reflections, fh, indent=2)
    print(f"\n[agent] {summ['total_kwh']} kWh, ${summ['total_cost_usd']}, "
          f"{summ['total_co2_kg']} kg CO2, comfort-OK {summ['comfort_ok_rate']*100:.1f}% "
          f"({time.perf_counter()-t0:.1f}s)")
    print(f"[agent] {stats['llm_calls']} LLM calls, avg {stats['avg_latency_s']}s, "
          f"cache hit-rate {stats['hit_rate']*100:.0f}%, "
          f"decisions={stats['decisions']}")
    if sup.reflections:
        print(f"[agent] {len(sup.reflections)} nightly self-critiques; "
              f"final policy: {stats['final_policy']}")
        for r in sup.reflections:
            if r.get("accepted"):
                clamp = " (clamped)" if r.get("was_clamped") else ""
                print(f"         {r['day']}: {r['parameter']} "
                      f"{r['old_value']} -> {r['applied_value']}{clamp} :: {r['justification'][:70]}")

    if live_state is not None and hold_live:
        # Without this the process exits the instant the run ends and the dashboard goes
        # dead mid-demo. Hold the final state on screen until the presenter is done.
        print(f"\n[live] run complete — dashboard still serving the final state at "
              f"http://127.0.0.1:{live_port}\n[live] press Ctrl-C when you're done.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\n[live] dashboard stopped.")
    return summ, stats


def build_dashboard(cfg: dict, results_dir: str) -> None:
    from ecoloop.dashboard import make_dashboard
    stats_path = Path(results_dir) / "agent_runstats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    out = make_dashboard(cfg, results_dir, agent_stats=stats)
    sv = out["savings"]
    print("\n" + "=" * 60)
    print("  ECO-LOOP RESULTS")
    print("=" * 60)
    print(f"  Energy:  {sv['baseline_kwh']:.0f} -> {sv['ai_kwh']:.0f} kWh   "
          f"({sv['kwh_pct']:+.1f}%)")
    print(f"  Cost:    ${sv['baseline_cost']:.0f} -> ${sv['ai_cost']:.0f}   "
          f"({sv['cost_pct']:+.1f}%)")
    print(f"  Carbon:  {sv['baseline_co2']:.0f} -> {sv['ai_co2']:.0f} kg   "
          f"({sv['co2_pct']:+.1f}%)")
    print(f"  Peak:    {sv['baseline_peak_kw']:.1f} -> {sv['ai_peak_kw']:.1f} kW  "
          f"({sv['peak_pct']:+.1f}%)")
    print(f"  Comfort: {sv['baseline_comfort_ok']*100:.1f}% -> "
          f"{sv['ai_comfort_ok']*100:.1f}% occupied comfort-OK "
          f"{'(maintained)' if sv['comfort_maintained'] else '(DEGRADED)'}")
    print("=" * 60)
    print(f"  Dashboard: {out['html']}")


def _subrun(command: str, args, extra: list[str] | None = None) -> None:
    """Run one phase (baseline/agent) in its own OS process.

    EnergyPlus's C++ core keeps global state that is not safely reentrant within a
    single process: creating a second EnergyPlusAPI state after a prior run can
    corrupt memory (WinError 0xe06d7363 / SEH access violation), intermittently.
    Isolating every simulation phase in a fresh process sidesteps that entirely —
    each phase gets a brand-new library load, every time.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), command,
           "--backend", args.backend, "--results", args.results]
    if args.config:
        cmd += ["--config", args.config]
    if args.days:
        cmd += ["--days", str(args.days)]
    if args.interval:
        cmd += ["--interval", str(args.interval)]
    if args.model:
        cmd += ["--model", args.model]
    cmd += extra or []
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"\n[run.py] phase '{command}' failed (exit {exc.returncode})")
        raise SystemExit(exc.returncode) from None


def _write_replay_quietly(results_dir: str) -> None:
    """Refresh the interactive replay dashboard after a run (never fatal)."""
    try:
        from ecoloop.live_dashboard import write_replay
        print(f"  Replay:    {write_replay(results_dir)}")
    except Exception as exc:  # noqa: BLE001 - a presentation extra must not fail a run
        print(f"  (replay dashboard skipped: {exc})")


def apply_overrides(cfg: dict, args) -> dict:
    if args.days:
        cfg["run"]["days"] = args.days
    if args.interval:
        cfg["agent"]["control_interval_minutes"] = args.interval
    if args.model:
        cfg["agent"]["model"] = args.model
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Eco-Loop Building Agents")
    p.add_argument("command", choices=["baseline", "agent", "compare", "dashboard", "demo",
                                       "live-dashboard"])
    p.add_argument("--backend", default="rc", help="rc (default) or energyplus")
    p.add_argument("--config", default=None)
    p.add_argument("--results", default="results")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--interval", type=int, default=None, help="agent control interval (min)")
    p.add_argument("--model", default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-llm", action="store_true", help="run agent on heuristic policy only")
    p.add_argument("--live", action="store_true",
                   help="stream the running loop to a live dashboard in your browser")
    p.add_argument("--port", type=int, default=8765, help="live dashboard port")
    p.add_argument("--no-hold", action="store_true",
                   help="internal: don't keep the live dashboard open after the run "
                        "(set when this phase is a child of compare/demo, which must "
                        "continue on to build the dashboards)")
    args = p.parse_args()

    cfg = apply_overrides(load_config(args.config), args)
    Path(args.results).mkdir(parents=True, exist_ok=True)

    live_port = args.port if args.live else None

    if args.command == "baseline":
        run_baseline(cfg, args.backend, args.results)
    elif args.command == "agent":
        # Standalone agent runs hold the dashboard open afterwards (demo/video use).
        # `compare` must not block — it still has to build the dashboards.
        run_agent(cfg, args.backend, args.results, args.verbose, args.no_llm,
                  live_port, hold_live=bool(live_port) and not args.no_hold)
    elif args.command == "dashboard":
        build_dashboard(cfg, args.results)
    elif args.command == "live-dashboard":
        from ecoloop.live_dashboard import write_replay
        out = write_replay(args.results)
        print(f"Interactive replay dashboard written to: {out}")
        print("Open it in a browser and press play — no server needed.")
    elif args.command == "compare":
        _subrun("baseline", args)
        extra = (["--verbose"] if args.verbose else []) + (["--no-llm"] if args.no_llm else [])
        if args.live:
            # --no-hold: this phase is a child, the parent still has dashboards to build.
            extra += ["--live", "--port", str(args.port), "--no-hold"]
        _subrun("agent", args, extra)
        build_dashboard(cfg, args.results)
        _write_replay_quietly(args.results)
    elif args.command == "demo":
        args.days = args.days or 1
        cfg["run"]["days"] = args.days
        print("=== Eco-Loop live demo (short horizon, verbose) ===")
        _subrun("baseline", args)
        extra = ["--verbose"] + (["--no-llm"] if args.no_llm else [])
        if args.live:
            # --no-hold: this phase is a child, the parent still has dashboards to build.
            extra += ["--live", "--port", str(args.port), "--no-hold"]
        _subrun("agent", args, extra)
        build_dashboard(cfg, args.results)
        _write_replay_quietly(args.results)


if __name__ == "__main__":
    main()
