"""Eco-Loop MCP Server — a real Model Context Protocol server (stdio, JSON-RPC 2.0).

Dependency-free implementation of the MCP core (initialize / tools/list / tools/call)
so any MCP client — Claude Desktop, an IDE, or our own agent — can drive the building:
parse the model, read grid/comfort signals, triage EnergyPlus errors, and run the whole
closed loop to prove savings. This is the standardized-protocol surface the brief asks
for; the in-process ToolRegistry mirrors it for the latency-critical control loop.

Register with an MCP client, e.g. Claude Desktop config:
  { "mcpServers": { "eco-loop": { "command": "python",
      "args": ["F:/honeywell hackathon/mcp_server.py"] } } }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from ecoloop.backends import idf_tools
from ecoloop.comfort import pmv_ppd, ComfortSpec
from ecoloop.config import load_config
from ecoloop.signals import TariffCarbon

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "eco-loop", "version": "1.0.0"}

_cfg = load_config()
_tariff = TariffCarbon(_cfg)
_comfort = ComfortSpec.from_config(_cfg)


# -- tool implementations -----------------------------------------------------
def t_parse_building_model(idf_path: str) -> dict[str, Any]:
    text = Path(idf_path).read_text(encoding="utf-8", errors="ignore")
    return {
        "controlled_zones": idf_tools.parse_controlled_zones(text),
        "ideal_loads_units": idf_tools.parse_ideal_loads_units(text),
    }


def t_summarize_simulation_log(err_path: str, max_lines: int = 20) -> dict[str, Any]:
    return {"summary": idf_tools.summarize_err(err_path, max_lines)}


def t_evaluate_comfort(temp_c: float, rh: float = 50.0) -> dict[str, Any]:
    pmv, ppd = pmv_ppd(temp_c, temp_c, _comfort.vel, rh, _comfort.met, _comfort.clo)
    return {"temp_c": temp_c, "pmv": round(pmv, 2), "ppd_percent": round(ppd, 1),
            "comfortable": abs(pmv) <= _comfort.pmv_limit}


def t_grid_signals(iso_datetime: str) -> dict[str, Any]:
    from datetime import datetime
    dt = datetime.fromisoformat(iso_datetime)
    return {"time": iso_datetime, "price_per_kwh": round(_tariff.price(dt), 3),
            "tier": _tariff.tier(dt), "carbon_g_per_kwh": round(_tariff.carbon(dt))}


def t_run_closed_loop(backend: str = "rc", days: int = 2, use_llm: bool = False) -> dict[str, Any]:
    """Run baseline + agent and return the proven savings (RC backend by default)."""
    if backend in ("energyplus", "ep", "eplus"):
        # EnergyPlus's C++ core keeps global state that is not safely reentrant within
        # one process: running baseline then agent back-to-back in-process can corrupt
        # memory. run.py's "compare" already isolates each phase in its own OS process
        # (see _subrun there) — shell out to it instead of duplicating that risk here.
        return _run_closed_loop_subprocess(backend, days, use_llm)

    from ecoloop.control.baseline import BaselineController
    from ecoloop.control.loop import run_loop
    from ecoloop.control.safety import SafetyGuard
    from ecoloop.telemetry import Recorder
    from ecoloop.backends.rc_backend import RCBackend
    from ecoloop.agent.supervisor import AgentSupervisor
    from ecoloop.dashboard import compute_savings

    cfg = load_config()
    cfg["run"]["days"] = days
    if not use_llm:
        cfg["agent"]["disable_llm"] = True

    be = RCBackend(cfg); rec = Recorder("baseline", cfg, be.name, "baseline")
    sb = run_loop(be, BaselineController(cfg), rec); be.close()
    de = max(1, round(cfg["agent"]["control_interval_minutes"] / cfg["run"]["timestep_minutes"]))
    be2 = RCBackend(cfg); rec2 = Recorder("agent", cfg, be2.name, "agent")
    sa = run_loop(be2, AgentSupervisor(cfg), rec2, decide_every=de, safety=SafetyGuard(cfg)); be2.close()
    return {"baseline": sb, "agent": sa, "savings": compute_savings(sb, sa)}


def _run_closed_loop_subprocess(backend: str, days: int, use_llm: bool) -> dict[str, Any]:
    import subprocess
    import tempfile

    from ecoloop.dashboard import compute_savings

    results_dir = tempfile.mkdtemp(prefix="ecoloop_mcp_")
    cmd = [sys.executable, str(Path(__file__).parent / "run.py"), "compare",
           "--backend", backend, "--days", str(days), "--results", results_dir]
    if not use_llm:
        cmd.append("--no-llm")
    subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent))
    sb = json.loads((Path(results_dir) / "baseline_summary.json").read_text())
    sa = json.loads((Path(results_dir) / "agent_summary.json").read_text())
    return {"baseline": sb, "agent": sa, "savings": compute_savings(sb, sa)}


TOOLS: list[dict[str, Any]] = [
    {"name": "parse_building_model",
     "description": "Parse an EnergyPlus .idf and return its controlled zones and ideal-loads units.",
     "inputSchema": {"type": "object", "properties": {"idf_path": {"type": "string"}},
                     "required": ["idf_path"]},
     "fn": t_parse_building_model},
    {"name": "summarize_simulation_log",
     "description": "Compress an EnergyPlus .err log to its severe/warning/fatal lines for triage.",
     "inputSchema": {"type": "object", "properties": {"err_path": {"type": "string"},
                                                      "max_lines": {"type": "integer"}},
                     "required": ["err_path"]},
     "fn": t_summarize_simulation_log},
    {"name": "evaluate_comfort",
     "description": "Fanger PMV/PPD thermal comfort for a zone air temperature.",
     "inputSchema": {"type": "object", "properties": {"temp_c": {"type": "number"},
                                                      "rh": {"type": "number"}},
                     "required": ["temp_c"]},
     "fn": t_evaluate_comfort},
    {"name": "grid_signals",
     "description": "Electricity price, tariff tier, and grid carbon intensity at an ISO datetime.",
     "inputSchema": {"type": "object", "properties": {"iso_datetime": {"type": "string"}},
                     "required": ["iso_datetime"]},
     "fn": t_grid_signals},
    {"name": "run_closed_loop",
     "description": "Run baseline vs autonomous agent and return the proven energy savings. "
                    "backend='rc' (fast) or 'energyplus'; set use_llm=true to drive with the LLM.",
     "inputSchema": {"type": "object", "properties": {
         "backend": {"type": "string"}, "days": {"type": "integer"}, "use_llm": {"type": "boolean"}}},
     "fn": t_run_closed_loop},
]
_BY_NAME = {t["name"]: t for t in TOOLS}


# -- JSON-RPC / MCP plumbing --------------------------------------------------
def _result(id_: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def handle(msg: dict) -> dict | None:
    method = msg.get("method")
    id_ = msg.get("id")
    if method == "initialize":
        return _result(id_, {"protocolVersion": PROTOCOL_VERSION,
                             "capabilities": {"tools": {}}, "serverInfo": SERVER_INFO})
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "tools/list":
        return _result(id_, {"tools": [{k: t[k] for k in ("name", "description", "inputSchema")}
                                        for t in TOOLS]})
    if method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        tool = _BY_NAME.get(name)
        if not tool:
            return _error(id_, -32602, f"unknown tool '{name}'")
        try:
            out = tool["fn"](**args)
            return _result(id_, {"content": [{"type": "text", "text": json.dumps(out, indent=2)}]})
        except Exception as exc:  # noqa: BLE001 - report tool errors to the client
            return _result(id_, {"content": [{"type": "text", "text": f"error: {exc}"}],
                                 "isError": True})
    if method == "ping":
        return _result(id_, {})
    return _error(id_, -32601, f"method not found: {method}")


def main() -> None:
    for line in sys.stdin:
        line = line.strip().lstrip("﻿")   # tolerate a stray BOM on the first line
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(msg)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
