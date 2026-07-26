"""EnergyPlus worker subprocess.

Runs EnergyPlus on this process's own (and only) thread and bridges its per-timestep
callback to the parent over stdin/stdout newline-delimited JSON. This exists because a
background-THREAD bridge inside the main controller process was found to crash
intermittently near the end of a run on Windows (a native SEH exception, 0xe06d7363,
raised from deep inside ``run_energyplus`` when invoked off the main thread — likely
during output/report finalization). Running EnergyPlus as a dedicated child process's
main thread avoids that. As a bonus, a native crash here can never take down the
controller/agent process — the parent just sees the pipe end and can react, which is
strictly more robust than the in-process thread it replaces.

Protocol (one JSON object per line):
  worker -> parent : {"t_out":.., "month":.., "day":.., "hour":.., "minute":..,
                       "zones": {name: temp_c}, "cool_j": [..], "heat_j": [..],
                       "light_j": .., "plug_j": ..}
  parent -> worker : {"cool_sp":.., "heat_sp":.., "light":.., "plug":.., "vent":..}
  worker -> parent : {"event": "end"}                       (final line, then exits)
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    spec = json.loads(sys.argv[1])
    sys.path.insert(0, spec["install_dir"])
    from pyenergyplus.api import EnergyPlusAPI

    zones: list[str] = spec["zones"]
    units: list[str] = spec["units"]
    # Each entry: {"name":.., "schedule":.., "design_w":..}
    lights: list[dict] = spec.get("lights", [])
    equips: list[dict] = spec.get("equips", [])
    # Schedules are read (not actuated) so their shape stays available to scale against.
    sched_names = sorted({d["schedule"] for d in lights + equips})
    api = EnergyPlusAPI()
    state = api.state_manager.new_state()
    api.exchange.request_variable(state, "Site Outdoor Air Drybulb Temperature", "Environment")
    for z in zones:
        api.exchange.request_variable(state, "Zone Mean Air Temperature", z)
    for u in units:
        api.exchange.request_variable(state, "Zone Ideal Loads Zone Total Cooling Energy", u)
        api.exchange.request_variable(state, "Zone Ideal Loads Zone Total Heating Energy", u)
    for d in lights:
        api.exchange.request_variable(state, "Lights Electricity Energy", d["name"])
    for d in equips:
        api.exchange.request_variable(state, "Electric Equipment Electricity Energy", d["name"])
    for sn in sched_names:
        api.exchange.request_variable(state, "Schedule Value", sn)
    for u in units:
        api.exchange.request_variable(
            state, "Zone Ideal Loads Outdoor Air Mass Flow Rate", u)

    H: dict = {}

    def emit(obj: dict) -> None:
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    def cb(s) -> None:
        if not api.exchange.api_data_fully_ready(s) or api.exchange.warmup_flag(s):
            return
        if not H:
            H["tout"] = api.exchange.get_variable_handle(
                s, "Site Outdoor Air Drybulb Temperature", "Environment")
            for z in zones:
                H["t_" + z] = api.exchange.get_variable_handle(s, "Zone Mean Air Temperature", z)
                H["cool_" + z] = api.exchange.get_actuator_handle(
                    s, "Zone Temperature Control", "Cooling Setpoint", z)
                H["heat_" + z] = api.exchange.get_actuator_handle(
                    s, "Zone Temperature Control", "Heating Setpoint", z)
            H["cool_e"] = [api.exchange.get_variable_handle(
                s, "Zone Ideal Loads Zone Total Cooling Energy", u) for u in units]
            H["heat_e"] = [api.exchange.get_variable_handle(
                s, "Zone Ideal Loads Zone Total Heating Energy", u) for u in units]
            # Non-HVAC loads: actuators override the scheduled power (absolute watts),
            # while the paired output variables report what was actually consumed.
            H["light_a"] = [api.exchange.get_actuator_handle(
                s, "Lights", "Electricity Rate", d["name"]) for d in lights]
            H["light_e"] = [api.exchange.get_variable_handle(
                s, "Lights Electricity Energy", d["name"]) for d in lights]
            H["plug_a"] = [api.exchange.get_actuator_handle(
                s, "ElectricEquipment", "Electricity Rate", d["name"]) for d in equips]
            H["plug_e"] = [api.exchange.get_variable_handle(
                s, "Electric Equipment Electricity Energy", d["name"]) for d in equips]
            H["oa_a"] = [api.exchange.get_actuator_handle(
                s, "Ideal Loads Air System", "Outdoor Air Mass Flow Rate", u) for u in units]
            H["sched"] = {sn: api.exchange.get_variable_handle(s, "Schedule Value", sn)
                          for sn in sched_names}
            H["oa_v"] = [api.exchange.get_variable_handle(
                s, "Zone Ideal Loads Outdoor Air Mass Flow Rate", u) for u in units]
            # Design outdoor-air flow is autosized and occupancy-dependent, so there is no
            # constant to read. Instead we remember the highest un-throttled flow observed
            # and treat that as full ventilation to scale against.
            H["oa_max"] = [0.0] * len(units)

        def sched_frac(name: str) -> float:
            h = H["sched"].get(name, -1)
            return api.exchange.get_variable_value(s, h) if h > -1 else 1.0

        light_j = [api.exchange.get_variable_value(s, h) if h > -1 else 0.0
                   for h in H["light_e"]]
        plug_j = [api.exchange.get_variable_value(s, h) if h > -1 else 0.0
                  for h in H["plug_e"]]

        emit({
            "t_out": api.exchange.get_variable_value(s, H["tout"]),
            "month": api.exchange.month(s), "day": api.exchange.day_of_month(s),
            "hour": api.exchange.hour(s), "minute": api.exchange.minutes(s),
            "zones": {z: api.exchange.get_variable_value(s, H["t_" + z]) for z in zones},
            "cool_j": [api.exchange.get_variable_value(s, h) for h in H["cool_e"]],
            "heat_j": [api.exchange.get_variable_value(s, h) for h in H["heat_e"]],
            "light_j": sum(light_j),
            "plug_j": sum(plug_j),
        })

        line = sys.stdin.readline()
        if not line:
            return   # parent went away; let EnergyPlus wind down on its own
        try:
            action = json.loads(line)
        except json.JSONDecodeError:
            return
        for z in zones:
            api.exchange.set_actuator_value(s, H["cool_" + z], action["cool_sp"])
            api.exchange.set_actuator_value(s, H["heat_" + z], action["heat_sp"])

        # Non-HVAC loads: design watts x the model's own schedule x the agent's level.
        # Multiplying by the schedule keeps the building's normal daily shape intact, so
        # a level of 1.0 reproduces the baseline exactly rather than overriding it.
        lvl_light = float(action.get("light", 1.0))
        for i, d in enumerate(lights):
            h = H["light_a"][i]
            if h > -1:
                api.exchange.set_actuator_value(
                    s, h, d["design_w"] * sched_frac(d["schedule"]) * lvl_light)
        lvl_plug = float(action.get("plug", 1.0))
        for i, d in enumerate(equips):
            h = H["plug_a"][i]
            if h > -1:
                api.exchange.set_actuator_value(
                    s, h, d["design_w"] * sched_frac(d["schedule"]) * lvl_plug)
        # Ventilation: only ever throttled below the observed flow, never boosted past it.
        lvl_vent = float(action.get("vent", 1.0))
        for i, h in enumerate(H["oa_a"]):
            if h < 0 or i >= len(H["oa_v"]) or H["oa_v"][i] < 0:
                continue
            flow = api.exchange.get_variable_value(s, H["oa_v"][i])
            if lvl_vent >= 0.999:
                # Running un-throttled: learn what full ventilation looks like.
                H["oa_max"][i] = max(H["oa_max"][i], flow)
            elif H["oa_max"][i] > 0.0:
                api.exchange.set_actuator_value(s, h, H["oa_max"][i] * lvl_vent)

    # Set setpoints BEFORE the zone heat balance / predictor so the override actually
    # drives this timestep (after_init_heat_balance is too late — the setpoint is locked).
    api.runtime.callback_begin_zone_timestep_before_init_heat_balance(state, cb)
    api.runtime.set_console_output_status(state, False)
    try:
        api.runtime.run_energyplus(state, ["-w", spec["epw"], "-d", spec["out_dir"], spec["idf"]])
    except Exception as exc:  # noqa: BLE001 - report to the parent's log, still signal end
        sys.stderr.write(f"ep_worker: run_energyplus raised {type(exc).__name__}: {exc}\n")
        sys.stderr.flush()
    finally:
        emit({"event": "end"})


if __name__ == "__main__":
    main()
