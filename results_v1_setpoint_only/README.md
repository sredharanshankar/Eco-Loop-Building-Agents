# Superseded — earlier setpoint-only result

**This folder is kept for provenance only. It is NOT the submitted result.**

It holds an earlier 7-day run from a version of the agent that controlled **temperature
setpoints alone**, before lighting, plug-load and ventilation control were added. Its
headline figure is **23.4%**, measured against **HVAC electricity only**.

The submitted result lives in [`../results/`](../results) and reports **21.9%**, measured
against **whole-building electricity** (HVAC + lighting + equipment). The percentage is
lower because the denominator is roughly three times larger; the absolute saving is far
greater.

| | This folder (superseded) | `results/` (submitted) |
|---|---|---|
| Control levers | Setpoints only | Setpoints + lighting + plug loads + ventilation |
| Energy measured | HVAC only | Whole building |
| Baseline | 334 kWh | 1020 kWh |
| Agent | 256 kWh | 796 kWh |
| Reported saving | 23.4% | **21.9%** |

Quote `results/` for any comparison.
