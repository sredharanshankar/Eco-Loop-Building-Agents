"""Decision cache — the core of prompt-latency management.

A small local LLM on CPU is the slowest link in the loop. Building conditions repeat
daily, so we bucket the situation (hour, occupancy, outdoor-temp band, price tier,
carbon band) and reuse a prior decision for an equivalent bucket instead of paying for
another inference. Over a multi-day horizon this collapses hundreds of control steps to
a few dozen actual model calls, keeping the loop real-time while staying fully agent-
driven. Cache hits/misses are reported in the run summary.
"""

from __future__ import annotations

from ..backends.base import Action, Observation
from ..signals import TariffCarbon


class DecisionCache:
    def __init__(self, tariff: TariffCarbon, enabled: bool = True):
        self.tariff = tariff
        self.enabled = enabled
        self._store: dict[tuple, Action] = {}
        self.hits = 0
        self.misses = 0

    def key(self, obs: Observation, occupied: bool) -> tuple:
        # Coarse buckets: the ECM strategy is driven by hour, occupancy, price tier and a
        # broad outdoor band, so this collapses a multi-day horizon to a few dozen unique
        # inferences (the rest are cache hits) while keeping decisions sensible.
        dt = obs.time
        return (
            dt.hour,
            occupied,
            self.tariff.tier(dt),
            round(obs.outdoor_temp / 4.0) * 4,        # 4 C bands
        )

    def get(self, obs: Observation, occupied: bool) -> dict | None:
        """Return the cached *decision arguments* for an equivalent situation.

        We deliberately cache the chosen ECM strategy rather than the resulting
        setpoints. The nightly self-critique retunes what a strategy means, which would
        invalidate cached numbers every single night — that is what drove the hit rate to
        zero and roughly doubled a 7-day run. The strategy itself stays valid, so the
        caller re-derives fresh setpoints from the current policy on every hit.
        """
        if not self.enabled:
            return None
        hit = self._store.get(self.key(obs, occupied))
        if hit is not None:
            self.hits += 1
            return dict(hit)
        self.misses += 1
        return None

    def put(self, obs: Observation, occupied: bool, args: dict) -> None:
        if self.enabled and args:
            self._store[self.key(obs, occupied)] = dict(args)

    @property
    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
            "unique_decisions": len(self._store),
        }
