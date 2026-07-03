"""Per-run provider + cache statistics (thread-safe)."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ProviderStats:
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    calls: Dict[str, int] = field(default_factory=dict)        # provider -> network calls
    latencies: List[float] = field(default_factory=list)       # seconds per network call
    cache_hits: int = 0
    cache_misses: int = 0
    dedup_removed: int = 0
    breaker_trips: int = 0

    def record_call(self, provider: str, latency: float) -> None:
        with self._lock:
            self.calls[provider] = self.calls.get(provider, 0) + 1
            self.latencies.append(latency)

    def hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def miss(self) -> None:
        with self._lock:
            self.cache_misses += 1

    def removed_by_dedup(self, n: int) -> None:
        with self._lock:
            self.dedup_removed += n

    def breaker_tripped(self) -> None:
        with self._lock:
            self.breaker_trips += 1

    # --- reporting ---------------------------------------------------------
    def avg_latency(self) -> float:
        with self._lock:
            return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    def total_search_time(self) -> float:
        with self._lock:
            return sum(self.latencies)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "brave_searches": self.calls.get("brave", 0),
                "wikipedia_calls": self.calls.get("wikipedia", 0),
                "wikidata_calls": self.calls.get("wikidata", 0),
                "openalex_calls": self.calls.get("openalex", 0),
                "opencorporates_calls": self.calls.get("opencorporates", 0),
                "edgar_calls": self.calls.get("edgar", 0),
                "propublica_calls": self.calls.get("propublica", 0),
                "duckduckgo_searches": self.calls.get("duckduckgo", 0),
                "page_fetches": self.calls.get("fetch", 0),
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "dedup_removed": self.dedup_removed,
                "breaker_trips": self.breaker_trips,
                "avg_latency_s": round(
                    sum(self.latencies) / len(self.latencies), 3) if self.latencies else 0.0,
                "total_search_time_s": round(sum(self.latencies), 2),
            }

    def reset(self) -> None:
        with self._lock:
            self.calls.clear()
            self.latencies.clear()
            self.cache_hits = self.cache_misses = 0
            self.dedup_removed = self.breaker_trips = 0


# process-wide singleton
STATS = ProviderStats()
