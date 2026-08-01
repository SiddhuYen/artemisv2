"""Admission control for graph builds.

A build is minutes of live web crawling plus paid API calls. Three things have
to be true at once:

  - a bounded number run CONCURRENTLY (they multiply against
    EXPAND_NODE_CONCURRENCY and SEARCH_WORKERS, and each one costs money),
  - a bounded number WAIT behind those, because a queue longer than anyone
    will sit through is worse than an honest refusal,
  - a waiting caller can be told WHERE IT IS in the line, so the UI shows
    "2 ahead of you" instead of a spinner that looks broken.

This replaced a single process-wide mutex. That mutex existed only because
connect_people mutated a config global; with that global now a per-call
argument (expand_graph's `prefer_reachable`), the constraint is capacity, not
correctness — so it is a bounded, FIFO, observable queue rather than a lock.

FIFO matters: a plain Semaphore makes no ordering promise, so under sustained
load an unlucky caller can be passed over repeatedly. Here a ticket only runs
when it reaches the head of the line, which also makes `position()` meaningful.

BACKGROUND tickets are the one exception to strict FIFO. An initial-enrichment
run is hours of expansions, and with a capacity of 2 it would otherwise hold
both slots and leave every interactive /connect queued behind it for the whole
run. Background tickets therefore run against a REDUCED capacity that keeps
`reserved` slots free for interactive callers, and an interactive ticket may
overtake a background one that is blocked on that reduction — without the
overtake the reservation would accomplish nothing, since the blocked background
ticket would still sit at the head of the line holding everyone up.

The consequence is deliberate: background work fills idle capacity and yields
the moment real traffic arrives.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable, Optional

from . import config


class QueueFull(Exception):
    """Raised by reserve() when the queue is at capacity — caller should 429."""


class Ticket:
    """A caller's place in line. Opaque; only BuildQueue interprets it."""

    __slots__ = ("acquired", "background")

    def __init__(self, background: bool = False) -> None:
        self.acquired = False
        self.background = background


class BuildQueue:
    def __init__(self, capacity: int, max_queued: int, reserved: int = 0) -> None:
        self._capacity = max(1, int(capacity))
        self._max_queued = max(0, int(max_queued))
        # Clamped below capacity: reserving every slot would mean background
        # work could never run at all, which is a misconfiguration rather than
        # a policy anyone wants.
        self._reserved = max(0, min(int(reserved), self._capacity - 1))
        self._cv = threading.Condition()
        self._pending: "deque[Ticket]" = deque()
        self._running = 0

    # -- admission (called on the request thread, must not block) -----------
    def reserve(self, background: bool = False) -> Ticket:
        """Claim a place in line, or raise QueueFull.

        Separate from acquire() on purpose: the HTTP handler needs a yes/no
        answer immediately so it can return 429, while the actual waiting
        happens on the worker thread where blocking is free.
        """
        with self._cv:
            # Bound TOTAL outstanding work, not each half separately: a build
            # that is merely reserved has not started yet, so gating on
            # `_running` alone admits an unbounded number of tickets whenever
            # the running count happens to be low.
            outstanding = len(self._pending) + self._running
            if outstanding >= self._capacity + self._max_queued:
                raise QueueFull()
            ticket = Ticket(background=background)
            self._pending.append(ticket)
            return ticket

    # -- the worker side ----------------------------------------------------
    def _limit_for(self, ticket: Ticket) -> int:
        """How many slots this ticket's class is allowed to occupy."""
        return self._capacity - (self._reserved if ticket.background else 0)

    def _next_eligible(self) -> Optional[Ticket]:
        """The earliest waiting ticket that could start right now.

        Scanning rather than just checking the head is what lets an interactive
        ticket overtake a background one that is blocked on the reservation.
        Among tickets that CAN run, the earliest still wins, so FIFO holds
        within each class.
        """
        for ticket in self._pending:
            if self._running < self._limit_for(ticket):
                return ticket
        return None

    def acquire(self, ticket: Ticket,
                check_cancel: Optional[Callable[[], None]] = None,
                poll_s: float = 0.25) -> None:
        """Block until `ticket` is the earliest waiting ticket that can start.

        `check_cancel` runs OUTSIDE the lock between waits, so a user who gives
        up while queued stops immediately instead of holding a place until a
        slot happens to free. It raises (JobCancelled) to abort; the caller's
        finally-block releases the ticket.
        """
        while True:
            if check_cancel is not None:
                check_cancel()
            with self._cv:
                if self._next_eligible() is ticket:
                    self._pending.remove(ticket)
                    self._running += 1
                    ticket.acquired = True
                    return
                self._cv.wait(poll_s)

    def release(self, ticket: Ticket) -> None:
        """Give up a slot, or leave the line if never acquired.

        Safe to call for a ticket in either state and safe to call twice, so a
        worker can put it in a plain finally-block without tracking how far it
        got.
        """
        with self._cv:
            if ticket.acquired:
                self._running -= 1
                ticket.acquired = False
            else:
                try:
                    self._pending.remove(ticket)
                except ValueError:
                    pass
            self._cv.notify_all()

    # -- observation --------------------------------------------------------
    def position(self, ticket: Ticket) -> int:
        """0 once running (or released); otherwise 1-based place in line."""
        with self._cv:
            if ticket.acquired:
                return 0
            for i, t in enumerate(self._pending):
                if t is ticket:
                    return i + 1
            return 0

    def stats(self) -> dict:
        with self._cv:
            return {
                "running": self._running,
                "queued": len(self._pending),
                "capacity": self._capacity,
                "max_queued": self._max_queued,
                "reserved_for_interactive": self._reserved,
            }

    def reset(self) -> None:
        """Drop all state. Tests only — never call with builds in flight."""
        with self._cv:
            self._pending.clear()
            self._running = 0
            self._cv.notify_all()


BUILDS = BuildQueue(config.MAX_CONCURRENT_BUILDS, config.MAX_QUEUED_BUILDS,
                    config.ENRICH_RESERVED_BUILD_SLOTS)
