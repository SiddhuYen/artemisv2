"""The background lane in BuildQueue.

An enrichment run is hours of expansions. Without a reservation it would hold
every build slot for that whole time and leave interactive /connect queued
behind it, so the queue has to guarantee two things at once: background work
uses idle capacity, and it never occupies the last slot.
"""
import threading

import pytest

from app.buildqueue import BuildQueue, QueueFull

_SETTLE = 0.2   # long enough for a waiter to have acquired if it ever could
_GRANT = 2.0    # generous: a pass here should never depend on scheduler luck


def _waiter(queue, ticket):
    """Start ONE background acquire() for `ticket` and return its done-event.

    One thread per ticket, deliberately: a second waiter on the same ticket
    would race the first, and the loser blocks forever once the winner removes
    the ticket from the queue — an artifact of the test, not of BuildQueue.
    """
    done = threading.Event()

    def _go():
        queue.acquire(ticket, poll_s=0.01)
        done.set()

    threading.Thread(target=_go, daemon=True).start()
    return done


def test_background_cannot_take_the_last_slot():
    queue = BuildQueue(capacity=2, max_queued=8, reserved=1)
    interactive = queue.reserve()
    queue.acquire(interactive)                     # 1 of 2 running

    waiting = _waiter(queue, queue.reserve(background=True))
    assert not waiting.wait(_SETTLE)               # the free slot is reserved

    queue.release(interactive)                     # now idle
    assert waiting.wait(_GRANT)


def test_interactive_overtakes_a_blocked_background_ticket():
    """Without the overtake the reservation would accomplish nothing: the
    blocked background ticket would sit at the head of the line holding up
    exactly the callers it was meant to protect."""
    queue = BuildQueue(capacity=2, max_queued=8, reserved=1)
    queue.acquire(queue.reserve())

    background = _waiter(queue, queue.reserve(background=True))  # queued FIRST
    interactive = _waiter(queue, queue.reserve())                # queued second

    assert interactive.wait(_GRANT)
    assert not background.wait(_SETTLE)


def test_only_one_background_build_runs_at_a_time():
    queue = BuildQueue(capacity=2, max_queued=8, reserved=1)
    assert _waiter(queue, queue.reserve(background=True)).wait(_GRANT)

    second = _waiter(queue, queue.reserve(background=True))
    assert not second.wait(_SETTLE)
    # …and a slot is still free for interactive work
    assert _waiter(queue, queue.reserve()).wait(_GRANT)


def test_fifo_still_holds_within_a_class():
    queue = BuildQueue(capacity=1, max_queued=8, reserved=0)
    holder = queue.reserve()
    queue.acquire(holder)                          # popped out of the line

    first, second = queue.reserve(), queue.reserve()
    assert queue.position(first) == 1
    assert queue.position(second) == 2

    first_waiting, second_waiting = _waiter(queue, first), _waiter(queue, second)
    queue.release(holder)
    assert first_waiting.wait(_GRANT)
    assert not second_waiting.wait(_SETTLE)        # capacity is 1


def test_reservation_is_clamped_below_capacity():
    """Reserving every slot would mean background work could never run — a
    misconfiguration, not a policy, so it is clamped rather than honoured."""
    queue = BuildQueue(capacity=2, max_queued=1, reserved=5)
    assert queue.stats()["reserved_for_interactive"] == 1
    assert _waiter(queue, queue.reserve(background=True)).wait(_GRANT)


def test_background_still_counts_against_queue_capacity():
    queue = BuildQueue(capacity=1, max_queued=1, reserved=0)
    queue.acquire(queue.reserve())
    queue.reserve(background=True)
    with pytest.raises(QueueFull):
        queue.reserve(background=True)


def test_default_queue_reserves_a_slot_for_interactive_work():
    from app.buildqueue import BUILDS
    assert BUILDS.stats()["reserved_for_interactive"] >= 1
