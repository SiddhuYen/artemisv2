import app.main as M


def test_cancel_job_marks_running_job_and_signals_worker():
    M._JOBS.clear()
    job_id = M._new_job("connect")

    result = M.cancel_job(job_id)

    assert result["kind"] == "connect"
    assert result["status"] == "cancelling"
    assert result["cancel_requested"] is True
    assert result["message"] == "cancelling…"
    assert "_cancel_event" not in result
    assert M._job_cancel_event(job_id).is_set()


def test_job_progress_is_monotonic_until_done():
    M._JOBS.clear()
    job_id = M._new_job("connect")

    M._update_job(job_id, pct=42, message="hop 1")
    M._update_job(job_id, pct=18, message="hop start should not reset")
    result = M._get_job(job_id)

    assert result["pct"] == 42
    assert result["message"] == "hop start should not reset"

    M._update_job(job_id, pct=250, message="clamped")
    assert M._get_job(job_id)["pct"] == 100


def test_cancelled_job_cannot_be_overwritten_as_done():
    M._JOBS.clear()
    job_id = M._new_job("connect")
    M.cancel_job(job_id)

    M._update_job(job_id, status="done", pct=100, message="done", result={"ok": True})
    result = M._get_job(job_id)

    assert result["status"] == "cancelled"
    assert result["message"] == "cancelled"
    assert result["result"] is None


def test_cancelled_job_cannot_be_overwritten_as_error():
    M._JOBS.clear()
    job_id = M._new_job("connect")
    M.cancel_job(job_id)

    M._update_job(job_id, status="error", error="late unwind failure")
    result = M._get_job(job_id)

    assert result["status"] == "cancelled"
    assert result["error"] is None


# ---------------------------------------------------------------------------
# _append_job_log -- the "thinking" transcript (graph.*'s progress() lines,
# previously generated and discarded since no HTTP job passed a progress
# callback into connect_people at all).
# ---------------------------------------------------------------------------
def test_new_job_starts_with_an_empty_log():
    M._JOBS.clear()
    job_id = M._new_job("connect")
    assert M._get_job(job_id)["log"] == []


def test_append_job_log_accumulates_in_order():
    M._JOBS.clear()
    job_id = M._new_job("connect")

    M._append_job_log(job_id, "first line")
    M._append_job_log(job_id, "second line")

    assert M._get_job(job_id)["log"] == ["first line", "second line"]


def test_append_job_log_is_a_noop_for_an_unknown_job():
    M._JOBS.clear()
    M._append_job_log("does-not-exist", "line")  # must not raise


def test_append_job_log_bounds_growth_to_the_most_recent_lines():
    M._JOBS.clear()
    job_id = M._new_job("connect")

    for i in range(M._JOB_LOG_MAX + 50):
        M._append_job_log(job_id, f"line {i}")

    log = M._get_job(job_id)["log"]
    assert len(log) == M._JOB_LOG_MAX
    # oldest lines dropped, newest kept, order preserved
    assert log[0] == f"line {50}"
    assert log[-1] == f"line {M._JOB_LOG_MAX + 49}"
