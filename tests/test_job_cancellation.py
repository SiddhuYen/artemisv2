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
