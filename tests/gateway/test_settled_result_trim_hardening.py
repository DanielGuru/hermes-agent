"""Settled-result ageing on the live SQLite holder, including competing and replayed writes."""
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from gateway import session_bot
from gateway.session_bot_outcome import FORGOTTEN_REPLY_PREFIX
from persistence.authority.bindings import local_session_id
from persistence.authority.vocabulary import RETRY_HORIZON_S
from persistence.core.errors import DiskFull, PersistenceError, SettledResultTrimIncomplete
from persistence.sqlite.conversation._store.terminal import settled_key
from persistence.sqlite.housekeeping import maintenance
from persistence_assembly import open_persistence, open_store_admin, profile_scope
from session_retention import run_settled_result_trim
from tests.contracts.authority.test_bindings_contract import make_binding
from tests.fixtures.storage_holder import admission_result_key, raw_holder
from tests.fixtures.storage_seed import seed_admission, seed_messages, seed_session, settle_claim


NOW = 1_800_000_000
DAY = 86400


def _result(reply="done", **fields):
    return json.dumps({"result": {"final_response": reply, "messages": [
        {"role": "assistant", "content": "visible words"}], **fields}})


def _insert(conn, admission_id, raw, stamp):
    conn.execute("INSERT INTO state_meta(key,value) VALUES(?,?)", (admission_result_key(admission_id), raw))
    if stamp is not None:
        conn.execute("INSERT INTO state_meta(key,value) VALUES(?,?)", (settled_key(admission_id), stamp))


def _seed(holder, n=10, *, raw=None, stamp="1"):
    def insert(conn):
        for i in range(n):
            _insert(conn, f"row-{i}", raw if raw is not None else _result(), stamp)
    holder._execute_write(insert)


def _trimmed_count(conn):
    return conn.execute("SELECT count(*) FROM state_meta WHERE key GLOB ? "
                        "AND json_extract(value,'$.result.messages')='[]'",
                        (admission_result_key() + "*",)).fetchone()[0]


@pytest.fixture
def batches(owner, monkeypatch):
    monkeypatch.setattr(maintenance, "BATCH", 3, raising=False)
    monkeypatch.setattr(maintenance, "BATCH_YIELD_S", 0, raising=False)
    return raw_holder(owner.persistence)


def _observe_writes(holder, monkeypatch, before_commit=None):
    """Count only committed SQL changes. Keep retry, rollback, lock and commit on the real holder."""
    write = holder._execute_write
    committed = []

    def observe(fn, *args, **kwargs):
        changes = 0

        def measured(conn):
            nonlocal changes
            start = conn.total_changes
            value = fn(conn)
            changes = conn.total_changes - start
            if before_commit is not None:
                before_commit(conn, changes)
            return value

        value = write(measured, *args, **kwargs)
        committed.append(changes)
        return value

    monkeypatch.setattr(holder, "_execute_write", observe)
    return committed


@pytest.mark.parametrize("stamp", [None, "1"], ids=["backfill", "trim"])
def test_ten_changes_are_committed_in_slices_of_at_most_three(owner, batches, monkeypatch, stamp):
    _seed(batches, stamp=stamp)
    changes = _observe_writes(batches, monkeypatch)
    assert owner.persistence.maintenance.trim_settled_results() == (0 if stamp is None else 10)
    assert sum(changes) == 10 and max(changes) <= 3, changes
    assert all(batches.get_meta(settled_key(f"row-{i}")) is not None for i in range(10))
    with batches._read_ctx() as conn:
        assert _trimmed_count(conn) == (0 if stamp is None else 10)


def test_even_an_all_skipped_workset_terminates(owner, batches, monkeypatch):
    raw = _result("")
    _seed(batches, raw=raw)

    def bound_calls(conn, changes):
        assert len(commits) < 25, "revisiting skipped candidates"

    commits = _observe_writes(batches, monkeypatch, bound_calls)
    assert owner.persistence.maintenance.trim_settled_results() == 0
    assert not any(commits)
    assert all(batches.get_meta(admission_result_key(f"row-{i}")) == raw for i in range(10))


def test_holder_replays_second_trim_slice_without_double_counting(owner, batches, monkeypatch):
    _seed(batches)
    injected = False

    def lock_once(conn, changes):
        nonlocal injected
        if changes and not injected and sum(commits) > 0:
            injected = True
            raise sqlite3.OperationalError("database is locked")  # after the callback, before commit

    commits = _observe_writes(batches, monkeypatch, lock_once)
    assert owner.persistence.maintenance.trim_settled_results() == 10
    assert injected and sum(commits) == 10
    with batches._read_ctx() as conn:
        assert _trimmed_count(conn) == 10


def test_waiting_writer_commits_before_the_last_slice(owner, batches, monkeypatch):
    _seed(batches)
    first_slice, issued = threading.Event(), threading.Event()
    write = batches._execute_write
    committed = threading.Event()

    def wait_for_writer(_delay):
        assert committed.wait(10)

    monkeypatch.setattr(maintenance, "time", SimpleNamespace(time=time.time, sleep=wait_for_writer))

    def writer():
        assert first_slice.wait(10)
        issued.set()

        def record(conn):
            count = _trimmed_count(conn)
            conn.execute("INSERT INTO state_meta(key,value) VALUES('writer-saw',?)", (str(count),))
            return count

        count = write(record)
        committed.set()
        return count

    def queue_writer(conn, changes):
        if changes and not first_slice.is_set():
            first_slice.set()  # maintenance still holds this transaction's lock
            assert issued.wait(10)
        if _trimmed_count(conn) == 10:
            assert conn.execute("SELECT value FROM state_meta WHERE key='writer-saw'").fetchone() is not None

    _observe_writes(batches, monkeypatch, queue_writer)
    with ThreadPoolExecutor(max_workers=1) as pool:
        other = pool.submit(writer)
        assert owner.persistence.maintenance.trim_settled_results() == 10
        assert 0 < other.result(timeout=10) < 10


def test_partial_failure_reports_only_commits_releases_lock_and_can_resume(owner, batches, monkeypatch, tmp_path):
    store = owner.persistence
    admin = Mock(wraps=open_store_admin(profile_scope(home=tmp_path), read_only=False))
    _seed(batches)
    batches._execute_write(lambda conn: _insert(conn, "legacy", _result(), None))

    write = batches._execute_write

    def fail_after_first_slice(fn, *args, **kwargs):
        with batches._read_ctx() as conn:
            if _trimmed_count(conn):
                raise sqlite3.OperationalError("injected write failure")
        return write(fn, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(batches, "_execute_write", fail_after_first_slice)
        first = run_settled_result_trim(store, admin, vacuum=False)
    with batches._read_ctx() as conn:
        committed = _trimmed_count(conn)
    assert 0 < committed <= 3
    assert first["incomplete"] and "injected write failure" in first["error"]
    assert (first["trimmed"], first["stamped"]) == (committed, 1)
    assert batches.get_meta(settled_key("legacy")) is not None
    assert store.state_meta.get("last_settled_result_trim") is None
    admin.release_auto_maintenance_lock.assert_called_once()

    again = run_settled_result_trim(store, admin, vacuum=False)
    assert "error" not in again and again["trimmed"] == 10 - committed
    assert store.state_meta.get("last_settled_result_trim") is not None
    with batches._read_ctx() as conn:
        assert _trimmed_count(conn) == 10


def test_post_snapshot_insert_reclaim_restamp_and_rewrite(owner, batches, monkeypatch):
    initial, changed = _result(), _result("new reply", usage_revision=2)
    write = batches._execute_write

    def setup(conn):
        _insert(conn, "reclaimed", initial, None)
        _insert(conn, "restamped", initial, "1")
        _insert(conn, "rewritten", initial, "1")

    def concurrent_changes(conn):
        conn.execute("DELETE FROM state_meta WHERE key=?", (admission_result_key("reclaimed"),))
        conn.execute("UPDATE state_meta SET value=? WHERE key=?",
                     (str(int(time.time())), settled_key("restamped")))
        conn.execute("UPDATE state_meta SET value=? WHERE key=?", (changed, admission_result_key("rewritten")))
        _insert(conn, "zz-new", initial, "1")

    injected = False

    def first_write(fn, *args, **kwargs):
        nonlocal injected
        if not injected:
            write(concurrent_changes)  # between the read snapshot and the first write transaction
            injected = True
        return write(fn, *args, **kwargs)

    write(setup)
    monkeypatch.setattr(batches, "_execute_write", first_write)
    assert owner.persistence.maintenance.trim_settled_results() == 1
    assert batches.get_meta(settled_key("reclaimed")) is None
    assert batches.get_meta(admission_result_key("zz-new")) == initial
    assert batches.get_meta(admission_result_key("restamped")) == initial
    assert json.loads(batches.get_meta(admission_result_key("rewritten")))["result"] == {
        "final_response": "new reply", "messages": [], "usage_revision": 2}
    assert owner.persistence.maintenance.trim_settled_results() == 1


def test_oversize_payload_gets_a_slice_to_itself_and_progresses(owner, batches, monkeypatch):
    monkeypatch.setattr(maintenance, "BATCH_MAX_BYTES", 1000, raising=False)
    _seed(batches, n=4, raw=_result(last_reasoning="x" * 2000))
    changes = _observe_writes(batches, monkeypatch)
    assert owner.persistence.maintenance.trim_settled_results() == 4
    assert [n for n in changes if n] == [1] * 4


@pytest.mark.parametrize("turn", ["preflight-timeout", "after-reset"])
def test_unstamped_result_ages_from_first_observation(owner, batches, monkeypatch, turn):
    store = owner.persistence
    sid = local_session_id("default", "user-1", "req-1")
    binding = make_binding(sid)
    store.bindings.reserve(epoch=owner.epoch, binding=binding, owner=None)
    seed_messages(store, sid, {"role": "assistant", "content": "visible words", "timestamp": NOW - 40 * DAY})
    if turn == "after-reset":
        physical = "reset-target"
        store.bindings.reset_target(epoch=owner.epoch, parent_session_id=sid, entry={**binding.entry, "session_id": physical})
        seed_messages(store, physical, {"role": "user", "content": "new request", "timestamp": NOW})
    admission = seed_admission(store, sid, turn, status="terminal", outcome="completed")
    raw = _result("preflight compression timed out" if turn == "preflight-timeout" else "new reply")
    batches._execute_write(lambda conn: _insert(conn, admission, raw, None))
    monkeypatch.setattr(time, "time", lambda: NOW)

    assert store.maintenance.trim_settled_results() == 0
    assert batches.get_meta(settled_key(admission)) == str(NOW)
    assert batches.get_meta(admission_result_key(admission)) == raw


def test_age_boundary_requires_the_entire_stamped_second(owner, batches, monkeypatch):
    _seed(batches, n=1, stamp=str(NOW))
    monkeypatch.setattr(time, "time", lambda: NOW + RETRY_HORIZON_S + .75)
    assert owner.persistence.maintenance.trim_settled_results() == 0
    assert batches.get_meta(admission_result_key("row-0")) == _result()
    monkeypatch.setattr(time, "time", lambda: NOW + RETRY_HORIZON_S + 1)
    assert owner.persistence.maintenance.trim_settled_results() == 1


def test_malformed_stamps_never_turn_into_zero_or_partial_integers(owner, batches):
    stamps = ["", "not a timestamp", "3oops", "1.5", "1e2"]

    def seed_bad_stamps(conn):
        for i, stamp in enumerate(stamps):
            _insert(conn, str(i), _result(), stamp)

    batches._execute_write(seed_bad_stamps)
    assert owner.persistence.maintenance.trim_settled_results() == 0
    for i, stamp in enumerate(stamps):
        assert batches.get_meta(settled_key(str(i))) == stamp
        assert batches.get_meta(admission_result_key(str(i))) == _result()


def test_canonical_encoding_failure_is_isolated_and_byte_preserved(owner, batches):
    bad = _result(cost=float("inf"))
    batches._execute_write(lambda conn: _insert(conn, "a-bad", bad, "1"))
    batches._execute_write(lambda conn: _insert(conn, "z-good", _result(), "1"))
    assert owner.persistence.maintenance.trim_settled_results() == 1
    assert batches.get_meta(admission_result_key("a-bad")) == bad
    assert json.loads(batches.get_meta(admission_result_key("z-good")))["result"]["messages"] == []


def test_silence_marker_keeps_the_senders_forgotten_reply(owner, batches, tmp_path):
    store = owner.persistence
    sid = seed_session(store, "chat", "api_server")
    store.admissions.admit(epoch=owner.epoch, principal_id="bot", session_id=sid,
                           request_id="ask", payload={"text": "ask"})
    claimed = store.admissions.claim(epoch=owner.epoch, session_id=sid)
    marker = "[ SILENT ]"
    words = "here is the answer"
    settle_claim(store, claimed, result={"result": {"final_response": marker, "messages": [
        {"role": "assistant", "content": words}, {"role": "assistant", "content": marker}]}})
    admission_id = claimed.admission_id
    record = {"delivery_id": "ask", "session_id": sid, "profile_home": str(tmp_path), "admission_id": admission_id,
              "message": "ask", "author": {"name": "sender", "is_bot": True}}
    store.state_meta.set(settled_key(admission_id), "1")
    before = session_bot._result(owner, record)
    outcome = session_bot._admission_outcome(owner, admission_id)  # v0.23.3 keeps the marker; main normalizes it
    assert before["status"] == "settled"  # v0.23.3 replies with the marker; main recovers FORGOTTEN_REPLY_PREFIX + words

    store.maintenance.trim_settled_results()
    assert session_bot._admission_outcome(owner, admission_id) == outcome
    assert session_bot._result(owner, record) == before
    # v0.23.3 does not treat a spaced "[ SILENT ]" as silence: it is an ordinary delivered reply, so its
    # transcript copy may be trimmed. The only contract here is that the sender-visible reply is unchanged.


def test_unsupported_maintenance_skips_before_lock_or_throttle():
    admin, state_meta = Mock(), Mock()
    store = SimpleNamespace(maintenance=object(), state_meta=state_meta)
    result = run_settled_result_trim(store, admin)
    assert result["skipped"] and "error" not in result
    assert not admin.mock_calls and not state_meta.mock_calls


@pytest.mark.parametrize("stamp", [None, "1"], ids=["backfill", "trim"])
def test_watchdog_renewed_after_each_committed_slice(owner, batches, monkeypatch, stamp):
    _seed(batches, stamp=stamp)
    observed = []

    def progress(lease, *, phase):
        assert phase == "state_db_settled_result_trim" and lease == 900
        with batches._read_ctx() as conn:
            count = (_trimmed_count(conn) if stamp else conn.execute(
                "SELECT count(*) FROM state_meta WHERE key GLOB ?", (settled_key("row-*"),)).fetchone()[0])
        observed.append(count)

    monkeypatch.setattr("session_retention.report_startup_progress", progress)
    result = run_settled_result_trim(owner.persistence, None, vacuum=False)
    assert "error" not in result
    # The initial lease and completed discovery page both precede the first committed slice.
    assert observed == [0, 0, 3, 6, 9, 10]


def test_skipped_payloads_consume_slice_byte_budget(owner, batches, monkeypatch):
    raw = _result("", last_reasoning="x" * 500)
    budget = 1000
    assert budget / 2 < len(raw.encode()) < budget
    monkeypatch.setattr(maintenance, "BATCH_MAX_BYTES", budget)
    _seed(batches, n=5, raw=raw)
    commits = _observe_writes(batches, monkeypatch)
    assert owner.persistence.maintenance.trim_settled_results() == 0
    assert len(commits) >= 3
    assert all(batches.get_meta(admission_result_key(f"row-{i}")) == raw for i in range(5))


def test_failure_clears_prior_success_even_if_retry_interval_grows(owner, batches, monkeypatch):
    store = owner.persistence
    _seed(batches, n=5)
    store.state_meta.set("last_settled_result_trim", str(time.time() - 25 * 3600))

    def fail_second(conn, changes):
        if _trimmed_count(conn) == 5:
            raise sqlite3.OperationalError("injected write failure")

    with monkeypatch.context() as patch:
        _observe_writes(batches, patch, fail_second)
        first = run_settled_result_trim(store, None, vacuum=False)
    assert first["incomplete"] and first["trimmed"] == 3
    assert store.state_meta.get("last_settled_result_trim") is None
    again = run_settled_result_trim(store, None, min_interval_hours=48, vacuum=False)
    assert not again["skipped"] and again["trimmed"] == 2


def test_partial_store_failure_preserves_disk_full_cause_and_counts(owner, batches, monkeypatch):
    """A direct caller can catch the store failure without importing the retention runner."""
    _seed(batches, n=5)

    def disk_full(conn, changes):
        if sum(commits):
            raise sqlite3.OperationalError("database or disk is full")

    commits = _observe_writes(batches, monkeypatch, disk_full)
    with pytest.raises(PersistenceError) as failure:
        owner.persistence.maintenance.trim_settled_results()
    assert isinstance(failure.value, SettledResultTrimIncomplete)
    assert isinstance(failure.value.__cause__, DiskFull)
    assert (failure.value.trimmed, failure.value.stamped) == (3, 0)


@pytest.mark.parametrize("readers", [0, 1], ids=["throttle", "discovery"])
def test_no_pooled_reader_uses_normal_read_fallback(owner, batches, monkeypatch, readers):
    store = owner.persistence
    _seed(batches, n=1)
    prior = str(time.time() - 25 * 3600)
    store.state_meta.set("last_settled_result_trim", prior)
    with monkeypatch.context() as patch:
        checkout = batches._checkout_read_conn
        permits = iter(range(readers))
        patch.setattr(batches, "_checkout_read_conn", lambda: checkout() if next(permits, None) is not None else None)
        result = run_settled_result_trim(store, None, vacuum=False)
    # A missing reader now uses the normal locked read for one page instead of deferring the pass.
    assert not result["skipped"] and result["trimmed"] == 1, result
    assert float(store.state_meta.get("last_settled_result_trim")) > float(prior)
    assert json.loads(batches.get_meta(admission_result_key("row-0")))["result"]["messages"] == []
    assert run_settled_result_trim(store, None, vacuum=False)["skipped"]


@pytest.mark.parametrize("failed_slice", [1, 2])
def test_incomplete_slice_retries_with_longer_interval(owner, batches, monkeypatch, failed_slice):
    store = owner.persistence
    _seed(batches, n=5)
    store.state_meta.set("last_settled_result_trim", str(time.time() - 25 * 3600))
    write = batches._execute_write
    calls = 0

    def fail_once(fn, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failed_slice:
            def roll_back(conn):
                fn(conn)
                raise sqlite3.OperationalError("injected commit failure")
            return write(roll_back)
        return write(fn, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(batches, "_execute_write", fail_once)
        first = run_settled_result_trim(store, None, vacuum=False)
    assert first["incomplete"] and first["trimmed"] == (failed_slice - 1) * 3
    with batches._read_ctx() as conn:
        assert _trimmed_count(conn) == first["trimmed"]
    if failed_slice == 2:
        assert store.state_meta.get("settled_result_trim_incomplete") is not None
    retry = run_settled_result_trim(store, None, min_interval_hours=48, vacuum=False)
    assert not retry["skipped"] and retry["trimmed"] == 5 - first["trimmed"]
    assert store.state_meta.get("settled_result_trim_incomplete") is None
    assert run_settled_result_trim(store, None, vacuum=False)["skipped"]


def test_minimal_store_trims_with_only_the_horizon(owner):
    seen = []

    def trim(older_than=None):
        seen.append(older_than)
        return 4

    store = SimpleNamespace(maintenance=SimpleNamespace(trim_settled_results=trim), state_meta=owner.persistence.state_meta)
    result = run_settled_result_trim(store, None, older_than=30 * DAY)
    assert result["trimmed"] == 4 and seen == [30 * DAY], result
    assert run_settled_result_trim(store, None)["skipped"]


def test_writer_commits_during_paged_discovery_and_new_key_waits(owner, batches, monkeypatch):
    _seed(batches, n=5)
    from persistence.sqlite.housekeeping import _retirement

    monkeypatch.setattr(_retirement, "SNAPSHOT_PAGE_SIZE", 2, raising=False)
    read = batches._read_ctx
    pages = []

    @contextmanager
    def observe_read(**kwargs):
        with read(**kwargs) as conn:
            def on_statement(sql):
                if sql.lstrip().upper().startswith("SELECT"):
                    pages.append(sql)
                    if len(pages) == 2:
                        writer.submit(batches._execute_write, lambda c: _insert(c, "zz-new", _result(), "1")).result(10)
            conn.set_trace_callback(on_statement)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)

    with ThreadPoolExecutor(max_workers=1) as writer, monkeypatch.context() as patch:
        patch.setattr(batches, "_read_ctx", observe_read)
        assert owner.persistence.maintenance.trim_settled_results() == 5
    assert len(pages) >= 3
    assert batches.get_meta(admission_result_key("zz-new")) == _result()


def test_delete_journal_pass_backfills_and_trims(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("database:\n  journal_mode: delete\n", encoding="utf-8")
    store = open_persistence(profile_scope(home=tmp_path))
    try:
        holder = raw_holder(store)
        with holder._read_ctx() as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        _seed(holder, n=1)
        holder._execute_write(lambda conn: _insert(conn, "legacy", _result(), None))
        result = run_settled_result_trim(store, None, vacuum=False)
        assert result["trimmed"] == 1, result
        assert json.loads(holder.get_meta(admission_result_key("row-0")))["result"]["messages"] == []
        assert holder.get_meta(settled_key("legacy")) is not None
        assert store.state_meta.get("last_settled_result_trim") is not None
    finally:
        store.close()


def test_paged_wal_discovery_renews_lease_without_writer_mutex(owner, batches, monkeypatch):
    from persistence.sqlite.housekeeping import _retirement

    _seed(batches, n=5, stamp=str(int(time.time())))  # discovery only: no slice can renew the lease
    monkeypatch.setattr(_retirement, "SNAPSHOT_PAGE_SIZE", 2)
    read = batches._read_ctx
    clock, deadline, renewals = 0, 900, 0
    reading = False

    @contextmanager
    def slow_read():
        nonlocal clock, reading
        with read() as conn:
            reading = True
            clock += 400
            assert clock < deadline, "watchdog lease expired during discovery"
            yield conn
        reading = False

    def renew():
        nonlocal deadline, renewals
        assert not reading
        deadline = clock + 900
        renewals += 1

    with monkeypatch.context() as patch:
        patch.setattr(batches, "_read_ctx", slow_read)
        patch.setattr(batches, "_lock", None)  # a pooled WAL reader needs no writer mutex
        assert owner.persistence.maintenance.trim_settled_results(progress=renew) == 0
    assert renewals == 3



def test_trimmed_compressed_result_is_no_longer_a_replacement_history(owner, batches):
    _seed(batches, n=1, raw=_result(_compressed=True))
    assert owner.persistence.maintenance.trim_settled_results() == 1
    inner = json.loads(batches.get_meta(admission_result_key("row-0")))["result"]
    assert inner["messages"] == [] and "_compressed" not in inner
    assert inner["final_response"] == "done"
