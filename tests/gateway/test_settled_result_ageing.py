"""A settled result sheds its transcript copies once it can no longer answer an exact retry.

The settlement writes one result per settled turn under ``gateway.admission.result.v1.<id>`` and used
to keep every one of them, transcript copies included, for the life of its session: only a *retired*
session's tombstones were reclaimed, so one long-lived chat carried one full result per turn forever
(751 rows / 127 MB on the reference install, 96% of it the per-turn ``messages`` copies).

``RETRY_HORIZON_S`` already bounds which retries get a byte-exact reply. Past it the row is trimmed,
**not deleted**: ``finish_result_precheck`` and the hosted-task recovery path read "terminal admission
with no stored result" as a storage fault, and the bot receipt path degrades a settled turn to
``ambiguous`` without a ``final_response``. These tests pin the stamp, the trim, and the invariants
those readers depend on.
"""
import json
import time

import pytest

from gateway.session_authority import admission_row
from gateway.session_persistence import authority_persistence
from gateway.session_results import admission_result, finish_result
from persistence.authority.vocabulary import RETRY_HORIZON_S
from persistence.sqlite.authority._admission_owner import finish_result_precheck
from persistence.sqlite.conversation._store.terminal import settled_key
from session_retention import run_settled_result_trim
from tests.fixtures.storage_holder import admission_result_key, raw_holder
from tests.fixtures.storage_seed import seed_session


def _settle(owner, request_id="r1", history=None, principal_id="messaging:telegram:1"):
    seed_session(owner.persistence, "s", "telegram")
    owner.persistence.admissions.admit(epoch=owner.epoch, principal_id=principal_id, session_id="s",
                                      request_id=request_id, payload={"text": "ask"})
    row = admission_row(owner.persistence.admissions.claim(epoch=owner.epoch, session_id="s"))
    messages = history or [{"role": "assistant", "content": "done"}]
    finish_result(authority_persistence(owner), epoch=owner.epoch, row=row, response="done", outcome="completed",
                  result={"result": {"final_response": "done", "messages": messages,
                                     "completed": True, "model": "m", "estimated_cost_usd": .25,
                                     "_turn_user_row": {"role": "user", "content": "ask"}},
                          "usage": {"input_tokens": 3}})
    return row


def _raw(owner, admission_id):
    with raw_holder(owner.persistence)._read_ctx() as conn:
        row = conn.execute("SELECT value FROM state_meta WHERE key=?",
                           (admission_result_key(admission_id),)).fetchone()
    return None if row is None else row[0]


def _stamp_of(owner, admission_id):
    with raw_holder(owner.persistence)._read_ctx() as conn:
        row = conn.execute("SELECT value FROM state_meta WHERE key=?",
                           (settled_key(admission_id),)).fetchone()
    return None if row is None else int(row[0])


def _backdate(owner, admission_id, seconds):
    raw_holder(owner.persistence)._execute_write(lambda conn: conn.execute(
        "UPDATE state_meta SET value=? WHERE key=?",
        (str(int(time.time()) - seconds), settled_key(admission_id))))


def _admission_status(owner, admission_id):
    with raw_holder(owner.persistence)._read_ctx() as conn:
        return conn.execute("SELECT status FROM session_admissions WHERE admission_id=?",
                            (admission_id,)).fetchone()[0]


def test_settlement_stamps_beside_the_result_without_touching_it(owner):
    row = _settle(owner)
    assert _stamp_of(owner, row["admission_id"]) == pytest.approx(time.time(), abs=60)
    # The stamp is a sibling key: the stored-result payload keeps exactly what it stored.
    raw = json.loads(_raw(owner, row["admission_id"]))
    assert "settled_at" not in raw
    assert admission_result(authority_persistence(owner), row["admission_id"])["result"]["messages"] == [
        {"role": "assistant", "content": "done"}]


def test_past_horizon_trims_transcript_copies_and_keeps_every_reader_field(owner):
    row = _settle(owner, history=[{"role": "assistant", "content": "done"},
                                  {"role": "user", "content": "x" * 4000}])
    admission_id = row["admission_id"]
    _backdate(owner, admission_id, RETRY_HORIZON_S + 60)
    before = len(_raw(owner, admission_id))
    assert owner.persistence.maintenance.trim_settled_results() == 1
    after_raw = _raw(owner, admission_id)
    assert before > len(after_raw)                          # the copies were the weight
    after = json.loads(after_raw)
    assert after["result"]["messages"] == []                 # transcript copies gone
    assert "_turn_user_row" not in after["result"]
    assert after["result"]["final_response"] == "done"       # what receipts report
    assert after["usage"] == {"input_tokens": 3}             # what runs/ledger report
    assert after["result"]["completed"] is True
    assert after["result"]["model"] == "m"
    # The row still exists: "terminal admission, no stored result" is a storage fault to two readers.
    assert admission_result(authority_persistence(owner), admission_id) is not None
    assert _admission_status(owner, admission_id) == "terminal"


def test_a_trimmed_admission_still_passes_the_settlement_precheck(owner):
    """``finish_result`` on an already-settled admission must not turn into ``storage_unavailable``."""
    row = _settle(owner)
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    assert owner.persistence.maintenance.trim_settled_results() == 1
    stored, result = finish_result_precheck(raw_holder(owner.persistence), epoch=owner.epoch,
                                            admission_id=row["admission_id"],
                                            generation=row["generation"])
    assert stored["status"] == "terminal" and result["result"]["final_response"] == "done"


def test_result_inside_horizon_is_not_touched(owner):
    row = _settle(owner)
    assert owner.persistence.maintenance.trim_settled_results() == 0
    assert admission_result(authority_persistence(owner), row["admission_id"])["result"]["messages"] == [
        {"role": "assistant", "content": "done"}]


def test_legacy_row_without_a_stamp_is_stamped_not_trimmed(owner):
    row = _settle(owner)
    raw_holder(owner.persistence)._execute_write(
        lambda conn: conn.execute("DELETE FROM state_meta WHERE key=?", (settled_key(row["admission_id"]),)))
    assert _stamp_of(owner, row["admission_id"]) is None
    assert owner.persistence.maintenance.trim_settled_results() == 0
    assert _stamp_of(owner, row["admission_id"]) is not None
    assert json.loads(_raw(owner, row["admission_id"]))["result"]["messages"]


def test_a_settled_but_empty_reply_keeps_its_transcript(owner):
    """The Bot delivery fallback rebuilds an empty reply from ``messages``; keep that copy."""
    seed_session(owner.persistence, "s", "api_server")
    owner.persistence.admissions.admit(epoch=owner.epoch, principal_id="api", session_id="s",
                                      request_id="empty", payload={"text": "ask"})
    row = admission_row(owner.persistence.admissions.claim(epoch=owner.epoch, session_id="s"))
    finish_result(authority_persistence(owner), epoch=owner.epoch, row=row, response="", outcome="completed",
                  result={"result": {"final_response": "", "messages": [
                      {"role": "assistant", "content": "the only copy of the reply"}]},
                          "usage": {"input_tokens": 3}})
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    assert owner.persistence.maintenance.trim_settled_results() == 0
    saved = admission_result(authority_persistence(owner), row["admission_id"])
    assert saved["result"]["messages"] == [{"role": "assistant", "content": "the only copy of the reply"}]


def test_an_unreadable_row_does_not_block_the_pass(owner):
    row = _settle(owner)
    raw_holder(owner.persistence)._execute_write(lambda conn: conn.execute(
        "INSERT INTO state_meta(key,value) VALUES(?,?)", ("gateway.admission.result.v1.broken", "{not json")))
    raw_holder(owner.persistence)._execute_write(lambda conn: conn.execute(
        "INSERT INTO state_meta(key,value) VALUES(?,?)",
        (settled_key("broken"), str(int(time.time()) - RETRY_HORIZON_S - 60))))
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    assert owner.persistence.maintenance.trim_settled_results() == 1


def test_trim_is_idempotent(owner):
    row = _settle(owner)
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    assert owner.persistence.maintenance.trim_settled_results() == 1
    assert owner.persistence.maintenance.trim_settled_results() == 0


def test_retirement_compaction_does_not_extend_a_row_life(owner):
    row = _settle(owner)
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    stamped = _stamp_of(owner, row["admission_id"])
    # A retirement's compaction re-encodes the result row; the stamp must not move with it.
    from persistence.sqlite.conversation._store.retirement import _compact_result
    raw_holder(owner.persistence)._execute_write(lambda conn: _compact_result(conn, row["admission_id"]))
    assert _stamp_of(owner, row["admission_id"]) == stamped


def test_trim_throttles_and_needs_no_auto_prune(owner):
    row = _settle(owner)
    _backdate(owner, row["admission_id"], RETRY_HORIZON_S + 60)
    first = run_settled_result_trim(owner.persistence, None, vacuum=False)
    assert first["trimmed"] == 1 and not first["skipped"]
    again = run_settled_result_trim(owner.persistence, None, vacuum=False)
    assert again["skipped"] and again["trimmed"] == 0
    assert json.loads(_raw(owner, row["admission_id"]))["result"]["messages"] == []


def _configure_housekeeping(home, monkeypatch, **sessions):
    import hermes_yaml as yaml

    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(yaml.safe_dump({"sessions": {
        "auto_archive": False, "auto_prune": False, "vacuum_after_prune": False,
        "min_interval_hours": 0, **sessions,
    }}), encoding="utf-8")


def _aged_results(owner, *ages_days):
    ids = []
    for days in ages_days:
        admission_id = _settle(owner, request_id=f"age-{days}")["admission_id"]
        _backdate(owner, admission_id, days * 86400)
        ids.append(admission_id)
    return ids


def test_housekeeping_configured_days_and_exact_retry_floor(owner, tmp_path, monkeypatch, caplog):
    from gateway.run import _housekeeping_state_db_maintenance

    three, ten, thirty_one = _aged_results(owner, 3, 10, 31)
    original = {key: _raw(owner, key) for key in (three, ten)}
    _configure_housekeeping(tmp_path, monkeypatch, trim_settled_results_after_days=30)
    _housekeeping_state_db_maintenance()
    assert _raw(owner, ten) == original[ten]
    assert json.loads(_raw(owner, thirty_one))["result"]["messages"] == []

    _configure_housekeeping(tmp_path, monkeypatch, trim_settled_results_after_days=1)
    _housekeeping_state_db_maintenance()
    _housekeeping_state_db_maintenance()
    assert sum("trim_settled_results_after_days" in r.getMessage() for r in caplog.records
               if r.levelname == "WARNING") == 1
    assert _raw(owner, three) == original[three]
    assert json.loads(_raw(owner, ten))["result"]["messages"] == []


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, "abc", -1, 0, 1e308])
def test_housekeeping_invalid_days_uses_default_and_warns_once(owner, tmp_path, monkeypatch, caplog, invalid):
    from gateway.run import _housekeeping_state_db_maintenance

    six, eight = _aged_results(owner, 6, 8)
    before = _raw(owner, six)
    _configure_housekeeping(tmp_path, monkeypatch, trim_settled_results_after_days=invalid)
    _housekeeping_state_db_maintenance()
    _housekeeping_state_db_maintenance()
    assert _raw(owner, six) == before
    assert json.loads(_raw(owner, eight))["result"]["messages"] == []
    warnings = [r for r in caplog.records if r.levelname == "WARNING"
                and "trim_settled_results_after_days" in r.getMessage()]
    assert len(warnings) == 1


def test_housekeeping_opt_out_leaves_legacy_and_aged_results_untouched(owner, tmp_path, monkeypatch):
    from gateway.run import _housekeeping_state_db_maintenance

    legacy, aged = _aged_results(owner, 10, 31)
    raw_holder(owner.persistence)._execute_write(lambda conn: conn.execute(
        "DELETE FROM state_meta WHERE key=?", (settled_key(legacy),)))
    before = [(_raw(owner, key), _stamp_of(owner, key)) for key in (legacy, aged)]
    _configure_housekeeping(tmp_path, monkeypatch, trim_settled_results=False, vacuum_after_prune=True)

    def unexpected_open(*args, **kwargs):
        pytest.fail("all state.db maintenance is disabled; no store or admin handle should open")

    monkeypatch.setattr("persistence_assembly.open_persistence", unexpected_open)
    monkeypatch.setattr("persistence_assembly.open_store_admin", unexpected_open)
    _housekeeping_state_db_maintenance()
    assert [(_raw(owner, key), _stamp_of(owner, key)) for key in (legacy, aged)] == before
    assert owner.persistence.state_meta.get("last_settled_result_trim") is None
    assert owner.persistence.state_meta.get("last_vacuum") is None


def test_housekeeping_warning_dedup_is_scoped_to_profile_and_bad_value(tmp_path, monkeypatch, caplog):
    from gateway.run import _housekeeping_state_db_maintenance

    secondary = tmp_path / "secondary"
    secondary.mkdir()
    for home, value in ((tmp_path, "oops"), (tmp_path, -1), (tmp_path, "oops"), (secondary, "oops")):
        _configure_housekeeping(home, monkeypatch, trim_settled_results_after_days=value)
        _housekeeping_state_db_maintenance()
    warnings = [r for r in caplog.records if r.levelname == "WARNING"
                and "trim_settled_results_after_days" in r.getMessage()]
    assert len(warnings) == 3


@pytest.mark.parametrize("setting", ["false", " Off ", "0", "${TRIM_RESULTS}", "garbage", 1])
def test_housekeeping_text_opt_out_never_opens_store(tmp_path, monkeypatch, caplog, setting):
    from gateway.run import _housekeeping_state_db_maintenance

    monkeypatch.setenv("TRIM_RESULTS", "false")
    _configure_housekeeping(tmp_path, monkeypatch, trim_settled_results=setting)
    monkeypatch.setattr("persistence_assembly.open_persistence", lambda *a, **kw: pytest.fail("opened store"))
    _housekeeping_state_db_maintenance()
    _housekeeping_state_db_maintenance()
    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "trim_settled_results" in r.getMessage()]
    assert len(warnings) == (1 if setting in ("garbage", 1) else 0)


def test_failed_config_read_preserves_aged_results(owner, tmp_path, monkeypatch, caplog):
    from gateway.run import _housekeeping_state_db_maintenance
    from hermes_cli.config import FailedConfigRead

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    aged, = _aged_results(owner, 31)
    before = _raw(owner, aged)
    failed = FailedConfigRead({"sessions": {}}, error=ValueError("invalid YAML"))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: failed)
    _housekeeping_state_db_maintenance()
    assert _raw(owner, aged) == before
    assert owner.persistence.state_meta.get("last_settled_result_trim") is None
    assert any("trim" in r.getMessage().lower() for r in caplog.records if r.levelname == "WARNING")


def test_absent_config_uses_default_trim(owner, tmp_path, monkeypatch):
    from gateway.run import _housekeeping_state_db_maintenance

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    aged, = _aged_results(owner, 31)
    assert not (tmp_path / "config.yaml").exists()
    _housekeeping_state_db_maintenance()
    assert json.loads(_raw(owner, aged))["result"]["messages"] == []


def test_disabled_housekeeping_never_imports_session_retention(tmp_path, monkeypatch):
    import sys
    from gateway.run import _housekeeping_state_db_maintenance

    _configure_housekeeping(tmp_path, monkeypatch, trim_settled_results=" Off ")
    monkeypatch.setitem(sys.modules, "session_retention", None)
    _housekeeping_state_db_maintenance()
    assert not (tmp_path / "state.db").exists()


@pytest.mark.asyncio
async def test_aged_responses_retry_after_eviction_keeps_tool_output_and_compressed_chain(api, owner):
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.session_api_turn import api_execution
    from gateway.session_results import execution_result

    compressed = [
        {"role": "assistant", "content": "summary", "_compressed_summary": True},
        {"role": "user", "content": "read"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": "file contents"},
        {"role": "assistant", "content": "answer"},
    ]
    histories = []

    async def handle(event):
        histories.append(api_execution.get()["history"])
        if event.text == "read":
            execution_result.get()["result"] = {"_compressed": True, "messages": compressed, "final_response": "answer"}
        return "answer"

    owner.runner._handle_message = handle
    app = web.Application()
    app.router.add_post("/v1/responses", api._handle_responses)
    headers = {"Idempotency-Key": "aged"}
    prior = [{"role": "user", "content": "uncompressed history"}]
    body = {"input": "read", "conversation_history": prior}
    async with TestClient(TestServer(app)) as client:
        first = await client.post("/v1/responses", json={**body, "stream": True}, headers=headers)
        assert first.status == 200
        frames = [json.loads(line[6:]) for line in (await first.text()).splitlines() if line.startswith("data: ")]
        original = frames[-1]["response"]
        assert {"function_call", "function_call_output"} <= {item["type"] for item in original["output"]}
        cache = api._current_response_store()
        capacity = cache._max_size
        cache._max_size = 1
        cache.put("unrelated", {"response": {}})
        cache._max_size = capacity
        assert cache.get(original["id"]) is None

        with raw_holder(owner.persistence)._read_ctx() as conn:
            (aid,) = conn.execute("SELECT admission_id FROM session_admissions WHERE request_id GLOB 'responses:*'").fetchone()
        messaging = _settle(owner, request_id="message")["admission_id"]
        for key in (aid, messaging):
            _backdate(owner, key, RETRY_HORIZON_S + 60)
        raw = _raw(owner, aid)
        owner.persistence.maintenance.trim_settled_results()

        retry = await client.post("/v1/responses", json=body, headers=headers)
        assert retry.status == 200
        assert await retry.json() == original
        follow = await client.post("/v1/responses", json={"input": "next", "previous_response_id": original["id"]})
        assert follow.status == 200, await follow.text()
    assert histories == [prior, compressed]
    assert _raw(owner, aid) == raw
    assert json.loads(_raw(owner, messaging))["result"]["messages"] == []


@pytest.mark.parametrize("request_id", ["chat:scope:key", "runs:scope:key", "responses:scope:key"])
def test_api_server_results_keep_their_replay_payload(owner, request_id):
    api = _settle(owner, request_id=request_id, principal_id="api",
                  history=[{"role": "assistant", "content": "done", "reasoning_content": "why"}])["admission_id"]
    messaging = _settle(owner, request_id="message")["admission_id"]
    for key in (api, messaging):
        _backdate(owner, key, RETRY_HORIZON_S + 60)
    raw = _raw(owner, api)
    owner.persistence.maintenance.trim_settled_results()
    assert _raw(owner, api) == raw
    assert json.loads(_raw(owner, messaging))["result"]["messages"] == []
