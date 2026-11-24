"""Reclaim aged deletion tombstones and project aged admission results without deleting them."""

from __future__ import annotations

import json
import time

from persistence.core.canonical import canonical_json

from ..conversation._store.retirement import RETIRED_PREFIX
from ..conversation._store.terminal import (
    ADMISSION_PREFIX, RESULT_PREFIX, SETTLED_PREFIX, WORKER_PREFIX, encode_result, identity_key, load_result, settled_key,
)
from ..gateway._routes import _EXPIRED, RETRY_HORIZON_S, ROUTE_FENCE_PREFIX, _reclaim_expired

__all__ = ["backfill_settled_stamps", "prune_retired_tombstones", "settled_result_candidates", "trim_settled_results"]

# Copies of the turn's transcript inside a stored result: dead once the exact-retry horizon passes,
# because the live transcript in ``messages`` already holds them.
_TRANSCRIPT_COPY_KEYS = frozenset({'_turn_user_row', 'last_reasoning', 'tools'})


def prune_retired_tombstones(conn, older_than=RETRY_HORIZON_S):
    """Reclaim deletion tombstones retired more than *older_than* seconds ago: the terminal admission
    with its identity pointer and compacted result, terminal workers and a sweep's (``expires``)
    retired-session marker or shared routing fence. A user delete's marker is the deletion fence itself and is never
    reclaimed: it holds only ``retired_at``, and without it a deleted id could be recreated,
    re-admitted or reissued as a Bot Chat generation.
    Tombstones written before ``retired_at`` existed are stamped now and age from this pass."""
    prefixes = (ADMISSION_PREFIX, WORKER_PREFIX)
    now = time.time()
    for prefix in prefixes:
        conn.execute("UPDATE state_meta SET value=json_set(value, '$.retired_at', ?) WHERE key GLOB ? "
                     "AND json_valid(value) AND json_extract(value, '$.retired_at') IS NULL", (now, prefix + '*'))
    cutoff = now - older_than
    for key, raw in conn.execute(f'SELECT key,value FROM state_meta WHERE key GLOB ? AND {_EXPIRED}',
                                 (ADMISSION_PREFIX + '*', cutoff)).fetchall():
        row = json.loads(raw)
        conn.execute('DELETE FROM state_meta WHERE key IN (?,?,?,?)', (key, RESULT_PREFIX + row['admission_id'],
                     settled_key(row['admission_id']),
                     identity_key(row['principal_id'], row['target_session_id'], row['request_id'])))
    conn.execute(f'DELETE FROM state_meta WHERE key GLOB ? AND {_EXPIRED}', (WORKER_PREFIX + '*', cutoff))
    for prefix in (RETIRED_PREFIX, ROUTE_FENCE_PREFIX):
        _reclaim_expired(conn, prefix, cutoff)


# A stamp is the TEXT spelling of an integer second. Round-tripping rejects e.g. "unknown"
# and "7.5", which SQLite would cast to an integer. Even the *end* of that second must be old.
_AGED_STAMP = ("s.value = CAST(CAST(s.value AS INTEGER) AS TEXT) "
               "AND CAST(s.value AS INTEGER) + 1 <= :cutoff")
SNAPSHOT_PAGE_SIZE = 1000


def settled_result_candidates(holder, cutoff, progress=None):
    """Collect keys, releasing the reader after each page (also releases the mutex in DELETE mode)."""
    missing, candidates = [], []
    # Fence the scan at entry: don't chase newly appended keys, including inserts inside the key range.
    with holder._read_ctx() as conn:
        last, high_rowid = conn.execute("""
            SELECT max(key), (SELECT max(rowid) FROM state_meta)
            FROM state_meta WHERE key > ? AND key < ?
        """, (RESULT_PREFIX, RESULT_PREFIX[:-1] + "/")).fetchone()
    after = RESULT_PREFIX
    while last is not None and after < last:
        with holder._read_ctx() as conn:
            page = conn.execute(f"""
                SELECT r.key, r.rowid <= :high_rowid, s.key IS NULL, ({_AGED_STAMP}) FROM state_meta r
                LEFT JOIN state_meta s ON s.key = :prefix || substr(r.key, :start)
                WHERE r.key > :after AND r.key <= :last ORDER BY r.key LIMIT :limit
            """, {"after": after, "last": last, "high_rowid": high_rowid, "start": len(RESULT_PREFIX) + 1,
                  "prefix": SETTLED_PREFIX, "cutoff": cutoff, "limit": SNAPSHOT_PAGE_SIZE}).fetchall()
        for key, existed, unstamped, aged in page:
            if existed and (unstamped or aged):
                (missing if unstamped else candidates).append(key[len(RESULT_PREFIX):])
        if progress is not None:
            progress()
        if len(page) < SNAPSHOT_PAGE_SIZE:
            break
        after = page[-1][0]
    return missing, candidates


_BACKFILL_STAMP = """
    INSERT INTO state_meta(key, value)
    SELECT :stamp, :now
    WHERE EXISTS (SELECT 1 FROM state_meta WHERE key = :result)
    ON CONFLICT(key) DO NOTHING
"""


def backfill_settled_stamps(conn, ids, now):
    """Age from first observation: a transcript need not contain the turn that settled."""
    return sum(conn.execute(_BACKFILL_STAMP, {
        "stamp": settled_key(aid), "result": RESULT_PREFIX + aid, "now": int(now),
    }).rowcount for aid in ids)


def _project_result(raw):
    """Return the smaller UTF-8 payload, or None when the row must be preserved byte-for-byte."""
    from gateway.response_filters import is_intentional_silence_response

    try:
        value = load_result(raw.decode("utf-8"))
        inner = value.get("result") if isinstance(value, dict) else None
        if not isinstance(inner, dict):
            return None
        reply = inner.get("final_response")
        if not isinstance(reply, str) or not reply.strip() or is_intentional_silence_response(reply):
            # The Bot receipt falls back to these messages to reconstruct a visible reply.
            return None
        if "messages" in inner and not isinstance(inner["messages"], list):
            return None
        if not inner.get("messages") and _TRANSCRIPT_COPY_KEYS.isdisjoint(inner):
            return None
        for key in _TRANSCRIPT_COPY_KEYS:
            inner.pop(key, None)
        inner["messages"] = []
        # An empty list is not a replacement history: CLI receipt recovery would adopt it whole.
        inner.pop("_compressed", None)
        return canonical_json(encode_result(value)).encode("utf-8")
    except (ValueError, TypeError, RecursionError):
        # Includes malformed UTF-8, lone surrogates, excessive nesting and nonfinite retained
        # numbers. No SQLite operations are inside this per-payload guard.
        return None


def trim_settled_results(conn, ids, cutoff, max_bytes):
    """Re-read both result and stamp under the writer; return (examined, rewritten).

    A terminal admission without its result is a storage fault for finish_result_precheck and
    hosted-task recovery. Preserve the result and all non-transcript fields for their readers.
    """
    examined = trimmed = read_bytes = 0
    for aid in ids:
        examined += 1
        key = RESULT_PREFIX + aid
        row = conn.execute(
            f"""SELECT CAST(r.value AS BLOB) FROM state_meta r JOIN state_meta s ON s.key = :stamp
                WHERE r.key = :result AND {_AGED_STAMP}
                AND NOT EXISTS (SELECT 1 FROM session_admissions WHERE admission_id = :aid
                                AND principal_id = 'api')""",
            {"stamp": settled_key(aid), "result": key, "cutoff": cutoff, "aid": aid},
        ).fetchone()
        if row is None or row[0] is None:
            continue
        projected = _project_result(row[0])
        if projected is not None:
            conn.execute("UPDATE state_meta SET value=CAST(? AS TEXT) WHERE key=?", (projected, key))
            trimmed += 1
        read_bytes += len(row[0])
        if read_bytes >= max_bytes:
            break
    return examined, trimmed
