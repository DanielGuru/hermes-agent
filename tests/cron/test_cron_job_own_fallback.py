"""A cron job may declare its OWN ``fallback_providers`` (local patch, Daniel 2026-11-12).

A pinned job still never borrows the *global* chain (#120312), but an operator who pins a job to a
cheaper model can name the backups that job is allowed to use. ``scoped_fallback_chain`` already
supports a declared chain for delegate children; the cron scheduler just never passed one.

- pinned + own chain  -> walks ONLY its own chain (never the global one), at resolve time and mid-run
- pinned + no chain   -> unchanged: no fallback
- unpinned + own chain -> its own chain replaces the inherited one
- own chain ``[]``    -> fallback disabled even when unpinned
"""

import pytest

from cron.scheduler import _CronJobConfig, _job_fallback_chain, _resolve_job_runtime
from hermes_cli.auth import AuthError
from tests.cron.test_cron_pinned_job_fallback import _CHAIN, _job, _run, _runtime

_OWN = [{"provider": "openai-codex", "model": "gpt-6-astra"}]
_PIN = {"provider": "anthropic", "model": "claude-sonnet-5"}


def test_pinned_job_with_own_chain_gets_only_its_own_chain():
    chain = _job_fallback_chain(_job(**_PIN, fallback_providers=list(_OWN)),
                                {"fallback_providers": list(_CHAIN)})
    assert chain == _OWN


def test_pinned_job_without_own_chain_still_gets_none():
    assert _job_fallback_chain(_job(**_PIN), {"fallback_providers": list(_CHAIN)}) is None


def test_unpinned_job_own_chain_replaces_the_inherited_one():
    chain = _job_fallback_chain(_job(fallback_providers=list(_OWN)),
                                {"fallback_providers": list(_CHAIN)})
    assert chain == _OWN


def test_own_empty_chain_disables_fallback_for_an_unpinned_job():
    assert _job_fallback_chain(_job(fallback_providers=[]),
                               {"fallback_providers": list(_CHAIN)}) is None


def test_resolve_walks_the_pinned_jobs_own_chain_not_the_global_one():
    jc = _CronJobConfig(cfg={"fallback_providers": list(_CHAIN)}, model="claude-sonnet-5",
                        model_cfg={}, cron_default_provider="")
    requested = []

    def resolve(**kwargs):
        requested.append(kwargs.get("requested"))
        if kwargs.get("requested") == "openai-codex":
            return _runtime("openai-codex")
        raise AuthError("No credentials stored")

    from unittest.mock import patch
    with patch("hermes_cli.runtime_provider.resolve_runtime_provider", side_effect=resolve):
        runtime, model = _resolve_job_runtime(
            _job(**_PIN, fallback_providers=list(_OWN)), "pin-job", jc)
    assert (runtime["provider"], model) == ("openai-codex", "gpt-6-astra")
    assert "openrouter" not in requested


def test_pinned_job_agent_gets_its_own_chain_as_the_runtime_ladder(tmp_path):
    """Mid-run (the 2026-11-12 incident: ClaudeCodeLoggedOut raised inside the conversation)."""
    success, error, _requested, agent_kwargs = _run(
        tmp_path, _job(**_PIN, fallback_providers=list(_OWN)))
    assert (success, error) == (True, None)
    assert agent_kwargs["fallback_model"] == _OWN
