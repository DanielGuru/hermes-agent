"""agent.pre_verify_every_turn lets pre_verify hooks see turns with no file edits."""
from types import SimpleNamespace
from unittest.mock import patch

from agent import turn_stop_gates
from agent.verify_hooks import pre_verify_every_turn


def _agent(edited=()):
    return SimpleNamespace(_turn_file_mutation_paths=set(edited), session_id="s", platform="telegram",
                           model="m", _resolved_is_coding=False)


def test_flag_default_off():
    assert pre_verify_every_turn({}) is False
    assert pre_verify_every_turn({"agent": {"pre_verify_every_turn": True}}) is True


def _run(agent, every):
    with patch("hermes_cli.lifecycle.has_hook", return_value=True), \
         patch("agent.verify_hooks.max_verify_nudges", return_value=3), \
         patch("agent.verify_hooks.pre_verify_every_turn", return_value=every), \
         patch("hermes_cli.plugins.get_pre_verify_continue_message", return_value="go on") as hook:
        out = turn_stop_gates._pre_verify_nudge(agent, "I'm starting now.", 0)
    return out, hook


def test_no_edits_flag_off_skips_hook():
    out, hook = _run(_agent(), False)
    assert out is None and not hook.called


def test_no_edits_flag_on_calls_hook():
    out, hook = _run(_agent(), True)
    assert out == "go on" and hook.call_args.kwargs["changed_paths"] == []


def test_edits_still_call_hook_when_flag_off():
    out, hook = _run(_agent(["a.py"]), False)
    assert out == "go on"
