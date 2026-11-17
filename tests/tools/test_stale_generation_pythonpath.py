"""A process that booted on an older dependency generation keeps that generation's
site-packages in its launcher PYTHONPATH. After a sync commits a newer generation the old
entry no longer matched the runtime's site-packages, so it leaked into every child and a
foreign interpreter (a user's 3.13 venv) imported 3.14 compiled extensions (PIL _imaging)."""
from pathlib import Path

from tools.environments import local_pythonpath as lp


def _strip(monkeypatch, gens: Path, pythonpath: str) -> str | None:
    monkeypatch.setattr(lp, "_dependency_generations_root", lambda: gens)
    monkeypatch.setattr(lp, "_get_hermes_site_packages", lambda env: [])
    env = {"PYTHONPATH": pythonpath}
    lp._strip_hermes_owned_pythonpath(env)
    return env.get("PYTHONPATH")


def test_stale_generation_site_packages_is_stripped(monkeypatch, tmp_path):
    gens = tmp_path / "installs" / "key" / "environments"
    old = gens / "07e02c" / "venv" / "lib" / "python3.14" / "site-packages"
    assert _strip(monkeypatch, gens, str(old)) is None


def test_user_paths_and_descendants_are_kept(monkeypatch, tmp_path):
    gens = tmp_path / "installs" / "key" / "environments"
    user = tmp_path / "myproj" / "lib"
    descendant = gens / "07e02c" / "venv" / "lib" / "python3.14" / "site-packages" / "extra"
    workspace = gens / "07e02c" / "workspace"
    old = gens / "07e02c" / "venv" / "lib" / "python3.14" / "site-packages"
    pp = ":".join(map(str, [user, old, descendant, workspace]))
    assert _strip(monkeypatch, gens, pp) == ":".join(map(str, [user, descendant, workspace]))


def test_no_generations_root_keeps_everything(monkeypatch, tmp_path):
    p = str(tmp_path / "environments" / "x" / "venv" / "lib" / "python3.14" / "site-packages")
    assert _strip(monkeypatch, None, p) == p
