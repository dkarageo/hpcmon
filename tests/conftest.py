"""Shared helpers for the hpcmon tests.

No test may connect to a real cluster: remote commands are replaced by
:class:`FakeRunner`, which returns canned :class:`CommandResult` objects.
The fixtures in ``tests/fixtures`` are real (anonymised) outputs of the
cluster tools.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import pytest

from hpcmon.config import ClusterConfig
from hpcmon.runner import CommandResult

FIXTURES = Path(__file__).parent / "fixtures"

#: The date on which the fixtures were captured.
TODAY = dt.date(2026, 11, 18)
NOW = dt.datetime(2026, 11, 18, 20, 30, 0)


def read_fixture(name: str) -> str:
    """Return the contents of ``tests/fixtures/<name>``."""
    return (FIXTURES / name).read_text(encoding="utf-8")


def ok(stdout: str = "", stderr: str = "") -> CommandResult:
    """A successful command result."""
    return CommandResult(stdout=stdout, stderr=stderr, returncode=0)


def failed(returncode: Optional[int] = 1, stdout: str = "", stderr: str = "") -> CommandResult:
    """A failed (or, with ``returncode=None``, unfinished) command result."""
    return CommandResult(stdout=stdout, stderr=stderr, returncode=returncode)


class FakeRunner:
    """Stand-in for :class:`hpcmon.runner.SSHRunner`.

    :param results: Results by command key; plain strings are turned into
        successful results.  Keys absent from ``results`` are absent from the
        returned mapping as well (as if the command never ran).
    :param error: Exception raised by :meth:`run` instead of returning.
    """

    def __init__(
        self,
        results: Optional[Mapping[str, Union[str, CommandResult]]] = None,
        error: Optional[BaseException] = None,
    ) -> None:
        self.results = {
            key: ok(value) if isinstance(value, str) else value for key, value in (results or {}).items()
        }
        self.error = error
        self.calls: List[Dict[str, str]] = []

    def run(self, commands: Mapping[str, str]) -> Dict[str, CommandResult]:
        self.calls.append(dict(commands))
        if self.error is not None:
            raise self.error
        return {key: self.results[key] for key in commands if key in self.results}


def make_config(name: str = "test", type: Optional[str] = None, **kwargs) -> ClusterConfig:
    """Build a :class:`ClusterConfig` with sensible defaults."""
    return ClusterConfig(name=name, type=type or name, ssh=kwargs.pop("ssh", ["ssh", name]), **kwargs)


@pytest.fixture
def fixture_text():
    """The :func:`read_fixture` helper as a pytest fixture."""
    return read_fixture


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path_factory):
    """Keep every test away from the user's configuration and from real SSH."""
    config_home = tmp_path_factory.mktemp("config-home")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("HPCMON_CONFIG", str(config_home / "hpcmon" / "config.toml"))

    import hpcmon.runner

    real_popen = hpcmon.runner.subprocess.Popen

    def guarded_popen(argv, *args, **kwargs):
        if argv and "ssh" in Path(str(argv[0])).name:
            raise AssertionError(f"tests must not run ssh: {argv!r}")
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(hpcmon.runner.subprocess, "Popen", guarded_popen)
