"""Tests of ``python -m hpcmon`` (:mod:`hpcmon.__main__`).

The in-process tests run the module with :mod:`runpy` after replacing
``hpcmon.cli.main`` (``__main__`` imports it when executed).  The subprocess
tests only use options that never contact a cluster, and run with an empty
``PATH`` so that no ``ssh`` binary could be found anyway.  They run without
``$HPCMON_CONFIG`` (which makes the file it names mandatory) unless a test
sets it, and with ``$XDG_CONFIG_HOME`` in the test's temporary directory.
"""

from __future__ import annotations

import importlib
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pytest

import hpcmon
import hpcmon.cli

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def fake_main(monkeypatch):
    """Replace ``hpcmon.cli.main`` and forget any cached ``hpcmon.__main__``."""
    calls: List[tuple] = []
    state = {"status": 0}

    def main(*args, **kwargs):
        calls.append((args, kwargs))
        return state["status"]

    monkeypatch.setattr(hpcmon.cli, "main", main)
    monkeypatch.delitem(sys.modules, "hpcmon.__main__", raising=False)
    monkeypatch.delattr(hpcmon, "__main__", raising=False)
    yield calls, state
    # Do not leave a module bound to the fake ``main`` behind.
    sys.modules.pop("hpcmon.__main__", None)
    if hasattr(hpcmon, "__main__"):
        delattr(hpcmon, "__main__")


@pytest.mark.parametrize("status", [0, 1, 2, 130])
def test_running_the_package_exits_with_the_status_of_main(fake_main, status):
    calls, state = fake_main
    state["status"] = status
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("hpcmon", run_name="__main__")
    assert excinfo.value.code == status
    assert calls == [((), {})]  # main() reads the arguments from sys.argv


def test_importing_the_main_module_does_not_run_the_cli(fake_main):
    calls, _ = fake_main
    module = importlib.import_module("hpcmon.__main__")
    assert calls == []
    assert module.main is hpcmon.cli.main


def test_run_module_under_its_own_name_does_not_run_the_cli(fake_main):
    calls, _ = fake_main
    namespace = runpy.run_module("hpcmon.__main__", run_name="hpcmon.__main__")
    assert calls == []
    assert namespace["main"] is hpcmon.cli.main


def _run_hpcmon(tmp_path, *args: str, extra_env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir(exist_ok=True)
    env = dict(os.environ)
    # The conftest sets HPCMON_CONFIG for the tests: drop it explicitly.
    for var in ("HPCMON_CONFIG", "FORCE_COLOR", "NO_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE"):
        env.pop(var, None)
    env.update(
        {
            "PATH": str(empty_bin),
            "PYTHONPATH": str(REPO_ROOT),
            "XDG_CONFIG_HOME": str(tmp_path / "xdg"),
            "COLUMNS": "300",
        }
    )
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "hpcmon", *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_python_dash_m_hpcmon_prints_the_version(tmp_path):
    result = _run_hpcmon(tmp_path, "--version")
    assert result.returncode == 0
    assert result.stdout == f"hpcmon {hpcmon.__version__}\n"
    assert result.stderr == ""


def test_python_dash_m_hpcmon_show_config_uses_builtin_defaults(tmp_path):
    result = _run_hpcmon(tmp_path, "--show-config")
    assert result.returncode == 0
    assert result.stdout.splitlines()[0] == (
        f"Configuration: built-in defaults (no file at {tmp_path / 'xdg' / 'hpcmon' / 'config.toml'})"
    )
    assert result.stderr == ""


def test_python_dash_m_hpcmon_requires_the_file_named_by_hpcmon_config(tmp_path):
    missing = tmp_path / "from-env.toml"
    result = _run_hpcmon(tmp_path, "--show-config", extra_env={"HPCMON_CONFIG": str(missing)})
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"error: configuration file not found: {missing}\n"


def test_python_dash_m_hpcmon_reads_the_file_named_by_hpcmon_config(tmp_path):
    path = tmp_path / "from-env.toml"
    path.write_text('[clusters.hpc]\ntype = "slurm"\n', encoding="utf-8")
    result = _run_hpcmon(tmp_path, "--show-config", extra_env={"HPCMON_CONFIG": str(path)})
    assert result.returncode == 0
    assert result.stdout.splitlines()[0] == f"Configuration: {path}"
    assert result.stderr == ""


def test_python_dash_m_hpcmon_parser_errors_exit_with_status_2(tmp_path):
    result = _run_hpcmon(tmp_path, "--force")
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.endswith("hpcmon: error: --force can only be used with --init-config\n")


def test_python_dash_m_hpcmon_propagates_usage_errors(tmp_path):
    missing = tmp_path / "missing.toml"
    result = _run_hpcmon(tmp_path, "-c", str(missing))
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr == f"error: configuration file not found: {missing}\n"
