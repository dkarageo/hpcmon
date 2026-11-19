"""Tests of the command line interface (:mod:`hpcmon.cli`).

``main`` is driven end to end without SSH: ``hpcmon.cli.create_cluster`` is
replaced by a factory that builds the real cluster classes on top of a
:class:`FakeRunner` loaded with the captured tool outputs, or by stub clusters
when the scheduling of the queries (sequential/parallel) is under test.  The
clock of the CLI is frozen at :data:`NOW_LOCAL` and the console width is pinned
so that the rich output is deterministic.

``$HPCMON_CONFIG`` makes the configuration file mandatory, so it is unset by
default here: the configuration is then looked up (optionally) at
``$XDG_CONFIG_HOME/hpcmon/config.toml``, in an empty directory of the test.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
import logging
import os
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest
import rich.console
from rich.console import Console

import hpcmon
from hpcmon import cli
from hpcmon.clusters import CLUSTER_TYPES, create_cluster
from hpcmon.clusters.base import HOME_COMMAND
from hpcmon.clusters.slurm import SQUEUE_COMMAND
from hpcmon.config import DEFAULT_CONFIG_TEMPLATE, SECTIONS, Config, ConfigError, load_config
from hpcmon.models import ClusterReport
from hpcmon.runner import RemoteError
from tests.conftest import NOW, TODAY, FakeRunner, failed, make_config, ok, read_fixture

try:
    import termios
except ImportError:  # pragma: no cover - not available on Windows
    termios = None  # type: ignore[assignment]

#: For the tests of the terminal settings, which only exist on POSIX systems.
requires_termios = pytest.mark.skipif(termios is None, reason="termios is not available")
#: ``termios.error``, or a stand-in so that the parameters can be built anyway.
TERMIOS_ERROR = getattr(termios, "error", OSError)

DEFAULT_NAMES = ["lumi", "jupiter", "leonardo", "meluxina"]
SYSTEMS = {"lumi": "LUMI", "jupiter": "JUPITER", "leonardo": "Leonardo", "meluxina": "MeluXina"}
GPUS = {
    "lumi": "4× AMD MI250X 128 GB (2 GCDs each)",
    "jupiter": "4× NVIDIA GH200 96 GB",
    "leonardo": "4× NVIDIA A100 64 GB",
    "meluxina": "4× NVIDIA A100 40 GB",
}
WAIT = 5.0  # seconds; only reached when a synchronisation test is broken

#: Time zone of the frozen clock (fixed, whatever the time zone of the host).
CET = dt.timezone(dt.timedelta(hours=1), "CET")
#: :data:`NOW` as returned by ``datetime.now().astimezone()`` in the CLI.
NOW_LOCAL = NOW.replace(tzinfo=CET)

LUMI_COMPUTE = {"allocations": "lumi-allocations", "projectinfo": "lumi-ldap-projectinfo"}
LUMI_STORAGE = {"workspaces": "lumi-workspaces", "home": HOME_COMMAND}
LUMI_JOBS = {"jobs": SQUEUE_COMMAND}


# -- helpers -----------------------------------------------------------------


def fixture_results(cluster_type: str) -> Dict[str, object]:
    """The captured outputs of every remote command of ``cluster_type``."""
    squeue = read_fixture("slurm/squeue.txt")
    if cluster_type == "lumi":
        return {
            "allocations": read_fixture("lumi/allocations.txt"),
            "projectinfo": read_fixture("lumi/projectinfo.txt"),
            "workspaces": read_fixture("lumi/workspaces.txt"),
            "home": "/users/jdoe1234\n",
            "jobs": squeue,
        }
    if cluster_type == "jupiter":
        return {
            "cpuquota": read_fixture("jupiter/cpuquota.json"),
            "userdata": read_fixture("jupiter/userdata.json"),
            "projectdata": read_fixture("jupiter/projectdata.json"),
            "home": "/p/home/jusers/jdoe123456789/jupiter\n",
            "jobs": "",
        }
    if cluster_type == "leonardo":
        return {
            "saldo": read_fixture("leonardo/saldo.txt"),
            "saldo_dcgp": read_fixture("leonardo/saldo_dcgp.txt"),
            "cindata": read_fixture("leonardo/cindata.txt"),
            "jobs": "",
        }
    if cluster_type == "meluxina":
        return {"myquota": read_fixture("meluxina/myquota.txt"), "jobs": ""}
    return {"jobs": squeue}


class ClusterFactory:
    """Replacement of ``create_cluster`` building real clusters on fake runners."""

    def __init__(self) -> None:
        self.configs: list = []
        self.runners: Dict[str, FakeRunner] = {}
        #: Results by cluster *name*, overriding the fixtures of its type.
        self.results: Dict[str, Dict[str, object]] = {}
        #: Exceptions raised by the runner, by cluster name.
        self.errors: Dict[str, BaseException] = {}

    def __call__(self, config):
        self.configs.append(config)
        results = self.results.get(config.name, fixture_results(config.type))
        runner = FakeRunner(results, error=self.errors.get(config.name))
        self.runners[config.name] = runner
        return CLUSTER_TYPES[config.type](config, runner=runner)


class StubCluster:
    """Minimal cluster recording how (and in which thread) it is collected."""

    display_name = "Stub system"
    gpu_label = "2× Stub GPU"

    def __init__(
        self,
        name: str,
        before: Optional[Callable[[], None]] = None,
        error: Optional[BaseException] = None,
        finish_log: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.before = before
        self.error = error
        self.finish_log = finish_log if finish_log is not None else []
        self.calls: List[dict] = []
        self.finished = threading.Event()

    def collect(self, sections, now=None):
        self.calls.append({"sections": list(sections), "now": now, "thread": threading.get_ident()})
        if self.before is not None:
            self.before()
        if self.error is not None:
            raise self.error
        self.finish_log.append(self.name)
        self.finished.set()
        return ClusterReport(name=self.name, system=self.display_name, sections=list(sections), collected_at=now)


def wait_for(event: threading.Event) -> None:
    """Wait for ``event``; a timeout becomes an error in the cluster's report."""
    if not event.wait(timeout=WAIT):
        raise RuntimeError("synchronisation timed out")


def run_cli(capsys, *argv: str):
    """Run ``main(argv)`` and return its exit status, stdout and stderr."""
    rc = cli.main(list(argv))
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def write_config(tmp_path: Path, text: str, name: str = "config.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def table_rows(text: str) -> List[List[str]]:
    """The body rows of a rich (heavy-head) table printed in ``text``."""
    return [[cell.strip() for cell in line.strip().strip("│").split("│")] for line in text.splitlines() if line.startswith("│")]


def default_location() -> Path:
    """The (optional) default configuration file, without ``$HPCMON_CONFIG``."""
    return Path(os.environ["XDG_CONFIG_HOME"]) / "hpcmon" / "config.toml"


# -- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cli_environment(monkeypatch, _isolate):
    """Pin the console, and make unexpected cluster creation fail loudly.

    Depends on the conftest's ``_isolate`` so that ``$HPCMON_CONFIG``, which
    it sets, is removed afterwards (the file it names would be required).
    """
    monkeypatch.delenv("HPCMON_CONFIG", raising=False)
    monkeypatch.setenv("COLUMNS", "400")
    for var in ("FORCE_COLOR", "NO_COLOR", "TTY_COMPATIBLE", "TTY_INTERACTIVE", "LINES"):
        monkeypatch.delenv(var, raising=False)

    def unexpected_create_cluster(config, runner=None):
        raise AssertionError(f"create_cluster({config.name!r}) called without a fake")

    monkeypatch.setattr(cli, "create_cluster", unexpected_create_cluster)


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """Freeze the clock of the CLI, whatever the time zone of the host.

    ``datetime.now()`` returns a naive local time whose ``astimezone()``
    gives ``clock["now"]`` (an aware datetime, :data:`NOW_LOCAL` by default);
    set ``clock["now"]`` to move the clock.  ``clock["astimezone"]`` counts
    the conversions to the local time zone.
    """
    state = {"now": NOW_LOCAL, "astimezone": 0}

    class LocalNaive(dt.datetime):
        def astimezone(self, tz=None):
            state["astimezone"] += 1
            return state["now"] if tz is None else state["now"].astimezone(tz)

    class FrozenDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return state["now"].astimezone(tz)
            return LocalNaive.combine(state["now"].date(), state["now"].time())

    monkeypatch.setattr(cli, "dt", types.SimpleNamespace(datetime=FrozenDatetime))
    return state


@pytest.fixture
def factory(monkeypatch) -> ClusterFactory:
    instance = ClusterFactory()
    monkeypatch.setattr(cli, "create_cluster", instance)
    return instance


@pytest.fixture
def stubs(monkeypatch) -> Dict[str, StubCluster]:
    """Make ``create_cluster`` return (and record) :class:`StubCluster` objects.

    Pre-populate the returned mapping to customise the stub of a cluster.
    """
    created: Dict[str, StubCluster] = {}

    def create(config, runner=None):
        return created.setdefault(config.name, StubCluster(config.name))

    monkeypatch.setattr(cli, "create_cluster", create)
    return created


@pytest.fixture
def basic_config_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: calls.append(kwargs))
    return calls


# -- build_parser ------------------------------------------------------------


def test_parser_defaults():
    args = cli.build_parser().parse_args([])
    assert cli.build_parser().prog == "hpcmon"
    assert vars(args) == {
        "clusters": [],
        "config": None,
        "compute": False,
        "storage": False,
        "jobs": False,
        "json": False,
        "timeout": None,
        "sequential": False,
        "no_color": False,
        "debug": False,
        "init_config": False,
        "force": False,
        "show_config": False,
    }


def test_parser_accepts_every_option():
    args = cli.build_parser().parse_args(
        [
            "lumi", "jupiter", "-c", "my.toml", "--compute", "--storage", "--jobs", "--json", "-t", "2.5",
            "--sequential", "--no-color", "--debug", "--init-config", "--force", "--show-config",
        ]
    )
    assert args.clusters == ["lumi", "jupiter"]
    assert args.config == Path("my.toml")
    assert (args.compute, args.storage, args.jobs, args.json) == (True, True, True, True)
    assert args.timeout == 2.5
    assert (args.sequential, args.no_color, args.debug) == (True, True, True)
    assert (args.init_config, args.force, args.show_config) == (True, True, True)


def test_parser_long_option_forms():
    args = cli.build_parser().parse_args(["--config", "x.toml", "--timeout", "30"])
    assert args.config == Path("x.toml")
    assert args.timeout == 30.0


@pytest.mark.parametrize("argv", [["-t", "-1"], ["--timeout=-1"]])
def test_parser_accepts_negative_timeouts_for_main_to_reject(argv):
    assert cli.build_parser().parse_args(argv).timeout == -1.0


@pytest.mark.parametrize("flag", ["-V", "--version"])
def test_version_prints_program_and_version(capsys, flag):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([flag])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out == f"hpcmon {hpcmon.__version__}\n"


def test_main_reads_sys_argv_by_default(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["hpcmon", "--version"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 0
    assert capsys.readouterr().out == "hpcmon 0.1.0\n"


def test_help_describes_configuration_location(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: hpcmon")
    assert "$HPCMON_CONFIG" in out
    assert "sections (default: all)" in out


def test_non_numeric_timeout_is_rejected_by_argparse(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["-t", "abc"])
    assert excinfo.value.code == 2
    assert "argument -t/--timeout: invalid float value: 'abc'" in capsys.readouterr().err


def test_unknown_option_is_rejected_by_argparse(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--bogus"])
    assert excinfo.value.code == 2
    assert "unrecognized arguments: --bogus" in capsys.readouterr().err


# -- select_clusters -------------------------------------------------------------


@pytest.fixture
def three_clusters() -> Config:
    return Config(
        clusters={
            "a": make_config("a", type="slurm"),
            "b": make_config("b", type="slurm", enabled=False),
            "c": make_config("c", type="slurm"),
        }
    )


def test_select_clusters_defaults_to_enabled_clusters_in_config_order(three_clusters):
    assert cli.select_clusters(three_clusters, []) == ["a", "c"]


def test_select_clusters_keeps_command_line_order_and_removes_duplicates(three_clusters):
    assert cli.select_clusters(three_clusters, ["c", "a", "c", "a"]) == ["c", "a"]


def test_select_clusters_allows_explicitly_named_disabled_cluster(three_clusters):
    assert cli.select_clusters(three_clusters, ("b",)) == ["b"]


def test_select_clusters_returns_nothing_when_all_disabled():
    config = Config(clusters={"a": make_config("a", type="slurm", enabled=False)})
    assert cli.select_clusters(config, []) == []


def test_select_clusters_rejects_unknown_names(three_clusters):
    with pytest.raises(ConfigError) as excinfo:
        cli.select_clusters(three_clusters, ["a", "x", "y"])
    assert str(excinfo.value) == "unknown cluster(s): x, y (configured: a, b, c)"


def test_select_clusters_unknown_name_with_no_configured_cluster():
    with pytest.raises(ConfigError) as excinfo:
        cli.select_clusters(Config(clusters={}), ["lumi"])
    assert str(excinfo.value) == "unknown cluster(s): lumi (configured: none)"


# -- collect_reports / _collect --------------------------------------------------


def test_sequential_collection_is_lazy_and_runs_in_calling_thread():
    a, b = StubCluster("a"), StubCluster("b")
    reports = cli.collect_reports([a, b], ["jobs"], parallel=False, now=NOW)
    assert a.calls == [] and b.calls == []  # nothing runs before iteration

    first = next(reports)
    assert first.name == "a"
    assert b.calls == []  # the next cluster is queried only when needed
    second = next(reports)
    assert second.name == "b"
    with pytest.raises(StopIteration):
        next(reports)

    main_thread = threading.get_ident()
    assert [call["thread"] for call in a.calls + b.calls] == [main_thread, main_thread]
    assert a.calls[0]["sections"] == ["jobs"]
    assert a.calls[0]["now"] is NOW


def test_single_cluster_is_collected_without_thread_pool(monkeypatch):
    def no_pool(*args, **kwargs):
        raise AssertionError("the thread pool must not be used for a single cluster")

    monkeypatch.setattr(cli, "ThreadPoolExecutor", no_pool)
    only = StubCluster("only")
    reports = list(cli.collect_reports([only], list(SECTIONS), parallel=True, now=NOW))
    assert [report.name for report in reports] == ["only"]
    assert only.calls[0]["thread"] == threading.get_ident()


@pytest.mark.parametrize("parallel", [True, False])
def test_collect_reports_without_clusters_yields_nothing(parallel):
    assert list(cli.collect_reports([], list(SECTIONS), parallel=parallel, now=NOW)) == []


def test_parallel_collection_keeps_order_when_first_cluster_finishes_last():
    finish_log: List[str] = []
    second = StubCluster("second", finish_log=finish_log)
    first = StubCluster("first", before=lambda: wait_for(second.finished), finish_log=finish_log)

    reports = list(cli.collect_reports([first, second], ["compute"], parallel=True, now=NOW))

    assert [report.errors for report in reports] == [[], []]
    assert [report.name for report in reports] == ["first", "second"]
    assert finish_log == ["second", "first"]
    main_thread = threading.get_ident()
    assert all(call["thread"] != main_thread for call in first.calls + second.calls)
    assert all(report.collected_at is NOW and report.sections == ["compute"] for report in reports)


def test_parallel_collection_queries_all_clusters_at_once():
    # Every cluster waits until all of them have started: this only completes
    # if there is one worker per cluster.
    barrier = threading.Barrier(3, timeout=WAIT)
    clusters = [StubCluster(name, before=barrier.wait) for name in ("a", "b", "c")]
    reports = list(cli.collect_reports(clusters, list(SECTIONS), parallel=True, now=NOW))
    assert [(report.name, report.errors) for report in reports] == [("a", []), ("b", []), ("c", [])]


def test_parallel_collection_yields_each_report_once_it_and_its_predecessors_are_done():
    release, c_started = threading.Event(), threading.Event()

    def gate() -> None:
        c_started.set()
        wait_for(release)

    a, b, c = StubCluster("a"), StubCluster("b"), StubCluster("c", before=gate)
    reports = cli.collect_reports([a, b, c], ["jobs"], parallel=True, now=NOW)
    try:
        assert next(reports).name == "a"
        assert next(reports).name == "b"
        assert c_started.wait(WAIT)
        assert not c.finished.is_set()  # a and b were yielded while c still runs
    finally:
        release.set()
    last = next(reports)
    assert (last.name, last.errors) == ("c", [])
    with pytest.raises(StopIteration):
        next(reports)


def test_parallel_collection_isolates_a_crashing_cluster():
    clusters = [StubCluster("bad", error=KeyError("oops")), StubCluster("good")]
    reports = list(cli.collect_reports(clusters, ["jobs"], parallel=True, now=NOW))
    assert [report.name for report in reports] == ["bad", "good"]
    assert reports[0].errors == ["internal error (KeyError: 'oops')"]
    assert reports[1].ok


def test_collect_turns_unexpected_exception_into_internal_error_report(caplog):
    cluster = StubCluster("broken", error=RuntimeError("kaput"))
    with caplog.at_level(logging.DEBUG, logger="hpcmon.cli"):
        report = cli._collect(cluster, ["storage", "jobs"], NOW)

    assert report.name == "broken"
    assert report.system == "Stub system"
    assert report.sections == ["storage", "jobs"]
    assert report.collected_at is NOW
    assert report.errors == ["internal error (RuntimeError: kaput)"]
    assert report.failed_sections == ["storage", "jobs"]
    assert report.failed_sections is not report.sections
    assert report.warnings == []
    assert (report.budgets, report.storage, report.jobs) == ([], [], None)
    assert not report.ok

    records = [record for record in caplog.records if record.name == "hpcmon.cli"]
    assert [(record.levelno, record.getMessage()) for record in records] == [
        (logging.DEBUG, "broken: collection failed")
    ]
    assert records[0].exc_info[0] is RuntimeError


def test_collect_reports_invalid_section_of_real_cluster_as_internal_error():
    cluster = create_cluster(make_config("lumi"), runner=FakeRunner())
    report = cli._collect(cluster, ["bogus"], NOW)
    assert report.system == "LUMI"
    assert report.errors == ["internal error (ValueError: unknown section(s): bogus)"]
    assert report.failed_sections == ["bogus"]
    assert cluster.runner.calls == []


def test_collect_does_not_catch_keyboard_interrupt():
    cluster = StubCluster("a", error=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        cli._collect(cluster, ["jobs"], NOW)


def test_collect_returns_the_cluster_report_unchanged():
    runner = FakeRunner({"jobs": read_fixture("slurm/squeue.txt")})
    cluster = create_cluster(make_config("hpc", type="slurm"), runner=runner)
    report = cli._collect(cluster, ["jobs"], NOW)
    assert report.ok
    assert report.failed_sections == []
    assert report.collected_at == NOW
    assert [job.job_id for job in report.jobs] == ["1234567", "1234570", "1234568", "1234569_[1-10]", "1234571"]
    assert runner.calls == [{"jobs": SQUEUE_COMMAND}]


# -- show_config -----------------------------------------------------------------


def test_show_config_function_prints_source_rows_and_types():
    config = Config(
        clusters={"x": make_config("x", type="slurm", ssh=["ssh", "-p", "22", "x"], timeout=1.5, projects=["p1"])},
        path=Path("/etc/hpcmon.toml"),
    )
    buffer = io.StringIO()
    cli.show_config(config, Console(file=buffer, width=200, highlight=False))
    out = buffer.getvalue()
    assert out.splitlines()[0] == "Configuration: /etc/hpcmon.toml"
    assert table_rows(out) == [["x", "slurm", "ssh -p 22 x", "yes", "1.5 s", "p1"]]
    assert out.splitlines()[-1] == "Supported types: lumi, jupiter, leonardo, meluxina, slurm"


def test_show_config_without_file_lists_builtin_defaults(capsys):
    rc, out, err = run_cli(capsys, "--show-config")
    assert (rc, err) == (cli.EXIT_OK, "")
    lines = out.splitlines()
    assert lines[0] == f"Configuration: built-in defaults (no file at {default_location()})"
    assert table_rows(out) == [[name, name, f"ssh {name}", "yes", "180 s", "all"] for name in DEFAULT_NAMES]
    assert lines[-1] == "Supported types: lumi, jupiter, leonardo, meluxina, slurm"


def test_show_config_with_file(capsys, tmp_path):
    path = write_config(
        tmp_path,
        """
[settings]
timeout = 60

[clusters.lumi]
ssh = ["ssh", "-p", "2222", "lumi.example.org"]
projects = ["project_465000123", "project_465000999"]

[clusters.hpc]
type = "slurm"
ssh = "ssh hpc"
enabled = false
timeout = 2.5

[clusters.none]
type = "jupiter"
projects = []
""",
    )
    rc, out, err = run_cli(capsys, "-c", str(path), "--show-config")
    assert (rc, err) == (cli.EXIT_OK, "")
    assert out.splitlines()[0] == f"Configuration: {path}"
    assert table_rows(out) == [
        ["lumi", "lumi", "ssh -p 2222 lumi.example.org", "yes", "60 s", "project_465000123, project_465000999"],
        ["hpc", "slurm", "ssh hpc", "no", "2.5 s", "all"],
        ["none", "jupiter", "ssh none", "yes", "60 s", ""],
    ]


def test_show_config_uses_the_default_config_file(capsys):
    path = default_location()
    path.parent.mkdir(parents=True)
    path.write_text('[clusters.lumi]\nssh = "ssh lumi-alias"\n', encoding="utf-8")
    rc, out, _ = run_cli(capsys, "--show-config")
    assert rc == cli.EXIT_OK
    assert out.splitlines()[0] == f"Configuration: {path}"
    assert table_rows(out) == [["lumi", "lumi", "ssh lumi-alias", "yes", "180 s", "all"]]


def test_show_config_ignores_query_checks(capsys, tmp_path):
    path = write_config(tmp_path, "[clusters.lumi]\nenabled = false\n")
    rc, out, err = run_cli(capsys, "-c", str(path), "--show-config", "--timeout", "0")
    assert (rc, err) == (cli.EXIT_OK, "")
    assert table_rows(out) == [["lumi", "lumi", "ssh lumi", "no", "180 s", "all"]]


def test_show_config_with_unknown_cluster_is_a_usage_error(capsys):
    rc, out, err = run_cli(capsys, "--show-config", "nope")
    assert rc == cli.EXIT_USAGE
    assert out == ""
    assert err == "error: unknown cluster(s): nope (configured: lumi, jupiter, leonardo, meluxina)\n"


def test_show_config_with_empty_cluster_table(capsys, tmp_path):
    path = write_config(tmp_path, "[clusters]\n")
    rc, out, _ = run_cli(capsys, "-c", str(path), "--show-config")
    assert rc == cli.EXIT_OK
    assert table_rows(out) == []
    assert "Name" in out and "Supported types:" in out


# -- --init-config ---------------------------------------------------------------


def test_init_config_writes_to_given_path(capsys, tmp_path):
    path = tmp_path / "nested" / "dir" / "hpcmon.toml"
    rc, out, err = run_cli(capsys, "--init-config", "-c", str(path))
    assert (rc, err) == (cli.EXIT_OK, "")
    assert out == f"Configuration written to {path}\n"
    assert path.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEMPLATE
    # The written file is a valid configuration of the default clusters.
    assert list(load_config(path, required=True).clusters) == DEFAULT_NAMES


def test_init_config_defaults_to_hpcmon_config_path(capsys, monkeypatch, tmp_path):
    # The file named by $HPCMON_CONFIG is required to *read* the
    # configuration, but --init-config creates it.
    path = tmp_path / "from-env" / "hpcmon.toml"
    monkeypatch.setenv("HPCMON_CONFIG", str(path))
    rc, out, err = run_cli(capsys, "--init-config")
    assert (rc, err) == (cli.EXIT_OK, "")
    assert out == f"Configuration written to {path}\n"
    assert path.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEMPLATE
    assert not default_location().exists()


@pytest.mark.parametrize("hpcmon_config", [None, ""], ids=["unset", "empty"])
def test_init_config_uses_xdg_config_home_without_hpcmon_config(capsys, monkeypatch, tmp_path, hpcmon_config):
    if hpcmon_config is not None:
        monkeypatch.setenv("HPCMON_CONFIG", hpcmon_config)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    rc, out, _ = run_cli(capsys, "--init-config")
    expected = tmp_path / "xdg" / "hpcmon" / "config.toml"
    assert rc == cli.EXIT_OK
    assert out == f"Configuration written to {expected}\n"
    assert expected.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEMPLATE


def test_init_config_refuses_to_overwrite_existing_file(capsys, tmp_path):
    path = write_config(tmp_path, "# keep me\n")
    rc, out, err = run_cli(capsys, "--init-config", "-c", str(path))
    assert rc == cli.EXIT_USAGE
    assert out == ""
    assert err == f"error: {path} already exists (use --force to overwrite it)\n"
    assert path.read_text(encoding="utf-8") == "# keep me\n"


def test_init_config_force_overwrites_even_an_invalid_file(capsys, tmp_path):
    path = write_config(tmp_path, "this is [not valid toml\n")
    rc, out, err = run_cli(capsys, "--init-config", "--force", "-c", str(path))
    assert (rc, err) == (cli.EXIT_OK, "")
    assert out == f"Configuration written to {path}\n"
    assert path.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEMPLATE


def test_init_config_error_message_keeps_brackets_literal(capsys, tmp_path):
    path = tmp_path / "[bold]odd" / "config.toml"
    path.parent.mkdir()
    path.write_text("", encoding="utf-8")
    rc, _, err = run_cli(capsys, "--init-config", "-c", str(path))
    assert rc == cli.EXIT_USAGE
    assert err == f"error: {path} already exists (use --force to overwrite it)\n"


def test_init_config_reports_unwritable_location(capsys, tmp_path):
    blocker = write_config(tmp_path, "", name="not-a-dir")
    path = blocker / "config.toml"
    rc, out, err = run_cli(capsys, "--init-config", "-c", str(path))
    assert rc == cli.EXIT_USAGE
    assert out == ""
    assert err.startswith(f"error: cannot write {path}: ")


INIT_CONFIG_CONFLICT = "--init-config cannot be combined with clusters, sections, --show-config or --json"


@pytest.mark.parametrize(
    "extra",
    [
        ["lumi"],
        ["lumi", "--json"],
        ["--compute"],
        ["--storage"],
        ["--jobs"],
        ["--show-config"],
        ["--json"],
        ["--force", "jupiter"],
    ],
    ids=["cluster", "cluster+json", "compute", "storage", "jobs", "show-config", "json", "force+cluster"],
)
def test_init_config_with_query_options_is_a_parser_error(capsys, tmp_path, extra):
    # The autouse fixture makes create_cluster raise if it is ever called.
    path = tmp_path / "config.toml"
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--init-config", "-c", str(path), *extra])
    assert excinfo.value.code == cli.EXIT_USAGE
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("usage: hpcmon")
    assert err.endswith(f"hpcmon: error: {INIT_CONFIG_CONFLICT}\n")
    assert not path.exists()  # nothing is written


def test_init_config_conflict_is_detected_with_cluster_names_after_options(capsys, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--init-config", "--force", "lumi"])
    assert excinfo.value.code == cli.EXIT_USAGE
    assert INIT_CONFIG_CONFLICT in capsys.readouterr().err
    assert not default_location().exists()


@pytest.mark.parametrize(
    "argv",
    [["--force"], ["--force", "--show-config"], ["--force", "lumi", "--jobs"], ["--json", "--force"]],
    ids=["alone", "show-config", "query", "json"],
)
def test_force_without_init_config_is_a_parser_error(capsys, argv):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == cli.EXIT_USAGE
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("usage: hpcmon")
    assert err.endswith("hpcmon: error: --force can only be used with --init-config\n")


def test_parser_errors_happen_before_logging_and_consoles(capsys, console_spy, basic_config_calls):
    with pytest.raises(SystemExit):
        cli.main(["--debug", "--force"])
    with pytest.raises(SystemExit):
        cli.main(["--debug", "--init-config", "--json"])
    assert console_spy == []
    assert basic_config_calls == []


# -- configuration and usage errors ---------------------------------------------


def test_invalid_toml_is_a_usage_error(capsys, tmp_path):
    path = write_config(tmp_path, "timeout = [1, \n")
    rc, out, err = run_cli(capsys, "-c", str(path))
    assert rc == cli.EXIT_USAGE
    assert out == ""
    assert err.startswith(f"error: {path}: invalid TOML: ")


def test_invalid_default_config_file_is_a_usage_error(capsys):
    path = default_location()
    path.parent.mkdir(parents=True)
    path.write_text("[clusters.x]\ntype = 'cray'\n", encoding="utf-8")
    rc, _, err = run_cli(capsys, "--show-config")
    assert rc == cli.EXIT_USAGE
    assert err == (
        f"error: {path}: [clusters.x]: unknown type 'cray' "
        "(one of: lumi, jupiter, leonardo, meluxina, slurm)\n"
    )


def test_configuration_error_keeps_markup_like_text_literal(capsys, tmp_path):
    path = write_config(tmp_path, "[settings]\nbogus = 1\n")
    rc, _, err = run_cli(capsys, "-c", str(path))
    assert rc == cli.EXIT_USAGE
    assert err == f"error: {path}: [settings]: unknown key(s): bogus\n"


def test_missing_explicit_config_file_is_a_usage_error(capsys, tmp_path):
    path = tmp_path / "missing.toml"
    rc, out, err = run_cli(capsys, "-c", str(path))
    assert rc == cli.EXIT_USAGE
    assert out == ""
    assert err == f"error: configuration file not found: {path}\n"


@pytest.mark.parametrize("argv", [[], ["--show-config"], ["--json", "lumi"]], ids=["query", "show-config", "json"])
def test_missing_file_named_by_hpcmon_config_is_a_usage_error(capsys, monkeypatch, tmp_path, factory, argv):
    path = tmp_path / "from-env.toml"
    monkeypatch.setenv("HPCMON_CONFIG", str(path))
    rc, out, err = run_cli(capsys, *argv)
    assert rc == cli.EXIT_USAGE
    assert out == ""
    assert err == f"error: configuration file not found: {path}\n"
    assert factory.configs == []


def test_file_named_by_hpcmon_config_is_used(capsys, monkeypatch, tmp_path):
    path = write_config(tmp_path, '[clusters.hpc]\ntype = "slurm"\n', name="from-env.toml")
    monkeypatch.setenv("HPCMON_CONFIG", str(path))
    rc, out, err = run_cli(capsys, "--show-config")
    assert (rc, err) == (cli.EXIT_OK, "")
    assert out.splitlines()[0] == f"Configuration: {path}"
    assert table_rows(out) == [["hpc", "slurm", "ssh hpc", "yes", "180 s", "all"]]


def test_config_option_takes_precedence_over_hpcmon_config(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HPCMON_CONFIG", str(tmp_path / "missing.toml"))
    path = write_config(tmp_path, '[clusters.hpc]\ntype = "slurm"\n')
    rc, out, err = run_cli(capsys, "-c", str(path), "--show-config")
    assert (rc, err) == (cli.EXIT_OK, "")
    assert out.splitlines()[0] == f"Configuration: {path}"


def test_empty_hpcmon_config_keeps_the_configuration_optional(capsys, monkeypatch):
    monkeypatch.setenv("HPCMON_CONFIG", "")
    rc, out, err = run_cli(capsys, "--show-config")
    assert (rc, err) == (cli.EXIT_OK, "")
    assert out.splitlines()[0] == f"Configuration: built-in defaults (no file at {default_location()})"


def test_load_config_is_required_only_when_a_file_is_named(capsys, monkeypatch, tmp_path):
    calls = []

    def fake_load_config(path=None, *, required=False):
        calls.append((path, required))
        raise ConfigError("stop here")

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    for argv in (["--show-config"], ["-c", "x.toml"]):
        assert run_cli(capsys, *argv)[0] == cli.EXIT_USAGE
    monkeypatch.setenv("HPCMON_CONFIG", "")
    assert run_cli(capsys, "lumi")[0] == cli.EXIT_USAGE
    monkeypatch.setenv("HPCMON_CONFIG", str(tmp_path / "env.toml"))
    assert run_cli(capsys, "lumi")[0] == cli.EXIT_USAGE
    assert calls == [(None, False), (Path("x.toml"), True), (None, False), (None, True)]


def test_unknown_cluster_lists_configured_clusters(capsys):
    rc, out, err = run_cli(capsys, "lumi", "nope", "other")
    assert rc == cli.EXIT_USAGE
    assert out == ""
    assert err == "error: unknown cluster(s): nope, other (configured: lumi, jupiter, leonardo, meluxina)\n"


@pytest.mark.parametrize(
    "text",
    [
        "[clusters.lumi]\nenabled = false\n[clusters.jupiter]\nenabled = false\n",
        "[clusters]\n",
    ],
    ids=["all-disabled", "no-clusters"],
)
def test_no_enabled_cluster_is_a_usage_error(capsys, tmp_path, text):
    path = write_config(tmp_path, text)
    rc, out, err = run_cli(capsys, "-c", str(path))
    assert rc == cli.EXIT_USAGE
    assert out == ""
    assert err == "error: no cluster is enabled in the configuration\n"


@pytest.mark.parametrize(
    "argv",
    [["-t", "0"], ["--timeout", "0"], ["-t", "-1"], ["--timeout=-0.5"], ["-t", "-0"]],
)
def test_non_positive_timeout_is_a_usage_error(capsys, factory, argv):
    rc, out, err = run_cli(capsys, *argv, "lumi")
    assert rc == cli.EXIT_USAGE
    assert out == ""
    assert err == "error: the timeout must be positive\n"
    assert factory.configs == []


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "1e999"])
def test_non_finite_timeout_is_a_usage_error(capsys, factory, value):
    rc, out, err = run_cli(capsys, f"--timeout={value}", "--jobs", "lumi")
    assert rc == cli.EXIT_USAGE
    assert out == ""
    assert err == "error: the timeout must be positive\n"
    assert factory.configs == []


@pytest.mark.parametrize("value", ["0.001", "1e-3", "86400"])
def test_small_or_large_finite_timeouts_are_accepted(capsys, factory, value):
    rc, _, err = run_cli(capsys, "-t", value, "--jobs", "lumi")
    assert (rc, err) == (cli.EXIT_OK, "")
    assert [config.timeout for config in factory.configs] == [float(value)]


# -- cluster configuration passed to create_cluster --------------------------------


def test_timeout_option_overrides_cluster_timeouts(capsys, tmp_path, monkeypatch):
    path = write_config(
        tmp_path,
        "[settings]\ntimeout = 75\n[clusters.lumi]\ntimeout = 10\n[clusters.hpc]\ntype = 'slurm'\nssh = 'ssh -J gw hpc'\n",
    )
    original = load_config(path, required=True)
    seen = []

    def create(config, runner=None):
        cluster = create_cluster(config)  # the real factory, with an SSHRunner
        seen.append((config, cluster.runner.timeout, cluster.runner.ssh_command))
        cluster.runner = FakeRunner({"jobs": ""})
        return cluster

    monkeypatch.setattr(cli, "create_cluster", create)
    rc, _, _ = run_cli(capsys, "-c", str(path), "--jobs", "--timeout", "42")

    assert rc == cli.EXIT_OK
    assert seen == [
        (dataclasses.replace(original.clusters["lumi"], timeout=42.0), 42.0, ["ssh", "lumi"]),
        (dataclasses.replace(original.clusters["hpc"], timeout=42.0), 42.0, ["ssh", "-J", "gw", "hpc"]),
    ]


def test_cluster_timeouts_come_from_configuration_without_option(capsys, tmp_path, factory):
    path = write_config(tmp_path, "[settings]\ntimeout = 75\n[clusters.lumi]\ntimeout = 10\n[clusters.jupiter]\n")
    rc, _, _ = run_cli(capsys, "-c", str(path), "--jobs")
    assert rc == cli.EXIT_OK
    assert [(config.name, config.timeout) for config in factory.configs] == [("lumi", 10.0), ("jupiter", 75.0)]
    assert factory.configs == list(load_config(path, required=True).clusters.values())


def test_default_configuration_queries_the_four_default_clusters(capsys, factory):
    rc, _, _ = run_cli(capsys, "--jobs", "--json")
    assert rc == cli.EXIT_OK
    assert [(c.name, c.type, c.ssh, c.timeout) for c in factory.configs] == [
        (name, name, ["ssh", name], 180.0) for name in DEFAULT_NAMES
    ]


def test_explicitly_named_disabled_cluster_is_queried(capsys, tmp_path, factory):
    path = write_config(tmp_path, "[clusters.lumi]\nenabled = false\n[clusters.jupiter]\n")

    rc, _, _ = run_cli(capsys, "-c", str(path), "--jobs")
    assert rc == cli.EXIT_OK
    assert [config.name for config in factory.configs] == ["jupiter"]

    factory.configs.clear()
    rc, out, _ = run_cli(capsys, "-c", str(path), "--jobs", "lumi")
    assert rc == cli.EXIT_OK
    assert [config.name for config in factory.configs] == ["lumi"]
    assert "LUMI" in out and "JUPITER" not in out


def test_duplicate_cluster_names_are_queried_once(capsys, factory):
    rc, out, _ = run_cli(capsys, "--json", "--jobs", "meluxina", "lumi", "meluxina", "lumi")
    assert rc == cli.EXIT_OK
    assert [config.name for config in factory.configs] == ["meluxina", "lumi"]
    assert [cluster["name"] for cluster in json.loads(out)["clusters"]] == ["meluxina", "lumi"]
    assert len(factory.runners["lumi"].calls) == 1


# -- sections ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags, expected_commands, expected_sections",
    [
        # The commands run in the order jobs -> compute -> storage.
        ([], {**LUMI_JOBS, **LUMI_COMPUTE, **LUMI_STORAGE}, ["compute", "storage", "jobs"]),
        (["--jobs"], LUMI_JOBS, ["jobs"]),
        (["--compute"], LUMI_COMPUTE, ["compute"]),
        (["--storage"], LUMI_STORAGE, ["storage"]),
        (["--jobs", "--compute"], {**LUMI_JOBS, **LUMI_COMPUTE}, ["compute", "jobs"]),
        (["--jobs", "--storage", "--compute"], {**LUMI_JOBS, **LUMI_COMPUTE, **LUMI_STORAGE}, SECTIONS),
        (["--storage", "--compute"], {**LUMI_COMPUTE, **LUMI_STORAGE}, ["compute", "storage"]),
    ],
    ids=["none", "jobs", "compute", "storage", "jobs+compute", "all-three", "storage+compute"],
)
def test_section_flags_select_remote_commands(capsys, factory, flags, expected_commands, expected_sections):
    rc, out, _ = run_cli(capsys, "--json", "lumi", *flags)
    assert rc == cli.EXIT_OK
    assert factory.runners["lumi"].calls == [expected_commands]
    assert list(factory.runners["lumi"].calls[0]) == list(expected_commands)
    (report,) = json.loads(out)["clusters"]
    assert report["sections"] == list(expected_sections)
    assert report["failed_sections"] == []


@pytest.mark.parametrize(
    "argv",
    [
        ["--jobs", "lumi"],
        ["lumi", "--jobs"],
        ["lumi", "--jobs", "leonardo"],
        ["--json", "lumi", "--jobs", "leonardo"],
        ["lumi", "-t", "30", "leonardo", "--jobs"],
    ],
    ids=["option-first", "option-last", "option-between", "json-between", "timeout-between"],
)
def test_cluster_names_may_follow_options(capsys, factory, argv):
    rc, _, err = run_cli(capsys, *argv)
    assert (rc, err) == (cli.EXIT_OK, "")
    expected = [name for name in ("lumi", "leonardo") if name in argv]
    assert [config.name for config in factory.configs] == expected
    assert all(runner.calls == [{"jobs": SQUEUE_COMMAND}] for runner in factory.runners.values())


def test_parse_intermixed_args_collects_every_cluster_name():
    parser = cli.build_parser()
    args = parser.parse_intermixed_args(["a", "--jobs", "b", "-c", "x.toml", "c", "--compute"])
    assert args.clusters == ["a", "b", "c"]
    assert (args.jobs, args.compute, args.storage, args.config) == (True, True, False, Path("x.toml"))
    # Plain parse_args would reject the names after the options.
    with pytest.raises(SystemExit):
        parser.parse_args(["a", "--jobs", "b"])


def test_generic_slurm_cluster_only_reports_jobs(capsys, tmp_path, factory):
    path = write_config(tmp_path, '[clusters.hpc]\ntype = "slurm"\n')
    rc, out, _ = run_cli(capsys, "-c", str(path), "--json")
    assert rc == cli.EXIT_OK
    (report,) = json.loads(out)["clusters"]
    assert report["sections"] == ["jobs"]
    assert factory.runners["hpc"].calls == [{"jobs": SQUEUE_COMMAND}]


def test_jobs_only_human_output_has_no_compute_or_storage(capsys, factory):
    rc, out, _ = run_cli(capsys, "--jobs", "lumi")
    assert rc == cli.EXIT_OK
    assert "Jobs: 2 running, 2 pending, 1 other" in out
    assert "Compute allocations" not in out
    assert "Storage" not in out


# -- output ------------------------------------------------------------------------


def test_json_output_of_all_default_clusters(capsys, factory):
    rc, out, err = run_cli(capsys, "--json")
    assert rc == cli.EXIT_OK
    assert err == ""
    assert out.endswith("}\n")
    document = json.loads(out)
    assert set(document) == {"generated_at", "clusters"}
    # The local time of the CLI, with its UTC offset.
    assert document["generated_at"] == "2026-11-18T20:30:00+01:00"

    clusters = {cluster["name"]: cluster for cluster in document["clusters"]}
    assert [cluster["name"] for cluster in document["clusters"]] == DEFAULT_NAMES
    for name, cluster in clusters.items():
        assert set(cluster) == {
            "name", "system", "gpus", "collected_at", "sections", "failed_sections", "budgets", "storage", "jobs",
            "errors", "warnings",
        }
        assert cluster["system"] == SYSTEMS[name]
        assert cluster["gpus"] == GPUS[name]
        assert cluster["collected_at"] == "2026-11-18T20:30:00+01:00"
        assert cluster["sections"] == ["compute", "storage", "jobs"]
        assert cluster["failed_sections"] == []
        assert (cluster["errors"], cluster["warnings"]) == ([], [])

    lumi = clusters["lumi"]
    # End = date of the last refresh of lumi-ldap-projectinfo + days left.
    assert [
        (b["resource"], b["unit"], b["total"], b["used"], b["end"], b["days_left"], b["account"], b["note"])
        for b in lumi["budgets"]
    ] == [
        ("CPU", "core-h", 50000.0, 0.0, "2027-04-04", 137, None, None),
        ("GPU", "GPU-h", 12000.0, 0.0, "2027-04-04", 137, None, None),
        ("Storage", "TB-h", 200000.0, 0.0, "2027-04-04", 137, None, "storage billing"),
    ]
    # Only the GPU budget can be spent on the GPU nodes.
    assert [(b["gpu_hours_per_unit"], b["gpu_hours_remaining"]) for b in lumi["budgets"]] == [
        (None, None), (1.0, 12000.0), (None, None),
    ]
    assert lumi["storage"][0]["name"] == "/users/jdoe1234"
    assert lumi["storage"][0]["used_bytes"] == 20 * 1024
    assert [job["job_id"] for job in lumi["jobs"]] == ["1234567", "1234570", "1234568", "1234569_[1-10]", "1234571"]
    assert lumi["jobs"][0]["name"] == "train|model [v2]"

    (jupiter_budget,) = clusters["jupiter"]["budgets"]
    assert (jupiter_budget["project"], jupiter_budget["account"], jupiter_budget["remaining"]) == (
        "e-dev-2026d00-001", None, 801261.12
    )
    assert (jupiter_budget["end"], jupiter_budget["days_left"]) == ("2027-05-31", 194)
    # 72 core-hours per GH200-hour.
    assert (jupiter_budget["gpu_hours_total"], jupiter_budget["gpu_hours_remaining"]) == pytest.approx(
        (12000.0, 801261.12 / 72)
    )
    assert clusters["jupiter"]["jobs"] == []

    (leonardo_budget,) = clusters["leonardo"]["budgets"]
    assert (
        leonardo_budget["month_total"], leonardo_budget["month_remaining"], leonardo_budget["month_remaining_fraction"],
        leonardo_budget["end"], leonardo_budget["days_left"],
    ) == (8250.0, 8250.0, 1.0, "2027-06-30", 224)
    # 8 core-hours per A100-hour.
    assert (leonardo_budget["gpu_hours_remaining"], leonardo_budget["month_gpu_hours_remaining"]) == (
        92813 / 8, 8250 / 8
    )
    assert {quota["note"] for quota in clusters["leonardo"]["storage"]} == {"as of 3 hours ago"}

    (meluxina_budget,) = clusters["meluxina"]["budgets"]
    assert (
        meluxina_budget["resource"], meluxina_budget["month_used"], meluxina_budget["month_total"],
        meluxina_budget["note"], meluxina_budget["days_left"],
    ) == ("GPU", 27.0, 210.0, "current month only", None)
    # 4 A100-hours per node-hour.
    assert (meluxina_budget["month_gpu_hours_total"], meluxina_budget["month_gpu_hours_remaining"]) == (840.0, 732.0)
    assert clusters["meluxina"]["storage"][0]["files_fraction"] == pytest.approx(4512 / 150000)


def test_json_output_reports_cluster_errors_with_exit_status_1(capsys, factory):
    factory.errors["jupiter"] = RemoteError("boom")
    rc, out, err = run_cli(capsys, "--json", "lumi", "jupiter", "--jobs")
    assert rc == cli.EXIT_CLUSTER_ERROR
    assert err == ""
    clusters = json.loads(out)["clusters"]
    assert [(c["name"], c["errors"]) for c in clusters] == [("lumi", []), ("jupiter", ["boom"])]
    assert [c["failed_sections"] for c in clusters] == [[], ["jobs"]]
    assert clusters[1]["jobs"] is None


def test_human_output_of_all_default_clusters(capsys, factory):
    rc, out, err = run_cli(capsys)
    assert rc == cli.EXIT_OK
    assert err == ""
    positions = [out.index(f"{SYSTEMS[name]} · GPU nodes: {GPUS[name]} ─") for name in DEFAULT_NAMES]
    assert positions == sorted(positions)
    # The GPU-hours summary of all the clusters comes last.
    assert out.index("All clusters ─") > positions[-1]
    assert out.index("GPU-hours left (estimated)") > positions[-1]
    for expected in (
        "project_465000123   CPU [core-h]",
        "2027-04-04 (137d)",
        "storage billing",  # the note of one LUMI budget only: a Notes column
        "/users/jdoe1234",
        "Jobs: 2 running, 2 pending, 1 other",
        "project_465000999",  # LUMI jobs use two accounts: Account column
        "Time / limit",
        "1:02:03 / 2-00:00:00",
        "train|model [v2]",
        "nid[005000-005003]",
        "801,261 (93%)",
        "92,813 (97%)",
        "8,250 (100%)",
        "2027-06-30 (224d)",
        "/leonardo/home/userexternal/jdoe5678",
        "no quota",
        "Note: as of 3 hours ago",  # shared by all the Leonardo storage rows
        "Note: current month only",  # the only MeluXina budget
        "/project/home/p200001",
        "~11,129 GPU-h",  # JUPITER: 801,261 core-h / 72
        "~11,602 GPU-h",  # Leonardo: 92,813 core-h / 8
        "~1,031 GPU-h",  # Leonardo this month: 8,250 core-h / 8
        "~732 GPU-h",  # MeluXina this month: 105 node-h * 4
        "11,129 (93%)",
        "11,602 (97%)",
        "1,031 (100%)",
        "732 (87%)",
    ):
        assert expected in out
    assert out.count("Notes") == 1  # LUMI compute only
    assert out.count("No jobs in the queue.") == 3
    assert "error:" not in out and "warning:" not in out


def test_human_output_title_shows_configured_name_when_it_differs(capsys, tmp_path, factory):
    path = write_config(tmp_path, '[clusters.lumi-test]\ntype = "lumi"\n')
    rc, out, _ = run_cli(capsys, "-c", str(path), "--compute")
    assert rc == cli.EXIT_OK
    assert out.startswith(f"LUMI (lumi-test) · GPU nodes: {GPUS['lumi']} ─")


def test_remote_error_is_shown_with_exit_status_1(capsys, factory):
    factory.errors["lumi"] = RemoteError("boom")
    rc, out, err = run_cli(capsys, "lumi")
    assert rc == cli.EXIT_CLUSTER_ERROR
    assert err == ""
    assert out.startswith(f"LUMI · GPU nodes: {GPUS['lumi']} ─")
    assert "error: boom" in out
    assert "Compute allocations" not in out
    # Failed sections are not reported as empty.
    assert "No compute allocations found." not in out
    assert "No storage quotas found." not in out
    assert "No jobs in the queue." not in out


def test_section_error_gives_exit_status_1(capsys, factory):
    results = fixture_results("lumi")
    del results["allocations"]
    factory.results["lumi"] = results
    rc, out, _ = run_cli(capsys, "lumi")
    assert rc == cli.EXIT_CLUSTER_ERROR
    assert "error: compute: no output from `lumi-allocations`" in out
    assert "No compute allocations found." not in out
    assert "/users/jdoe1234" in out  # the other sections are still shown

    rc, out, _ = run_cli(capsys, "lumi", "--json")
    (report,) = json.loads(out)["clusters"]
    assert (report["failed_sections"], report["budgets"]) == (["compute"], [])
    assert len(report["storage"]) == 4


def test_warnings_alone_keep_exit_status_0(capsys, factory):
    results = fixture_results("lumi")
    results["projectinfo"] = failed(1, stderr="no such project\n")
    factory.results["lumi"] = results
    rc, out, _ = run_cli(capsys, "lumi", "--compute")
    assert rc == cli.EXIT_OK
    assert "warning: end of projects unknown: `lumi-ldap-projectinfo` failed (exit status 1): no such project" in out
    assert "2027-04-04" not in out


def test_exit_status_1_when_any_of_several_clusters_fails(capsys, factory):
    factory.results["meluxina"] = {"myquota": ok(read_fixture("meluxina/myquota_rate_limited.txt")), "jobs": ""}
    rc, out, _ = run_cli(capsys)
    assert rc == cli.EXIT_CLUSTER_ERROR
    assert "error: compute: myquota refused to run" in out
    assert out.count("error:") == 2  # compute and storage of MeluXina only


def test_frozen_clock_drives_dates_of_the_output(capsys, factory, clock):
    clock["now"] = dt.datetime(2026, 12, 1, 9, 15, 30, tzinfo=CET)
    rc, out, _ = run_cli(capsys, "lumi", "leonardo", "--compute")
    assert rc == cli.EXIT_OK
    # LUMI: the end is the refresh date of the data (2026-11-18) + 137 days,
    # whatever the date of the query; the days left are counted from today.
    assert "2027-04-04 (124d)" in out
    assert "2027-06-30 (211d)" in out  # Leonardo: fixed end date

    rc, out, _ = run_cli(capsys, "lumi", "--compute", "--json")
    document = json.loads(out)
    assert document["generated_at"] == "2026-12-01T09:15:30+01:00"
    budget = document["clusters"][0]["budgets"][0]
    assert (budget["end"], budget["days_left"]) == ("2027-04-04", 124)


def test_today_is_the_local_date_of_the_clock(capsys, factory, clock):
    # Late in the evening in UTC-10 it is already the next day in UTC: the
    # days left are counted from the local date.
    clock["now"] = dt.datetime(2027, 6, 29, 23, 30, tzinfo=dt.timezone(dt.timedelta(hours=-10)))
    rc, out, _ = run_cli(capsys, "leonardo", "--compute")
    assert rc == cli.EXIT_OK
    assert "2027-06-30 (1d)" in out


@pytest.mark.parametrize(
    "now, expected",
    [
        (dt.datetime(2027, 6, 30, 8, 0, tzinfo=CET), "2027-06-30 (last day)"),
        (dt.datetime(2027, 7, 1, 8, 0, tzinfo=CET), "2027-06-30 (ended)"),
    ],
    ids=["last-day", "ended"],
)
def test_end_of_the_period_relative_to_the_clock(capsys, factory, clock, now, expected):
    clock["now"] = now
    rc, out, _ = run_cli(capsys, "leonardo", "--compute")
    assert rc == cli.EXIT_OK
    assert expected in out


def test_now_is_passed_to_every_cluster(capsys, stubs, clock):
    rc, _, _ = run_cli(capsys, "--json", "lumi", "jupiter", "--storage")
    assert rc == cli.EXIT_OK
    assert [(name, stub.calls[0]["now"], stub.calls[0]["sections"]) for name, stub in stubs.items()] == [
        ("lumi", NOW_LOCAL, ["storage"]),
        ("jupiter", NOW_LOCAL, ["storage"]),
    ]
    assert all(stub.calls[0]["now"].tzinfo is CET for stub in stubs.values())
    assert clock["astimezone"] == 1  # read once, in the local time zone
    assert NOW_LOCAL.date() == TODAY


def test_now_is_the_aware_local_time_of_the_real_clock(capsys, monkeypatch, stubs):
    monkeypatch.setattr(cli, "dt", dt)  # unfreeze the clock
    before = dt.datetime.now(dt.timezone.utc)
    rc, out, _ = run_cli(capsys, "--json", "lumi")
    after = dt.datetime.now(dt.timezone.utc)
    assert rc == cli.EXIT_OK
    now = stubs["lumi"].calls[0]["now"]
    assert now.tzinfo is not None
    assert now.utcoffset() == dt.datetime.now().astimezone().utcoffset()
    assert before <= now <= after
    generated_at = dt.datetime.fromisoformat(json.loads(out)["generated_at"])
    assert generated_at.utcoffset() == now.utcoffset()
    assert generated_at == now.replace(microsecond=0)


def test_projects_setting_hides_other_projects_and_warns(capsys, tmp_path, factory):
    path = write_config(tmp_path, '[clusters.lumi]\nprojects = ["project_465000999", "Project_465000123"]\n')
    rc, out, _ = run_cli(capsys, "-c", str(path), "--json")
    assert rc == cli.EXIT_OK
    (report,) = json.loads(out)["clusters"]
    assert len(report["budgets"]) == 3  # project names are case-insensitive
    assert [job["job_id"] for job in report["jobs"]] == ["1234567", "1234570", "1234568", "1234569_[1-10]", "1234571"]

    path = write_config(tmp_path, '[clusters.lumi]\nprojects = ["project_465000999"]\n')
    rc, out, _ = run_cli(capsys, "-c", str(path))
    assert rc == cli.EXIT_OK  # warnings only
    assert "No compute allocations found." in out
    assert "/users/jdoe1234" in out and "/project/project_465000123" not in out
    assert "1234569_[1-10]" in out and "1234567" not in out
    # Only one account is left among the jobs: no Account column.
    assert "Account" not in out
    # The name is only used by a job (jobs are not used to detect typos).
    assert "warning: 4 job(s) of other accounts hidden by the 'projects' setting" in out
    assert "warning: project(s) not found on this cluster: project_465000999" in out


def test_internal_error_of_a_cluster_is_reported(capsys, stubs):
    stubs["jupiter"] = StubCluster("jupiter", error=RuntimeError("kaput"))
    rc, out, err = run_cli(capsys, "lumi", "jupiter")
    assert rc == cli.EXIT_CLUSTER_ERROR
    assert err == ""
    assert "error: internal error (RuntimeError: kaput)" in out
    assert out.count("error:") == 1
    # The sections of the failed cluster are not reported as empty.
    assert out.count("No jobs in the queue.") == 0


# -- console options -----------------------------------------------------------------


@pytest.fixture
def console_spy(monkeypatch):
    """Record the consoles of the CLI and give them colours on non-terminals."""
    created: List[dict] = []

    class SpyConsole(Console):
        def __init__(self, *args, **kwargs):
            created.append(dict(kwargs))
            kwargs.setdefault("color_system", "standard")
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cli, "Console", SpyConsole)
    return created


@pytest.mark.parametrize("no_color", [False, True])
def test_console_options(capsys, console_spy, no_color):
    with pytest.raises(SystemExit):
        cli.main(["--version"] + (["--no-color"] if no_color else []))
    assert console_spy == []  # --version exits before creating the consoles

    run_cli(capsys, "--show-config", *(["--no-color"] if no_color else []))
    assert console_spy == [
        {"no_color": no_color, "highlight": False},
        {"stderr": True, "no_color": no_color, "highlight": False},
    ]


@pytest.mark.parametrize(
    "flags, title, error",
    [([], "\x1b[1;36mLUMI", "\x1b[1;31merror:"), (["--no-color"], "\x1b[1mLUMI", "\x1b[1merror:")],
    ids=["colour", "no-colour"],
)
def test_no_color_removes_colours_but_keeps_bold(capsys, console_spy, factory, flags, title, error):
    factory.errors["lumi"] = RemoteError("boom")
    rc, out, _ = run_cli(capsys, "lumi", *flags)
    assert rc == cli.EXIT_CLUSTER_ERROR
    assert out.startswith(title)
    assert ("\x1b[1;31merror: boom" in out) is not bool(flags)

    rc, _, err = run_cli(capsys, "nope", *flags)
    assert rc == cli.EXIT_USAGE
    assert err.startswith(error + "\x1b[0m unknown cluster(s): nope")


def test_debug_configures_logging(capsys, factory, basic_config_calls):
    rc, _, _ = run_cli(capsys, "--debug", "--jobs", "lumi")
    assert rc == cli.EXIT_OK
    assert basic_config_calls == [{"level": logging.DEBUG, "format": "%(levelname)s %(name)s: %(message)s"}]


def test_debug_is_configured_before_configuration_errors(capsys, basic_config_calls):
    rc, _, _ = run_cli(capsys, "--debug", "nope")
    assert rc == cli.EXIT_USAGE
    assert len(basic_config_calls) == 1


def test_logging_is_not_configured_without_debug(capsys, factory, basic_config_calls):
    rc, _, _ = run_cli(capsys, "--jobs", "lumi")
    assert rc == cli.EXIT_OK
    assert basic_config_calls == []


# -- sequential / parallel queries ------------------------------------------------------


@pytest.mark.parametrize(
    "settings, argv",
    [
        ("", ["--sequential", "lumi", "jupiter"]),
        ("[settings]\nparallel = false\n", ["lumi", "jupiter"]),
        ("", ["lumi"]),
    ],
    ids=["--sequential", "settings.parallel=false", "single-cluster"],
)
def test_sequential_queries_run_in_the_main_thread(capsys, tmp_path, stubs, settings, argv):
    path = write_config(tmp_path, settings + "[clusters.lumi]\n[clusters.jupiter]\n")
    rc, _, _ = run_cli(capsys, "-c", str(path), "--json", *argv)
    assert rc == cli.EXIT_OK
    assert [stub.calls[0]["thread"] for stub in stubs.values()] == [threading.get_ident()] * len(stubs)


def test_parallel_queries_run_in_worker_threads_and_keep_order(capsys, stubs):
    finish_log: List[str] = []
    stubs["jupiter"] = jupiter = StubCluster("jupiter", finish_log=finish_log)
    stubs["lumi"] = StubCluster("lumi", before=lambda: wait_for(jupiter.finished), finish_log=finish_log)

    rc, out, _ = run_cli(capsys, "--json", "lumi", "jupiter")

    assert rc == cli.EXIT_OK
    assert finish_log == ["jupiter", "lumi"]
    assert [(c["name"], c["errors"]) for c in json.loads(out)["clusters"]] == [("lumi", []), ("jupiter", [])]
    assert all(stub.calls[0]["thread"] != threading.get_ident() for stub in stubs.values())


def test_parallel_human_output_keeps_cluster_order(capsys, stubs):
    finish_log: List[str] = []
    stubs["jupiter"] = jupiter = StubCluster("jupiter", finish_log=finish_log)
    stubs["lumi"] = StubCluster("lumi", before=lambda: wait_for(jupiter.finished), finish_log=finish_log)

    rc, out, _ = run_cli(capsys, "lumi", "jupiter", "--jobs")

    assert rc == cli.EXIT_OK
    assert finish_log == ["jupiter", "lumi"]
    assert out.index("Stub system (lumi)") < out.index("Stub system (jupiter)")


# -- thread pool of the parallel queries ------------------------------------------------


@pytest.fixture
def pools(monkeypatch) -> List[ThreadPoolExecutor]:
    """Record the thread pools created by ``collect_reports`` and their shutdowns."""
    created: List[ThreadPoolExecutor] = []

    class RecordingPool(ThreadPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.init_args = (args, kwargs)
            self.shutdown_calls: List[tuple] = []
            created.append(self)

        def shutdown(self, *args, **kwargs):
            self.shutdown_calls.append((args, kwargs))
            super().shutdown(*args, **kwargs)

    monkeypatch.setattr(cli, "ThreadPoolExecutor", RecordingPool)
    return created


NO_WAIT_SHUTDOWN = [((), {"wait": False, "cancel_futures": True})]


def test_pool_has_one_worker_per_cluster_and_is_shut_down_when_exhausted(pools):
    clusters = [StubCluster(name) for name in ("a", "b", "c")]
    reports = cli.collect_reports(clusters, ["jobs"], parallel=True, now=NOW)
    assert pools == []  # created lazily, on the first report
    assert [report.name for report in reports] == ["a", "b", "c"]
    (pool,) = pools
    assert pool.init_args == ((), {"max_workers": 3})
    assert pool.shutdown_calls == NO_WAIT_SHUTDOWN


def test_pool_is_shut_down_without_waiting_when_the_generator_is_closed_early(pools):
    release = threading.Event()
    started = threading.Event()

    def block() -> None:
        started.set()
        wait_for(release)

    slow = StubCluster("slow", before=block)
    reports = cli.collect_reports([StubCluster("fast"), slow], ["jobs"], parallel=True, now=NOW)
    try:
        assert next(reports).name == "fast"
        # Only a running query outlives the shutdown: pending ones are cancelled.
        assert started.wait(WAIT)
        reports.close()  # e.g. the consumer stopped on Ctrl-C
        (pool,) = pools
        assert pool.shutdown_calls == NO_WAIT_SHUTDOWN
        assert not slow.finished.is_set()  # the shutdown did not wait for it
    finally:
        release.set()
    assert slow.finished.wait(WAIT)


def test_pool_is_shut_down_when_a_cluster_raises_keyboard_interrupt(pools):
    clusters = [StubCluster("a"), StubCluster("b", error=KeyboardInterrupt())]
    reports = cli.collect_reports(clusters, ["jobs"], parallel=True, now=NOW)
    assert next(reports).name == "a"
    with pytest.raises(KeyboardInterrupt):
        next(reports)
    (pool,) = pools
    assert pool.shutdown_calls == NO_WAIT_SHUTDOWN


@pytest.mark.parametrize("parallel, count", [(False, 2), (True, 1), (True, 0)])
def test_no_pool_without_parallelism(pools, parallel, count):
    clusters = [StubCluster(f"c{index}") for index in range(count)]
    assert len(list(cli.collect_reports(clusters, ["jobs"], parallel=parallel, now=NOW))) == count
    assert pools == []


# -- parallel mode and spinner of main --------------------------------------------------


@pytest.fixture
def collect_spy(monkeypatch) -> List[dict]:
    """Record the arguments of ``collect_reports`` as called by ``main``."""
    calls: List[dict] = []
    real = cli.collect_reports

    def spy(clusters, sections, parallel, now):
        calls.append({"names": [cluster.name for cluster in clusters], "sections": list(sections), "parallel": parallel})
        return real(clusters, sections, parallel=parallel, now=now)

    monkeypatch.setattr(cli, "collect_reports", spy)
    return calls


@pytest.fixture
def spinner_log(monkeypatch) -> List[dict]:
    """Record every ``Console.status`` spinner and when it is entered/exited."""
    calls: List[dict] = []

    class Spinner:
        def __init__(self, record: dict) -> None:
            self.record = record

        def __enter__(self):
            self.record["events"].append("enter")
            return self

        def __exit__(self, *exc_info):
            self.record["events"].append("exit")
            self.record["exc_type"] = exc_info[0]
            return False

    def status(self, status, *args, **kwargs):
        record = {"stderr": self.stderr, "text": str(status), "events": []}
        calls.append(record)
        return Spinner(record)

    monkeypatch.setattr(rich.console.Console, "status", status)
    return calls


TWO_CLUSTERS = "[clusters.lumi]\n[clusters.jupiter]\n"


@pytest.mark.parametrize(
    "settings, argv, parallel, names",
    [
        ("", ["lumi", "jupiter"], True, ["lumi", "jupiter"]),
        ("", [], True, ["lumi", "jupiter"]),
        ("", ["jupiter", "--jobs", "lumi"], True, ["jupiter", "lumi"]),
        ("", ["--sequential", "lumi", "jupiter"], False, ["lumi", "jupiter"]),
        ("[settings]\nparallel = false\n", ["lumi", "jupiter"], False, ["lumi", "jupiter"]),
        ("[settings]\nparallel = false\n", ["--sequential"], False, ["lumi", "jupiter"]),
        ("", ["lumi"], False, ["lumi"]),
        ("", ["lumi", "lumi"], False, ["lumi"]),
    ],
    ids=[
        "two-clusters", "all-enabled", "intermixed", "--sequential", "settings.parallel=false",
        "both", "single-cluster", "duplicate-names",
    ],
)
def test_parallel_only_with_several_clusters_and_parallel_settings(
    capsys, tmp_path, stubs, collect_spy, spinner_log, settings, argv, parallel, names
):
    path = write_config(tmp_path, settings + TWO_CLUSTERS)
    rc, out, err = run_cli(capsys, "-c", str(path), *argv)
    assert (rc, err) == (cli.EXIT_OK, "")
    assert [(call["names"], call["parallel"]) for call in collect_spy] == [(names, parallel)]
    if parallel:
        assert spinner_log == [
            {"stderr": False, "text": f"Querying {', '.join(names)}...", "events": ["enter", "exit"], "exc_type": None}
        ]
    else:
        # No spinner: it would hide the prompts of an interactive SSH login.
        assert spinner_log == []
    main_thread = threading.get_ident()
    assert all((stub.calls[0]["thread"] == main_thread) is not parallel for stub in stubs.values())


@pytest.mark.parametrize("argv, stderr", [([], False), (["--json"], True)], ids=["human", "json"])
def test_spinner_spans_the_collection_on_the_console_not_used_for_the_output(
    capsys, stubs, spinner_log, argv, stderr
):
    events: List[str] = []
    for name in ("lumi", "meluxina"):
        stubs[name] = StubCluster(name, before=lambda name=name: events.append(f"collect {name}"))

    rc, out, err = run_cli(capsys, *argv, "lumi", "meluxina")

    assert (rc, err) == (cli.EXIT_OK, "")
    (spinner,) = spinner_log
    assert spinner["stderr"] is stderr  # JSON goes to stdout: the spinner to stderr
    assert spinner["text"] == "Querying lumi, meluxina..."
    assert spinner["events"] == ["enter", "exit"]
    assert sorted(events) == ["collect lumi", "collect meluxina"]
    if argv:
        assert [cluster["name"] for cluster in json.loads(out)["clusters"]] == ["lumi", "meluxina"]


def test_spinner_is_entered_before_collection_and_exited_after_rendering(capsys, stubs, monkeypatch, spinner_log):
    events: List[str] = []
    for name in ("lumi", "jupiter"):
        stubs[name] = StubCluster(name, before=lambda name=name: events.append(f"collect {name}"))
    real_render = cli.render_reports

    def render(reports, console, today):
        events.append("render")
        return real_render(reports, console, today=today)

    monkeypatch.setattr(cli, "render_reports", render)
    original_status = rich.console.Console.status

    def status(self, *args, **kwargs):
        events.append("spinner created")
        manager = original_status(self, *args, **kwargs)

        class Tracking:
            def __enter__(self_inner):
                events.append("spinner entered")
                return manager.__enter__()

            def __exit__(self_inner, *exc_info):
                events.append("spinner exited")
                return manager.__exit__(*exc_info)

        return Tracking()

    monkeypatch.setattr(rich.console.Console, "status", status)
    rc, _, _ = run_cli(capsys, "lumi", "jupiter")
    assert rc == cli.EXIT_OK
    assert events[:3] == ["spinner created", "spinner entered", "render"]
    assert sorted(events[3:5]) == ["collect jupiter", "collect lumi"]
    assert events[5:] == ["spinner exited"]


# -- Ctrl-C ----------------------------------------------------------------------------


@pytest.fixture
def terminate_calls(monkeypatch) -> List[bool]:
    calls: List[bool] = []
    monkeypatch.setattr(cli, "terminate_all", lambda: calls.append(True))
    return calls


def run_cli_interruptible(capsys, *argv: str):
    """Like :func:`run_cli`, but a KeyboardInterrupt escaping ``main`` fails the test."""
    try:
        return run_cli(capsys, *argv)
    except KeyboardInterrupt:  # pragma: no cover - only when main is broken
        pytest.fail("KeyboardInterrupt escaped from main()")


@pytest.mark.parametrize("argv", [["lumi"], ["--json", "lumi"], ["--sequential", "--json", "lumi", "jupiter"]])
def test_interrupt_during_the_first_query(capsys, stubs, terminate_calls, spinner_log, argv):
    stubs["lumi"] = StubCluster("lumi", error=KeyboardInterrupt())
    rc, out, err = run_cli_interruptible(capsys, *argv)
    assert rc == cli.EXIT_INTERRUPTED == 130
    assert out == ""  # in particular, no partial JSON document
    assert err == "interrupted\n"
    assert terminate_calls == [True]
    assert "jupiter" not in stubs or stubs["jupiter"].calls == []  # not queried any more
    assert spinner_log == []


def test_interrupt_of_a_sequential_query_keeps_the_reports_already_printed(capsys, stubs, terminate_calls):
    stubs["jupiter"] = StubCluster("jupiter", error=KeyboardInterrupt())
    rc, out, err = run_cli_interruptible(capsys, "--sequential", "lumi", "jupiter", "meluxina")
    assert rc == cli.EXIT_INTERRUPTED
    assert out.startswith("Stub system (lumi)")
    assert "Stub system (jupiter)" not in out
    assert err == "interrupted\n"
    assert terminate_calls == [True]
    assert stubs["meluxina"].calls == []  # created, but never queried


@pytest.mark.parametrize("json_flag", [[], ["--json"]], ids=["human", "json"])
def test_interrupt_of_parallel_queries_stops_the_sessions(capsys, stubs, terminate_calls, spinner_log, json_flag):
    stubs["jupiter"] = StubCluster("jupiter", error=KeyboardInterrupt())
    rc, out, err = run_cli_interruptible(capsys, *json_flag, "lumi", "jupiter")
    assert rc == cli.EXIT_INTERRUPTED
    assert err == "interrupted\n"
    assert terminate_calls == [True]
    assert ("Stub system (lumi)" in out) is not bool(json_flag)
    if json_flag:
        assert out == ""
    (spinner,) = spinner_log
    assert spinner["events"] == ["enter", "exit"]
    assert spinner["exc_type"] is KeyboardInterrupt  # stopped before printing "interrupted"


def test_interrupt_while_rendering(capsys, stubs, monkeypatch, terminate_calls):
    def render(reports, console, today):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "render_reports", render)
    rc, out, err = run_cli_interruptible(capsys, "lumi")
    assert (rc, out, err) == (cli.EXIT_INTERRUPTED, "", "interrupted\n")
    assert terminate_calls == [True]


def test_interrupted_message_keeps_no_colour_option(capsys, console_spy, stubs, terminate_calls):
    stubs["lumi"] = StubCluster("lumi", error=KeyboardInterrupt())
    _, _, err = run_cli_interruptible(capsys, "lumi")
    assert err == "\x1b[1;31minterrupted\x1b[0m\n"
    _, _, err = run_cli_interruptible(capsys, "lumi", "--no-color")
    assert err == "\x1b[1minterrupted\x1b[0m\n"


def test_interrupt_stops_the_ssh_sessions_of_the_runner():
    import hpcmon.runner

    assert cli.terminate_all is hpcmon.runner.terminate_all


def test_terminate_all_is_not_called_without_interrupt(capsys, stubs, terminate_calls):
    rc, _, _ = run_cli(capsys, "lumi", "jupiter")
    assert rc == cli.EXIT_OK
    assert terminate_calls == []


@requires_termios
def test_interrupt_restores_the_terminal(capsys, monkeypatch, stubs, terminate_calls, fake_termios):
    monkeypatch.setattr(sys, "stdin", FakeStdin(tty=True, fd=11))
    stubs["lumi"] = StubCluster("lumi", error=KeyboardInterrupt())
    rc, _, _ = run_cli_interruptible(capsys, "lumi")
    assert rc == cli.EXIT_INTERRUPTED
    assert fake_termios["get"] == [11]
    assert fake_termios["set"] == [(11, termios.TCSADRAIN, ["saved", "attributes"])]


@requires_termios
def test_queries_run_with_the_terminal_preserved(capsys, monkeypatch, stubs, fake_termios):
    monkeypatch.setattr(sys, "stdin", FakeStdin(tty=True, fd=5))
    seen: List[tuple] = []
    stubs["lumi"] = StubCluster("lumi", before=lambda: seen.append((list(fake_termios["get"]), list(fake_termios["set"]))))
    rc, _, _ = run_cli(capsys, "--json", "lumi")
    assert rc == cli.EXIT_OK
    assert seen == [([5], [])]  # saved before the query, restored after it
    assert fake_termios["set"] == [(5, termios.TCSADRAIN, ["saved", "attributes"])]


# -- preserve_terminal --------------------------------------------------------------------


class FakeStdin:
    """Stand-in for ``sys.stdin``."""

    def __init__(self, tty: bool = True, fd: int = 7) -> None:
        self.tty = tty
        self.fd = fd
        self.fileno_calls = 0

    def isatty(self) -> bool:
        return self.tty

    def fileno(self) -> int:
        self.fileno_calls += 1
        return self.fd


@pytest.fixture
def fake_termios(monkeypatch) -> dict:
    """Replace ``termios.tcgetattr``/``tcsetattr``; set ``*_error`` to make them fail."""
    state: dict = {"get": [], "set": [], "get_error": None, "set_error": None}

    def tcgetattr(fd):
        state["get"].append(fd)
        if state["get_error"] is not None:
            raise state["get_error"]
        return ["saved", "attributes"]

    def tcsetattr(fd, when, attributes):
        state["set"].append((fd, when, attributes))
        if state["set_error"] is not None:
            raise state["set_error"]

    monkeypatch.setattr(termios, "tcgetattr", tcgetattr)
    monkeypatch.setattr(termios, "tcsetattr", tcsetattr)
    return state


@requires_termios
def test_preserve_terminal_saves_and_restores_the_attributes_of_a_tty(monkeypatch, fake_termios):
    monkeypatch.setattr(sys, "stdin", FakeStdin(tty=True, fd=7))
    with cli.preserve_terminal() as value:
        assert value is None
        assert (fake_termios["get"], fake_termios["set"]) == ([7], [])
    assert fake_termios["set"] == [(7, termios.TCSADRAIN, ["saved", "attributes"])]


@requires_termios
@pytest.mark.parametrize("error", [RuntimeError("boom"), KeyboardInterrupt()], ids=["error", "interrupt"])
def test_preserve_terminal_restores_when_the_body_raises(monkeypatch, fake_termios, error):
    monkeypatch.setattr(sys, "stdin", FakeStdin(tty=True, fd=3))
    with pytest.raises(type(error)):
        with cli.preserve_terminal():
            raise error
    assert fake_termios["set"] == [(3, termios.TCSADRAIN, ["saved", "attributes"])]


@requires_termios
@pytest.mark.parametrize(
    "error", [TERMIOS_ERROR(25, "Inappropriate ioctl for device"), OSError(9, "Bad file descriptor"), ValueError("closed file")],
    ids=["termios.error", "OSError", "ValueError"],
)
def test_preserve_terminal_ignores_errors_when_saving(monkeypatch, fake_termios, error):
    monkeypatch.setattr(sys, "stdin", FakeStdin(tty=True, fd=4))
    fake_termios["get_error"] = error
    with cli.preserve_terminal():
        pass
    assert fake_termios["get"] == [4]
    assert fake_termios["set"] == []  # nothing saved, nothing to restore


@requires_termios
@pytest.mark.parametrize(
    "error", [TERMIOS_ERROR(5, "Input/output error"), OSError(9, "Bad file descriptor"), ValueError("closed file")],
    ids=["termios.error", "OSError", "ValueError"],
)
def test_preserve_terminal_ignores_errors_when_restoring(monkeypatch, fake_termios, error):
    monkeypatch.setattr(sys, "stdin", FakeStdin(tty=True, fd=4))
    fake_termios["set_error"] = error
    with cli.preserve_terminal():
        pass
    assert fake_termios["set"] == [(4, termios.TCSADRAIN, ["saved", "attributes"])]


@requires_termios
def test_preserve_terminal_restore_error_does_not_hide_the_exception_of_the_body(monkeypatch, fake_termios):
    monkeypatch.setattr(sys, "stdin", FakeStdin(tty=True, fd=4))
    fake_termios["set_error"] = TERMIOS_ERROR(5, "Input/output error")
    with pytest.raises(KeyboardInterrupt):
        with cli.preserve_terminal():
            raise KeyboardInterrupt


@requires_termios
def test_preserve_terminal_does_nothing_when_stdin_is_not_a_tty(monkeypatch, fake_termios):
    stdin = FakeStdin(tty=False)
    monkeypatch.setattr(sys, "stdin", stdin)
    with cli.preserve_terminal():
        pass
    assert (fake_termios["get"], fake_termios["set"]) == ([], [])
    assert stdin.fileno_calls == 0


@requires_termios
def test_preserve_terminal_does_nothing_without_stdin(monkeypatch, fake_termios):
    monkeypatch.setattr(sys, "stdin", None)
    with cli.preserve_terminal():
        pass
    assert (fake_termios["get"], fake_termios["set"]) == ([], [])


@requires_termios
def test_preserve_terminal_restores_a_real_pseudo_terminal(monkeypatch):
    try:
        master, slave = os.openpty()
    except OSError as exc:  # pragma: no cover - e.g. sandboxes without pseudo-terminals
        pytest.skip(f"no pseudo-terminal available: {exc}")
    try:
        monkeypatch.setattr(sys, "stdin", FakeStdin(tty=os.isatty(slave), fd=slave))
        before = termios.tcgetattr(slave)
        assert before[3] & termios.ECHO
        with cli.preserve_terminal():
            # As left by an SSH password prompt killed on timeout.
            attributes = termios.tcgetattr(slave)
            attributes[3] &= ~termios.ECHO
            termios.tcsetattr(slave, termios.TCSANOW, attributes)
            assert not termios.tcgetattr(slave)[3] & termios.ECHO
        assert termios.tcgetattr(slave) == before
    finally:
        os.close(master)
        os.close(slave)
