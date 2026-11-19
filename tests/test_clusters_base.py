"""Tests of the cluster base class, the Slurm support and the cluster registry.

Covers :mod:`hpcmon.clusters.base` (through a minimal concrete cluster defined
here), :mod:`hpcmon.clusters.slurm` and :mod:`hpcmon.clusters` (``__init__``).
No remote host is ever contacted: commands are answered by :class:`FakeRunner`;
the only local processes are ``sh``/``bash`` shells checking the shape of the
remote command lines (with ``squeue`` and ``id`` stubbed by shell functions).
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import os
import shlex
import shutil
import subprocess
import types
from typing import Dict, List, Optional

import pytest

import hpcmon.clusters
import hpcmon.clusters.base as base
from hpcmon.clusters import CLUSTER_TYPES, create_cluster
from hpcmon.clusters.base import HOME_COMMAND, Cluster, CollectionError
from hpcmon.clusters.jupiter import JupiterCluster
from hpcmon.clusters.leonardo import LeonardoCluster
from hpcmon.clusters.lumi import LumiCluster
from hpcmon.clusters.meluxina import MeluxinaCluster
from hpcmon.clusters.slurm import (
    SQUEUE_COMMAND,
    SQUEUE_FORMAT,
    GenericSlurmCluster,
    SlurmCluster,
    parse_squeue,
)
from hpcmon.config import SECTIONS
from hpcmon.models import ComputeBudget, Job, StorageQuota
from hpcmon.runner import TIMEOUT_EXIT_CODE, CommandResult, RemoteError, SSHRunner
from hpcmon.units import parse_date
from tests.conftest import NOW, TODAY, FakeRunner, failed, make_config, ok, read_fixture

LOGGER = "hpcmon.clusters.base"

# ---------------------------------------------------------------------------
# A minimal concrete cluster
# ---------------------------------------------------------------------------

ALLOC_COMMAND = "show-allocations --all"
QUOTA_COMMAND = "show-quota"
JOBS_COMMAND = "list-jobs"

#: ``project resource used total end`` per line.
ALLOC_OUT = "proj_a GPU 120 1000 2027-03-31\nproj_b CPU 5000 20000 -\nproj_c GPU 0 10 -\n"
HOME_OUT = "/users/jdoe\n"
#: ``path project used_bytes`` per line.
QUOTA_OUT = "/project/proj_a proj_a 1024\n/project/proj_b proj_b 2048\n"
#: ``job_id,account,state`` per line (the account may be empty).
JOBS_OUT = "101,proj_a,RUNNING\n102,proj_b,PENDING\n103,,PENDING\n"

ALL_COMMANDS = {"alloc": ALLOC_COMMAND, "home": HOME_COMMAND, "quota": QUOTA_COMMAND, "jobs": JOBS_COMMAND}

BUDGET_A = ComputeBudget(project="proj_a", resource="GPU", unit="core-h", used=120.0, total=1000.0,
                         end=dt.date(2027, 3, 31))
BUDGET_B = ComputeBudget(project="proj_b", resource="CPU", unit="core-h", used=5000.0, total=20000.0)
BUDGET_C = ComputeBudget(project="proj_c", resource="GPU", unit="core-h", used=0.0, total=10.0)
HOME_QUOTA = StorageQuota(name="/users/jdoe", project=None)
QUOTA_A = StorageQuota(name="/project/proj_a", project="proj_a", used_bytes=1024)
QUOTA_B = StorageQuota(name="/project/proj_b", project="proj_b", used_bytes=2048)
JOB_101 = Job(job_id="101", account="proj_a", state="RUNNING")
JOB_102 = Job(job_id="102", account="proj_b", state="PENDING")
JOB_103 = Job(job_id="103", account="", state="PENDING")


class DummyCluster(Cluster):
    """Concrete cluster with trivial line-based formats.

    ``home`` is needed by both compute and storage, to exercise the
    de-duplication of commands.  Tests can make a parser warn (``to_warn``) or
    raise (``inject``) before it parses anything; every parser call is
    recorded in ``parsed`` as ``(section, today)``.
    """

    type_name = "dummy"
    display_name = "Dummy HPC"

    def __init__(self, config, runner=None) -> None:
        super().__init__(config, runner)
        self.parsed: List[tuple] = []
        self.to_warn: Dict[str, List[str]] = {}
        self.inject: Dict[str, BaseException] = {}

    def compute_commands(self) -> Dict[str, str]:
        return {"alloc": ALLOC_COMMAND, "home": HOME_COMMAND}

    def storage_commands(self) -> Dict[str, str]:
        return {"quota": QUOTA_COMMAND, "home": HOME_COMMAND}

    def jobs_commands(self) -> Dict[str, str]:
        return {"jobs": JOBS_COMMAND}

    def _enter(self, section: str, today: dt.date) -> None:
        self.parsed.append((section, today))
        for message in self.to_warn.get(section, []):
            self.warn(message)
        if section in self.inject:
            raise self.inject[section]

    def parse_compute(self, results, today):
        self._enter("compute", today)
        self.output(results, "home")
        budgets = []
        for line in self.output(results, "alloc").splitlines():
            project, resource, used, total, end = line.split()
            budgets.append(ComputeBudget(project=project, resource=resource, unit="core-h",
                                         used=float(used), total=float(total), end=parse_date(end)))
        return budgets

    def parse_storage(self, results, today):
        self._enter("storage", today)
        quotas = [StorageQuota(name=self.output(results, "home").strip())]
        for line in self.output(results, "quota").splitlines():
            path, project, used = line.split()
            quotas.append(StorageQuota(name=path, project=project, used_bytes=int(used)))
        return quotas

    def parse_jobs(self, results, today):
        self._enter("jobs", today)
        jobs = []
        for line in self.output(results, "jobs").splitlines():
            job_id, account, state = line.split(",")
            jobs.append(Job(job_id=job_id, account=account, state=state))
        return jobs


def dummy_runner(**overrides) -> FakeRunner:
    """A runner answering every command of :class:`DummyCluster`.

    Keyword arguments replace the result of a key; ``None`` removes it.
    """
    results = {"alloc": ALLOC_OUT, "home": HOME_OUT, "quota": QUOTA_OUT, "jobs": JOBS_OUT}
    results.update(overrides)
    return FakeRunner({key: value for key, value in results.items() if value is not None})


def dummy(runner: Optional[FakeRunner] = None, **config) -> DummyCluster:
    return DummyCluster(make_config("mycluster", type="dummy", **config), runner=runner or dummy_runner())


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [Cluster, SlurmCluster])
def test_abstract_cluster_classes_cannot_be_instantiated(cls):
    with pytest.raises(TypeError, match="abstract"):
        cls(make_config(), runner=FakeRunner())


@pytest.mark.parametrize("login_shell, remote", [(True, "bash -l -s"), (False, "bash -s")])
def test_default_runner_is_ssh_runner_built_from_config(login_shell, remote):
    ssh = ["ssh", "-o", "BatchMode=yes", "lumi"]
    cluster = DummyCluster(make_config("lumi", type="dummy", ssh=ssh, timeout=42.5, login_shell=login_shell))

    assert isinstance(cluster.runner, SSHRunner)
    assert cluster.runner.ssh_command == ssh
    assert cluster.runner.ssh_command is not ssh
    assert cluster.runner.timeout == 42.5
    assert cluster.runner.login_shell is login_shell
    assert cluster.runner.argv == ssh + [remote]


def test_default_runner_rejects_empty_ssh_command():
    with pytest.raises(ValueError, match="must not be empty"):
        DummyCluster(make_config("x", type="dummy", ssh=[]))


def test_given_runner_is_used_as_is():
    runner = FakeRunner()
    config = make_config("x", type="dummy")
    cluster = DummyCluster(config, runner=runner)
    assert cluster.runner is runner
    assert cluster.config is config


def test_name_is_the_configured_cluster_name_and_class_metadata():
    cluster = dummy()
    assert cluster.name == "mycluster"
    cluster.config.name = "renamed"
    assert cluster.name == "renamed"
    assert Cluster.type_name == "" and Cluster.display_name == ""


# ---------------------------------------------------------------------------
# commands()
# ---------------------------------------------------------------------------


def test_commands_default_to_union_of_all_sections_without_duplicates():
    commands = dummy().commands()
    assert commands == ALL_COMMANDS
    # Execution order: jobs, then compute, then storage ("home" only once,
    # where compute first needs it).
    assert list(commands) == ["jobs", "alloc", "home", "quota"]


def test_execution_order_constant():
    assert base.EXECUTION_ORDER == ("jobs", "compute", "storage")
    assert sorted(base.EXECUTION_ORDER) == sorted(SECTIONS)


@pytest.mark.parametrize(
    "sections, expected",
    [
        (["compute"], [("alloc", ALLOC_COMMAND), ("home", HOME_COMMAND)]),
        (["storage"], [("quota", QUOTA_COMMAND), ("home", HOME_COMMAND)]),
        (["jobs"], [("jobs", JOBS_COMMAND)]),
        (["storage", "jobs"], [("jobs", JOBS_COMMAND), ("quota", QUOTA_COMMAND), ("home", HOME_COMMAND)]),
        (["storage", "compute"], [("alloc", ALLOC_COMMAND), ("home", HOME_COMMAND), ("quota", QUOTA_COMMAND)]),
        (["storage", "compute", "jobs"],
         [("jobs", JOBS_COMMAND), ("alloc", ALLOC_COMMAND), ("home", HOME_COMMAND), ("quota", QUOTA_COMMAND)]),
        (("storage", "storage"), [("quota", QUOTA_COMMAND), ("home", HOME_COMMAND)]),
        ([], []),
    ],
)
def test_commands_for_a_subset_of_sections_follow_execution_order(sections, expected):
    commands = dummy().commands(sections)
    assert list(commands.items()) == expected


class SharedKeyCluster(DummyCluster):
    """Every section needs ``home``, with a different command each time."""

    def compute_commands(self) -> Dict[str, str]:
        return {"alloc": ALLOC_COMMAND, "home": "compute-home"}

    def storage_commands(self) -> Dict[str, str]:
        return {"home": "storage-home", "quota": QUOTA_COMMAND}

    def jobs_commands(self) -> Dict[str, str]:
        return {"jobs": JOBS_COMMAND, "home": "jobs-home"}


@pytest.mark.parametrize(
    "sections, expected",
    [
        (SECTIONS, [("jobs", JOBS_COMMAND), ("home", "jobs-home"), ("alloc", ALLOC_COMMAND), ("quota", QUOTA_COMMAND)]),
        (["storage", "compute"], [("alloc", ALLOC_COMMAND), ("home", "compute-home"), ("quota", QUOTA_COMMAND)]),
        (["storage"], [("home", "storage-home"), ("quota", QUOTA_COMMAND)]),
    ],
)
def test_commands_shared_by_sections_are_kept_once_from_the_first_section_to_run(sections, expected):
    cluster = SharedKeyCluster(make_config("c", type="dummy"), runner=FakeRunner())
    assert list(cluster.commands(sections).items()) == expected


def test_home_command_prints_home_in_a_local_shell():
    out = subprocess.run(["sh", "-c", HOME_COMMAND], capture_output=True, text=True, check=True,
                         env={"HOME": "/home/some user", "PATH": os.environ.get("PATH", "")}).stdout
    assert out == "/home/some user\n"


# ---------------------------------------------------------------------------
# collect(): the happy path
# ---------------------------------------------------------------------------


def test_collect_runs_all_commands_in_one_call_and_parses_every_section():
    cluster = dummy()
    report = cluster.collect(today=TODAY, now=NOW)

    assert cluster.runner.calls == [ALL_COMMANDS]
    assert list(cluster.runner.calls[0]) == ["jobs", "alloc", "home", "quota"]
    assert report.name == "mycluster"
    assert report.system == "Dummy HPC"
    assert report.sections == list(SECTIONS)
    assert report.collected_at == NOW
    assert report.budgets == [BUDGET_A, BUDGET_B, BUDGET_C]
    assert report.storage == [HOME_QUOTA, QUOTA_A, QUOTA_B]
    assert report.jobs == [JOB_101, JOB_102, JOB_103]
    assert report.errors == []
    assert report.warnings == []
    assert report.failed_sections == []
    # Parsing (unlike execution) follows the order of SECTIONS.
    assert cluster.parsed == [("compute", TODAY), ("storage", TODAY), ("jobs", TODAY)]


def test_collect_rejects_unknown_sections_before_running_anything():
    cluster = dummy()
    with pytest.raises(ValueError) as excinfo:
        cluster.collect(["compute", "gpus", "jobs", "disk"], today=TODAY, now=NOW)
    assert str(excinfo.value) == "unknown section(s): gpus, disk"
    assert cluster.runner.calls == []
    assert cluster.parsed == []


@pytest.mark.parametrize(
    "sections, commands, parsed",
    [
        (["compute"], {"alloc": ALLOC_COMMAND, "home": HOME_COMMAND}, ["compute"]),
        (["storage"], {"quota": QUOTA_COMMAND, "home": HOME_COMMAND}, ["storage"]),
        (["jobs"], {"jobs": JOBS_COMMAND}, ["jobs"]),
        (["jobs", "compute"], {"jobs": JOBS_COMMAND, "alloc": ALLOC_COMMAND, "home": HOME_COMMAND},
         ["compute", "jobs"]),
        (["storage", "compute"], {"alloc": ALLOC_COMMAND, "home": HOME_COMMAND, "quota": QUOTA_COMMAND},
         ["compute", "storage"]),
        ([], {}, []),
    ],
)
def test_collect_subset_runs_only_needed_commands_and_parsers(sections, commands, parsed):
    cluster = dummy()
    report = cluster.collect(sections, today=TODAY, now=NOW)

    assert cluster.runner.calls == [commands]
    assert list(cluster.runner.calls[0]) == list(commands)
    assert [section for section, _ in cluster.parsed] == parsed
    # The report lists the sections in the order of SECTIONS.
    assert report.sections == parsed
    assert report.budgets == ([BUDGET_A, BUDGET_B, BUDGET_C] if "compute" in sections else [])
    assert report.storage == ([HOME_QUOTA, QUOTA_A, QUOTA_B] if "storage" in sections else [])
    assert report.jobs == ([JOB_101, JOB_102, JOB_103] if "jobs" in sections else None)
    assert report.errors == []
    assert report.failed_sections == []


def test_collect_accepts_a_tuple_of_sections_and_stores_a_list():
    report = dummy().collect(("storage",), today=TODAY, now=NOW)
    assert report.sections == ["storage"]


def test_collect_orders_sections_like_sections_and_drops_duplicates():
    cluster = dummy()
    report = cluster.collect(("jobs", "storage", "jobs", "compute"), today=TODAY, now=NOW)
    assert report.sections == ["compute", "storage", "jobs"]
    assert [section for section, _ in cluster.parsed] == ["compute", "storage", "jobs"]
    assert list(cluster.runner.calls[0]) == ["jobs", "alloc", "home", "quota"]


def test_supported_sections_default_to_all_sections():
    assert Cluster.supported_sections == SECTIONS
    assert DummyCluster.supported_sections == SECTIONS
    for cls in (LumiCluster, JupiterCluster, LeonardoCluster, MeluxinaCluster):
        assert cls.supported_sections == SECTIONS


class JobsAndStorageCluster(DummyCluster):
    """A cluster that cannot report compute budgets."""

    supported_sections = ("storage", "jobs")


@pytest.mark.parametrize(
    "requested, sections, commands",
    [
        (SECTIONS, ["storage", "jobs"], ["jobs", "quota", "home"]),
        (["compute"], [], []),
        (["compute", "storage"], ["storage"], ["quota", "home"]),
        (["jobs", "compute"], ["jobs"], ["jobs"]),
    ],
)
def test_collect_keeps_only_requested_and_supported_sections(requested, sections, commands):
    runner = dummy_runner()
    cluster = JobsAndStorageCluster(make_config("c", type="dummy"), runner=runner)
    report = cluster.collect(requested, today=TODAY, now=NOW)

    assert report.sections == sections
    assert runner.calls == [{key: ALL_COMMANDS[key] for key in commands}]
    assert list(runner.calls[0]) == commands
    assert [section for section, _ in cluster.parsed] == sections
    assert report.budgets == []
    assert report.storage == ([HOME_QUOTA, QUOTA_A, QUOTA_B] if "storage" in sections else [])
    assert report.jobs == ([JOB_101, JOB_102, JOB_103] if "jobs" in sections else None)
    assert report.errors == [] and report.warnings == [] and report.failed_sections == []


def test_unsupported_sections_are_not_checked_for_unknown_projects():
    # Compute was requested but is not supported, so only jobs are collected
    # and the configured projects cannot be checked.
    cluster = JobsAndStorageCluster(make_config("c", type="dummy", projects=["nowhere"]), runner=dummy_runner())
    report = cluster.collect(["compute", "jobs"], today=TODAY, now=NOW)
    assert report.sections == ["jobs"]
    assert report.warnings == ["2 job(s) of other accounts hidden by the 'projects' setting"]


def test_generic_slurm_cluster_supports_jobs_only():
    assert GenericSlurmCluster.supported_sections == ("jobs",)
    assert SlurmCluster.supported_sections == SECTIONS


def test_generic_slurm_cluster_without_supported_sections_never_starts_ssh():
    # Compute and storage are not supported, so nothing is collected and the
    # real SSHRunner returns immediately (the autouse guard fails the test if
    # ssh is run).
    cluster = GenericSlurmCluster(make_config("cluster", type="slurm"))
    report = cluster.collect(["compute", "storage"], today=TODAY, now=NOW)
    assert isinstance(cluster.runner, SSHRunner)
    assert report.sections == []
    assert report.budgets == [] and report.storage == [] and report.jobs is None
    assert report.errors == [] and report.warnings == [] and report.failed_sections == []


# ---------------------------------------------------------------------------
# collect(): dates
# ---------------------------------------------------------------------------


#: The local time zone of the frozen clock (independent of the machine's).
FROZEN_TZ = dt.timezone(dt.timedelta(hours=5, minutes=30), "TEST")
#: The aware local time returned by the frozen clock; its date is the 19th
#: although the naive clock reading is still on the 18th in UTC terms.
FROZEN_NOW = dt.datetime(2026, 11, 19, 0, 29, 30, tzinfo=FROZEN_TZ)


class _FrozenDateTime(dt.datetime):
    calls: List[tuple] = []

    @classmethod
    def now(cls, tz=None):
        cls.calls.append(("now", tz))
        return cls(2026, 11, 18, 18, 59, 30)

    def astimezone(self, tz=None):
        type(self).calls.append(("astimezone", tz))
        return FROZEN_NOW


@pytest.mark.parametrize(
    "today, now, expected_today, expected_now",
    [
        (None, None, FROZEN_NOW.date(), FROZEN_NOW),
        (None, NOW, NOW.date(), NOW),
        (dt.date(2026, 1, 31), None, dt.date(2026, 1, 31), FROZEN_NOW),
        (dt.date(2026, 1, 31), NOW, dt.date(2026, 1, 31), NOW),
    ],
)
def test_collect_defaults_now_to_the_aware_local_clock_and_today_to_now(
    monkeypatch, today, now, expected_today, expected_now
):
    monkeypatch.setattr(_FrozenDateTime, "calls", [])
    monkeypatch.setattr(base, "dt", types.SimpleNamespace(datetime=_FrozenDateTime, date=dt.date))
    cluster = dummy()
    report = cluster.collect(today=today, now=now)

    assert report.collected_at == expected_now
    assert report.collected_at.tzinfo is expected_now.tzinfo
    assert cluster.parsed == [(section, expected_today) for section in SECTIONS]
    # The clock is read only when no time is given, and converted to the
    # local time zone (astimezone() without argument).
    assert _FrozenDateTime.calls == ([("now", None), ("astimezone", None)] if now is None else [])


def test_collect_default_today_is_the_date_of_the_aware_now():
    stamp = dt.datetime(2026, 11, 19, 0, 15, tzinfo=dt.timezone(dt.timedelta(hours=2)))
    cluster = dummy()
    report = cluster.collect(now=stamp)
    assert report.collected_at == stamp
    assert cluster.parsed == [(section, dt.date(2026, 11, 19)) for section in SECTIONS]


def test_collect_without_clock_patch_uses_an_aware_timestamp():
    report = dummy().collect(["jobs"])
    assert report.collected_at is not None
    assert report.collected_at.tzinfo is not None
    assert report.collected_at.utcoffset() is not None


def test_collect_fills_missing_end_dates_from_configuration_only():
    config_end = {"PROJ_A": dt.date(2030, 1, 1), "proj_b": dt.date(2027, 6, 30)}
    report = dummy(project_end=config_end).collect(["compute"], today=TODAY, now=NOW)

    ends = {budget.project: budget.end for budget in report.budgets}
    # proj_a: reported by the cluster, so the configuration is ignored.
    # proj_b: not reported, taken from the configuration.
    # proj_c: neither reported nor configured.
    assert ends == {"proj_a": dt.date(2027, 3, 31), "proj_b": dt.date(2027, 6, 30), "proj_c": None}


# ---------------------------------------------------------------------------
# collect(): project filtering
# ---------------------------------------------------------------------------


def test_collect_filters_by_configured_projects_case_insensitively():
    report = dummy(projects=["PROJ_A"]).collect(today=TODAY, now=NOW)

    assert report.budgets == [BUDGET_A]
    # Personal storage (project None) is always kept.
    assert report.storage == [HOME_QUOTA, QUOTA_A]
    # Jobs without an account are always kept.
    assert report.jobs == [JOB_101, JOB_103]
    assert report.errors == []
    assert report.warnings == ["1 job(s) of other accounts hidden by the 'projects' setting"]


def test_collect_with_empty_project_list_keeps_only_personal_items():
    report = dummy(projects=[]).collect(today=TODAY, now=NOW)
    assert report.budgets == []
    assert report.storage == [HOME_QUOTA]
    assert report.jobs == [JOB_103]
    # Nothing was configured, so no project can be missing.
    assert report.warnings == ["2 job(s) of other accounts hidden by the 'projects' setting"]


@pytest.mark.parametrize(
    "projects, jobs, warnings",
    [
        (None, [JOB_101, JOB_102, JOB_103], []),
        (["proj_a", "proj_b"], [JOB_101, JOB_102, JOB_103], []),
        (["proj_b"], [JOB_102, JOB_103], ["1 job(s) of other accounts hidden by the 'projects' setting"]),
        (["proj_c"], [JOB_103], ["2 job(s) of other accounts hidden by the 'projects' setting"]),
    ],
)
def test_hidden_jobs_warning_counts_jobs_of_other_accounts(projects, jobs, warnings):
    report = dummy(projects=projects).collect(["jobs"], today=TODAY, now=NOW)
    assert report.jobs == jobs
    assert report.warnings == warnings


def test_no_hidden_jobs_warning_when_the_jobs_section_fails():
    cluster = dummy(projects=["proj_c"])
    cluster.inject["jobs"] = CollectionError("squeue missing")
    report = cluster.collect(["jobs"], today=TODAY, now=NOW)
    assert report.jobs is None
    assert report.warnings == []
    assert report.failed_sections == ["jobs"]


def test_filtered_out_budgets_do_not_receive_configured_end_dates():
    cluster = dummy(projects=["proj_c"], project_end={"proj_b": dt.date(2027, 6, 30), "proj_c": dt.date(2028, 2, 29)})
    report = cluster.collect(["compute"], today=TODAY, now=NOW)
    assert report.budgets == [ComputeBudget(project="proj_c", resource="GPU", unit="core-h", used=0.0, total=10.0,
                                            end=dt.date(2028, 2, 29))]


class PresetCluster(DummyCluster):
    """A cluster whose parsers return copies of preset budgets and quotas."""

    def __init__(self, config, budgets=(), quotas=()) -> None:
        super().__init__(config, runner=dummy_runner())
        self.budgets = list(budgets)
        self.quotas = list(quotas)

    def parse_compute(self, results, today):
        self._enter("compute", today)
        return [dataclasses.replace(budget) for budget in self.budgets]

    def parse_storage(self, results, today):
        self._enter("storage", today)
        return [dataclasses.replace(quota) for quota in self.quotas]


#: A JSC-style budget charged to an account named differently from the project.
BUDGET_ACC = ComputeBudget(project="proj_x", account="budget_acc", resource="CPU", unit="core-h", total=10.0, used=1.0)
#: A budget without a separate account.
BUDGET_NOACC = ComputeBudget(project="proj_y", resource="GPU", unit="GPU-h", total=5.0, used=1.0)
#: A budget whose account is spelled in upper case.
BUDGET_UPPER = ComputeBudget(project="proj_z", account="ACC_Z", resource="GPU", unit="GPU-h", total=8.0, used=2.0)
PRESET_BUDGETS = [BUDGET_ACC, BUDGET_NOACC, BUDGET_UPPER]
QUOTA_STORE = StorageQuota(name="/store/proj_s", project="proj_s", used_bytes=1)


def preset(projects=None, **config) -> PresetCluster:
    return PresetCluster(make_config("preset", type="dummy", projects=projects, **config),
                         budgets=PRESET_BUDGETS, quotas=[HOME_QUOTA, QUOTA_STORE])


@pytest.mark.parametrize(
    "projects, kept",
    [
        (None, [BUDGET_ACC, BUDGET_NOACC, BUDGET_UPPER]),
        # Kept through the account, case-insensitively.
        (["Budget_Acc"], [BUDGET_ACC]),
        (["acc_z"], [BUDGET_UPPER]),
        # Kept through the project.
        (["PROJ_X"], [BUDGET_ACC]),
        (["proj_y", "acc_z"], [BUDGET_NOACC, BUDGET_UPPER]),
        # Neither the project nor the account is configured.
        (["proj_s"], []),
        ([], []),
    ],
)
def test_budgets_are_kept_when_their_project_or_account_is_configured(projects, kept):
    report = preset(projects).collect(["compute"], today=TODAY, now=NOW)
    assert report.budgets == kept
    assert report.errors == []


def test_budget_kept_through_its_account_gets_the_end_date_of_its_project():
    cluster = preset(["budget_acc"], project_end={"proj_x": dt.date(2027, 4, 30), "budget_acc": dt.date(2030, 1, 1)})
    report = cluster.collect(["compute"], today=TODAY, now=NOW)
    assert [(budget.project, budget.end) for budget in report.budgets] == [("proj_x", dt.date(2027, 4, 30))]


def test_budget_end_dates_from_the_cluster_are_not_overridden():
    ending = dataclasses.replace(BUDGET_NOACC, end=dt.date(2027, 1, 31))
    cluster = PresetCluster(make_config("p", type="dummy", project_end={"proj_y": dt.date(2030, 1, 1)}),
                            budgets=[ending])
    report = cluster.collect(["compute"], today=TODAY, now=NOW)
    assert report.budgets == [ending]


# ---------------------------------------------------------------------------
# collect(): configured projects that do not exist
# ---------------------------------------------------------------------------


def test_unknown_configured_projects_are_reported_in_configured_order_and_spelling():
    report = dummy(projects=["Typo_1", "proj_a", "PROJ_C", "other"]).collect(today=TODAY, now=NOW)
    assert report.budgets == [BUDGET_A, BUDGET_C]
    assert report.warnings == [
        "1 job(s) of other accounts hidden by the 'projects' setting",
        "project(s) not found on this cluster: Typo_1, other",
    ]
    assert report.errors == []


@pytest.mark.parametrize(
    "projects, sections, warnings",
    [
        # Found through the account or the project of a budget (any case).
        (["BUDGET_ACC", "proj_y", "Acc_Z", "PROJ_Z"], ["compute"], []),
        (["proj_x", "nowhere"], ["compute"], ["project(s) not found on this cluster: nowhere"]),
        # Found through the project of a storage area.
        (["PROJ_S"], ["storage"], []),
        (["proj_s"], ["compute", "storage"], []),
        # Storage projects do not count when storage is not collected, nor
        # budget names when compute is not.
        (["proj_s"], ["compute"], ["project(s) not found on this cluster: proj_s"]),
        (["budget_acc"], ["storage"], ["project(s) not found on this cluster: budget_acc"]),
        (["a", "proj_x", "b"], ["compute", "storage"], ["project(s) not found on this cluster: a, b"]),
    ],
)
def test_unknown_projects_are_looked_up_in_budgets_and_storage(projects, sections, warnings):
    report = preset(projects).collect(sections, today=TODAY, now=NOW)
    assert report.warnings == warnings
    assert report.errors == []


def test_project_known_only_from_job_accounts_is_reported_as_not_found():
    # The queue may be empty, so job accounts are not used to detect typos.
    runner = dummy_runner(jobs=JOBS_OUT + "104,proj_jobs,RUNNING\n")
    report = dummy(runner, projects=["proj_jobs"]).collect(today=TODAY, now=NOW)

    assert report.jobs == [JOB_103, Job(job_id="104", account="proj_jobs", state="RUNNING")]
    assert report.warnings == [
        "2 job(s) of other accounts hidden by the 'projects' setting",
        "project(s) not found on this cluster: proj_jobs",
    ]


@pytest.mark.parametrize("projects", [None, ["proj_a"], ["nowhere"]])
def test_unknown_projects_are_not_checked_for_jobs_only(projects):
    report = dummy(projects=projects).collect(["jobs"], today=TODAY, now=NOW)
    assert not any(warning.startswith("project(s) not found") for warning in report.warnings)


def test_unknown_projects_are_not_checked_without_a_projects_setting():
    report = dummy(dummy_runner(alloc="", quota="")).collect(today=TODAY, now=NOW)
    assert report.budgets == [] and report.storage == [HOME_QUOTA]
    assert report.warnings == []


def test_no_unknown_project_warning_when_all_are_found():
    report = dummy(projects=["proj_a", "PROJ_B", "proj_c"]).collect(today=TODAY, now=NOW)
    assert report.warnings == []


@pytest.mark.parametrize("broken", SECTIONS)
def test_unknown_projects_are_not_checked_when_a_section_failed(broken):
    cluster = dummy(projects=["proj_a", "nowhere"])
    cluster.inject[broken] = CollectionError("broken")
    report = cluster.collect(today=TODAY, now=NOW)

    assert report.failed_sections == [broken]
    assert not any(warning.startswith("project(s) not found") for warning in report.warnings)


def test_unknown_projects_are_not_checked_when_the_session_failed():
    cluster = dummy(FakeRunner(error=RemoteError("down")), projects=["nowhere"])
    report = cluster.collect(today=TODAY, now=NOW)
    assert report.warnings == []
    assert report.errors == ["down"]


def test_generic_slurm_cluster_never_checks_unknown_projects():
    runner = FakeRunner({"jobs": read_fixture("slurm/squeue.txt")})
    cluster = GenericSlurmCluster(make_config("c", type="slurm", projects=["nowhere"]), runner=runner)
    report = cluster.collect(today=TODAY, now=NOW)
    assert report.sections == ["jobs"]
    assert report.jobs == []
    assert report.warnings == ["5 job(s) of other accounts hidden by the 'projects' setting"]


# ---------------------------------------------------------------------------
# collect(): errors and warnings
# ---------------------------------------------------------------------------


def test_remote_error_is_the_only_thing_reported():
    cluster = dummy(FakeRunner(error=RemoteError("SSH connection failed: Permission denied (publickey).")))
    report = cluster.collect(today=TODAY, now=NOW)

    assert report.to_dict() == {
        "name": "mycluster",
        "system": "Dummy HPC",
        "gpus": None,
        "collected_at": "2026-11-18T20:30:00",
        "sections": ["compute", "storage", "jobs"],
        "failed_sections": ["compute", "storage", "jobs"],
        "budgets": [],
        "storage": [],
        "jobs": None,
        "errors": ["SSH connection failed: Permission denied (publickey)."],
        "warnings": [],
    }
    assert cluster.runner.calls == [ALL_COMMANDS]
    assert cluster.parsed == []


@pytest.mark.parametrize(
    "requested, failed_sections",
    [
        (["jobs", "storage"], ["storage", "jobs"]),
        (["compute"], ["compute"]),
        ([], []),
    ],
)
def test_remote_error_fails_every_collected_section(requested, failed_sections):
    cluster = dummy(FakeRunner(error=RemoteError("Connection closed by remote host")))
    report = cluster.collect(requested, today=TODAY, now=NOW)
    assert report.sections == failed_sections
    assert report.failed_sections == failed_sections
    assert report.failed_sections is not report.sections
    assert report.errors == ["Connection closed by remote host"]


def test_remote_error_fails_only_the_supported_sections():
    cluster = JobsAndStorageCluster(make_config("c", type="dummy"), runner=FakeRunner(error=RemoteError("x")))
    report = cluster.collect(today=TODAY, now=NOW)
    assert report.failed_sections == ["storage", "jobs"]

    slurm = GenericSlurmCluster(make_config("s", type="slurm"), runner=FakeRunner(error=RemoteError("y")))
    report = slurm.collect(today=TODAY, now=NOW)
    assert report.sections == ["jobs"]
    assert report.failed_sections == ["jobs"]
    assert report.errors == ["y"]


def test_remote_error_from_default_ssh_runner_is_reported(monkeypatch):
    import hpcmon.runner

    def no_popen(argv, *args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", argv[0])

    monkeypatch.setattr(hpcmon.runner.subprocess, "Popen", no_popen)
    cluster = DummyCluster(make_config("c", type="dummy", ssh=["ssh", "host"], login_shell=False))
    report = cluster.collect(["jobs"], today=TODAY, now=NOW)

    argv = shlex.join(["ssh", "host", "bash -s"])
    assert report.errors == [f"cannot run {argv!r}: [Errno 2] No such file or directory: 'ssh'"]
    assert report.jobs is None
    assert report.failed_sections == ["jobs"]


def test_non_remote_errors_from_the_runner_propagate():
    cluster = dummy(FakeRunner(error=ValueError("invalid command key: 'a b'")))
    with pytest.raises(ValueError, match="invalid command key"):
        cluster.collect(today=TODAY, now=NOW)


def test_collection_errors_are_reported_per_section_in_section_order():
    runner = dummy_runner(
        alloc=None,
        quota=failed(2, stdout="partial\n", stderr="quota: warning\nquota: cannot reach server\n\n"),
        jobs=failed(None, stdout="101,proj_a,RUNNING\n"),
    )
    report = dummy(runner).collect(today=TODAY, now=NOW)

    assert report.errors == [
        "compute: no output from `show-allocations --all`",
        "storage: `show-quota` failed (exit status 2): quota: cannot reach server",
        "jobs: `list-jobs` did not complete: no exit status",
    ]
    assert report.budgets == []
    assert report.storage == []
    assert report.jobs is None
    assert report.warnings == []
    assert report.failed_sections == ["compute", "storage", "jobs"]


def test_reason_of_an_unfinished_command_is_reported():
    unfinished = CommandResult(stdout="101,proj_a,RUNNING\n", returncode=None,
                               reason="the SSH session timed out after 195 s")
    report = dummy(dummy_runner(jobs=unfinished)).collect(today=TODAY, now=NOW)
    assert report.errors == ["jobs: `list-jobs` did not complete: the SSH session timed out after 195 s"]
    assert report.failed_sections == ["jobs"]


@pytest.mark.parametrize("broken", SECTIONS)
def test_a_failing_section_does_not_affect_the_others(broken):
    cluster = dummy()
    cluster.inject[broken] = CollectionError("tool not installed")
    report = cluster.collect(today=TODAY, now=NOW)

    assert report.errors == [f"{broken}: tool not installed"]
    assert report.failed_sections == [broken]
    assert report.sections == list(SECTIONS)
    assert report.budgets == ([] if broken == "compute" else [BUDGET_A, BUDGET_B, BUDGET_C])
    assert report.storage == ([] if broken == "storage" else [HOME_QUOTA, QUOTA_A, QUOTA_B])
    assert report.jobs == (None if broken == "jobs" else [JOB_101, JOB_102, JOB_103])
    assert [section for section, _ in cluster.parsed] == list(SECTIONS)


@pytest.mark.parametrize(
    "exception, message",
    [
        (RuntimeError("boom"), "storage: unexpected output (RuntimeError: boom)"),
        (KeyError("total"), "storage: unexpected output (KeyError: 'total')"),
        (IndexError(), "storage: unexpected output (IndexError: )"),
    ],
)
def test_unexpected_parser_exception_is_reported_and_logged(caplog, exception, message):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    cluster = dummy()
    cluster.inject["storage"] = exception
    report = cluster.collect(today=TODAY, now=NOW)

    assert report.errors == [message]
    assert report.failed_sections == ["storage"]
    assert report.storage == []
    assert report.budgets == [BUDGET_A, BUDGET_B, BUDGET_C]
    assert report.jobs == [JOB_101, JOB_102, JOB_103]
    [record] = [r for r in caplog.records if r.getMessage() == "mycluster: cannot parse storage"]
    assert record.name == LOGGER
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None and record.exc_info[1] is exception


def test_unparsable_output_is_reported_as_unexpected_output():
    runner = dummy_runner(alloc="proj_a GPU lots 1000 -\n", jobs="101;proj_a;RUNNING\n")
    report = dummy(runner).collect(today=TODAY, now=NOW)
    assert report.errors == [
        "compute: unexpected output (ValueError: could not convert string to float: 'lots')",
        "jobs: unexpected output (ValueError: not enough values to unpack (expected 3, got 1))",
    ]
    assert report.failed_sections == ["compute", "jobs"]
    assert report.storage == [HOME_QUOTA, QUOTA_A, QUOTA_B]


@pytest.mark.parametrize("exception", [KeyboardInterrupt(), SystemExit(3)])
def test_base_exceptions_from_parsers_are_not_swallowed(exception):
    cluster = dummy()
    cluster.inject["compute"] = exception
    with pytest.raises(type(exception)):
        cluster.collect(today=TODAY, now=NOW)


def test_every_command_output_is_logged_at_debug_level(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    runner = dummy_runner(
        alloc=ok(ALLOC_OUT, "deprecated option\n"),
        quota=failed(None, stdout="", stderr=""),
        jobs=None,
    )
    dummy(runner).collect(today=TODAY, now=NOW)

    messages = [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno == logging.DEBUG]
    assert messages == [
        f"mycluster: `{ALLOC_COMMAND}` exited with 0\n--- stdout ---\n{ALLOC_OUT}\n--- stderr ---\ndeprecated option\n",
        f"mycluster: `{HOME_COMMAND}` exited with 0\n--- stdout ---\n{HOME_OUT}\n--- stderr ---\n",
        f"mycluster: `{QUOTA_COMMAND}` exited with None\n--- stdout ---\n\n--- stderr ---\n",
    ]


def test_nothing_is_logged_when_the_session_fails(caplog):
    caplog.set_level(logging.DEBUG, logger=LOGGER)
    dummy(FakeRunner(error=RemoteError("down"))).collect(today=TODAY, now=NOW)
    assert [r for r in caplog.records if r.name == LOGGER] == []


def test_warnings_are_propagated_in_section_order():
    cluster = dummy()
    cluster.to_warn = {"jobs": ["squeue is slow"], "compute": ["budget data is stale", "unknown unit"]}
    report = cluster.collect(today=TODAY, now=NOW)
    assert report.warnings == ["budget data is stale", "unknown unit", "squeue is slow"]
    assert report.errors == []


def test_warnings_are_kept_when_the_section_then_fails():
    cluster = dummy()
    cluster.to_warn = {"storage": ["home not found"]}
    cluster.inject["storage"] = CollectionError("no quota tool")
    report = cluster.collect(today=TODAY, now=NOW)
    assert report.warnings == ["home not found"]
    assert report.errors == ["storage: no quota tool"]
    assert report.failed_sections == ["storage"]


def test_failed_sections_are_reset_between_collections():
    cluster = dummy()
    cluster.inject["compute"] = CollectionError("down")
    assert cluster.collect(today=TODAY, now=NOW).failed_sections == ["compute"]
    cluster.inject = {}
    assert cluster.collect(today=TODAY, now=NOW).failed_sections == []


def test_warnings_are_reset_between_collections():
    cluster = dummy()
    cluster.to_warn = {"compute": ["first run only"]}
    first = cluster.collect(today=TODAY, now=NOW)
    cluster.to_warn = {}
    second = cluster.collect(today=TODAY, now=NOW)

    assert first.warnings == ["first run only"]
    assert second.warnings == []
    assert len(cluster.runner.calls) == 2


def test_warnings_are_reset_when_the_next_session_fails():
    cluster = dummy()
    cluster.to_warn = {"compute": ["first run only"]}
    assert cluster.collect(today=TODAY, now=NOW).warnings == ["first run only"]
    cluster.runner = FakeRunner(error=RemoteError("Connection timed out"))
    report = cluster.collect(today=TODAY, now=NOW)
    assert report.warnings == []
    assert report.errors == ["Connection timed out"]


# ---------------------------------------------------------------------------
# output()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "results, check, message",
    [
        ({}, True, "no output from `alloc`"),
        ({}, False, "no output from `alloc`"),
        ({"alloc": failed(None, stdout="half")}, True, "`alloc` did not complete: no exit status"),
        ({"alloc": failed(None, stdout="half")}, False, "`alloc` did not complete: no exit status"),
        ({"alloc": failed(TIMEOUT_EXIT_CODE, stderr="Terminated")}, True,
         "`alloc` timed out (time limit of 180 s reached)"),
        ({"alloc": failed(TIMEOUT_EXIT_CODE, stderr="Terminated")}, False,
         "`alloc` timed out (time limit of 180 s reached)"),
        ({"alloc": failed(1, stdout="out\n", stderr="first\n  last line  \n\n")}, True,
         "`alloc` failed (exit status 1): last line"),
        ({"alloc": failed(2, stdout="partial\nError: access denied\n\n")}, True,
         "`alloc` failed (exit status 2): Error: access denied"),
        ({"alloc": failed(3)}, True, "`alloc` failed (exit status 3)"),
        ({"alloc": failed(1, stderr="  \n")}, True, "`alloc` failed (exit status 1)"),
        ({"alloc": failed(-9, stderr="Killed")}, True, "`alloc` failed (exit status -9): Killed"),
        ({"alloc": failed(255, stdout="x")}, True, "`alloc` failed (exit status 255): x"),
    ],
    ids=[
        "missing", "missing-nocheck", "unfinished", "unfinished-nocheck", "timeout", "timeout-nocheck",
        "stderr-detail", "stdout-detail", "no-detail", "blank-stderr-no-stdout", "signal", "255",
    ],
)
def test_output_errors(results, check, message):
    cluster = dummy()
    with pytest.raises(CollectionError) as excinfo:
        cluster.output(results, "alloc", check=check)
    assert str(excinfo.value) == message


@pytest.mark.parametrize("check", [True, False])
@pytest.mark.parametrize(
    "reason, message",
    [
        ("the SSH session timed out after 195 s", "`alloc` did not complete: the SSH session timed out after 195 s"),
        ("the SSH session ended unexpectedly (Connection reset by peer)",
         "`alloc` did not complete: the SSH session ended unexpectedly (Connection reset by peer)"),
        (None, "`alloc` did not complete: no exit status"),
        ("", "`alloc` did not complete: no exit status"),
    ],
)
def test_output_of_an_unfinished_command_gives_the_reason(reason, message, check):
    result = CommandResult(stdout="partial\n", stderr="err\n", returncode=None, reason=reason)
    with pytest.raises(CollectionError) as excinfo:
        dummy().output({"alloc": result}, "alloc", check=check)
    assert str(excinfo.value) == message


@pytest.mark.parametrize("check", [True, False])
@pytest.mark.parametrize(
    "timeout, shown",
    [(180.0, "180"), (42.5, "42.5"), (60, "60"), (0.25, "0.25")],
)
def test_output_of_a_timed_out_command_gives_the_time_limit(timeout, shown, check):
    cluster = dummy(timeout=timeout)
    with pytest.raises(CollectionError) as excinfo:
        cluster.output({"alloc": failed(TIMEOUT_EXIT_CODE, stdout="partial\n")}, "alloc", check=check)
    assert str(excinfo.value) == f"`alloc` timed out (time limit of {shown} s reached)"


def test_output_failure_detail_falls_back_to_stdout_when_stderr_is_blank():
    cluster = dummy()
    with pytest.raises(CollectionError) as excinfo:
        cluster.output({"alloc": failed(1, stdout="Error: project expired\n", stderr="\n \n")}, "alloc")
    assert str(excinfo.value) == "`alloc` failed (exit status 1): Error: project expired"


@pytest.mark.parametrize(
    "result, check",
    [
        (ok("a\n  b  \n", "noise on stderr"), True),
        (ok("a\n  b  \n"), False),
        (failed(1, stdout="a\n  b  \n", stderr="grep: no match"), False),
    ],
)
def test_output_returns_stdout_verbatim(result, check):
    assert dummy().output({"alloc": result}, "alloc", check=check) == "a\n  b  \n"


def test_output_names_the_full_command_once_collected():
    cluster = dummy(dummy_runner())
    cluster.collect(["compute"], today=TODAY, now=NOW)
    with pytest.raises(CollectionError) as excinfo:
        cluster.output({"alloc": failed(5)}, "alloc")
    assert str(excinfo.value) == "`show-allocations --all` failed (exit status 5)"
    with pytest.raises(CollectionError) as excinfo:
        cluster.output({}, "not-a-command")
    assert str(excinfo.value) == "no output from `not-a-command`"


def test_collection_error_is_an_exception_and_reexported():
    assert issubclass(CollectionError, Exception)
    assert hpcmon.clusters.CollectionError is CollectionError
    assert hpcmon.clusters.Cluster is Cluster


# ---------------------------------------------------------------------------
# Slurm: squeue command and parser
# ---------------------------------------------------------------------------

FIXTURE_JOBS = [
    Job(job_id="1234567", name="train|model [v2]", partition="standard-g", account="project_465000123",
        state="RUNNING", elapsed="1:02:03", time_limit="2-00:00:00", nodes=4, reason="nid[005000-005003]"),
    Job(job_id="1234570", name="cleanup", partition="dev-g", account="project_465000123",
        state="COMPLETING", elapsed="10:00", time_limit="30:00", nodes=1, reason="nid005100"),
    Job(job_id="1234568", name="eval", partition="small-g", account="project_465000123",
        state="PENDING", elapsed="0:00", time_limit="1:00:00", nodes=1, reason="(Priority)"),
    Job(job_id="1234569_[1-10]", name="sweep", partition="small-g", account="project_465000999",
        state="PENDING", elapsed="0:00", time_limit="30:00", nodes=1, reason="(AssocGrpGRESMinutes)"),
    Job(job_id="1234571", name="suspended-job", partition="small", account="project_465000123",
        state="SUSPENDED", elapsed="5:00", time_limit="30:00", nodes=1, reason="nid001001"),
]


def test_squeue_format_and_command_shape():
    assert SQUEUE_FORMAT == "%i|%P|%a|%T|%M|%l|%D|%R|%j"
    assert SQUEUE_FORMAT.endswith("|%j")  # the name, which may contain '|', comes last
    assert SQUEUE_COMMAND == "squeue --noheader --user=\"$(id -un)\" --format='%i|%P|%a|%T|%M|%l|%D|%R|%j'"
    assert shlex.split(SQUEUE_COMMAND) == [
        "squeue", "--noheader", "--user=$(id -un)", "--format=%i|%P|%a|%T|%M|%l|%D|%R|%j",
    ]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_squeue_command_passes_expected_arguments_in_a_local_shell():
    # ``squeue`` and ``id`` are stubbed by shell functions: nothing depends on
    # the local user or on Slurm being installed.
    stubs = (
        "squeue() { for arg in \"$@\"; do printf '%s\\n' \"$arg\"; done; }\n"
        "id() { [ \"$*\" = \"-un\" ] && printf 'jdoe\\n'; }\n"
    )
    env = {"PATH": os.environ.get("PATH", "")}

    out = subprocess.run(["bash", "-c", stubs + SQUEUE_COMMAND], capture_output=True, text=True, check=True,
                         env=env).stdout

    assert out.splitlines() == ["--noheader", "--user=jdoe", f"--format={SQUEUE_FORMAT}"]


def test_parse_squeue_fixture_exact_fields():
    assert parse_squeue(read_fixture("slurm/squeue.txt")) == FIXTURE_JOBS


def test_parse_squeue_orders_running_then_pending_then_others_stably():
    lines = read_fixture("slurm/squeue.txt").splitlines()
    jobs = parse_squeue("\n".join(reversed(lines)))
    assert [job.job_id for job in jobs] == ["1234570", "1234567", "1234569_[1-10]", "1234568", "1234571"]


def test_parse_squeue_orders_mixed_states_and_short_codes():
    text = "\n".join([
        "1|p|a|FAILED|0:00|1:00|1|n1|x1",
        "2|p|a|PD|0:00|1:00|1|(Resources)|x2",
        "3|p|a|running|0:01|1:00|1|n3|x3",
        "4|p|a|SUSPENDED|0:00|1:00|1|n4|x4",
        "5|p|a|REQUEUE_HOLD|0:00|1:00|1|(JobHeldAdmin)|x5",
        "6|p|a|R|0:02|1:00|1|n6|x6",
        "7|p|a|CONFIGURING|0:00|1:00|1|n7|x7",
    ])
    assert [job.job_id for job in parse_squeue(text)] == ["3", "6", "7", "2", "5", "1", "4"]


def test_parse_squeue_keeps_delimiters_and_brackets_in_job_names():
    jobs = parse_squeue("9|gpu|acc|RUNNING|1:00|2:00|2|nid[1-2]|a|b||c [d] (e)|\n")
    assert jobs == [Job(job_id="9", name="a|b||c [d] (e)|", partition="gpu", account="acc", state="RUNNING",
                        elapsed="1:00", time_limit="2:00", nodes=2, reason="nid[1-2]")]


def test_parse_squeue_strips_whitespace_around_fields():
    jobs = parse_squeue("  42 | p | a | PENDING | 0:00 | 2:00 | 1,024 | (Priority) |  my job  \r\n")
    assert jobs == [Job(job_id="42", name="my job", partition="p", account="a", state="PENDING",
                        elapsed="0:00", time_limit="2:00", nodes=1024, reason="(Priority)")]


@pytest.mark.parametrize("nodes", ["", "N/A", "-", "unlimited", "abc", "1-2", "?"])
def test_parse_squeue_non_numeric_node_count_is_none(nodes):
    [job] = parse_squeue(f"1|p|a|PENDING|0:00|1:00|{nodes}|(None)|n")
    assert job.nodes is None
    assert job.name == "n"


@pytest.mark.parametrize("text", ["", "\n", "   \n\t\n"])
def test_parse_squeue_empty_output_gives_no_jobs(text):
    assert parse_squeue(text) == []


def test_parse_squeue_skips_blank_lines():
    lines = read_fixture("slurm/squeue.txt").splitlines()
    text = "\n\n" + "\n   \n".join(lines) + "\n\t\n"
    assert parse_squeue(text) == FIXTURE_JOBS


@pytest.mark.parametrize(
    "line, shown",
    [
        ("squeue: error: Invalid user: jdoe", "squeue: error: Invalid user: jdoe"),
        ("  1|p|a|RUNNING|0:00|1:00|1|n1  ", "1|p|a|RUNNING|0:00|1:00|1|n1"),
        ("JOBID PARTITION NAME USER ST TIME NODES", "JOBID PARTITION NAME USER ST TIME NODES"),
    ],
)
def test_parse_squeue_malformed_line_raises(line, shown):
    text = "1234567|standard-g|p|RUNNING|0:01|1:00|1|nid1|ok\n" + line + "\n"
    with pytest.raises(CollectionError) as excinfo:
        parse_squeue(text)
    assert str(excinfo.value) == f"unexpected squeue output: {shown!r}"


# ---------------------------------------------------------------------------
# Slurm: cluster classes
# ---------------------------------------------------------------------------


def test_generic_slurm_cluster_interface():
    cluster = GenericSlurmCluster(make_config("cluster", type="slurm"), runner=FakeRunner())
    assert GenericSlurmCluster.type_name == "slurm"
    assert GenericSlurmCluster.display_name == "Slurm cluster"
    assert issubclass(GenericSlurmCluster, SlurmCluster)
    assert cluster.compute_commands() == {}
    assert cluster.storage_commands() == {}
    assert cluster.jobs_commands() == {"jobs": SQUEUE_COMMAND}
    assert cluster.commands() == {"jobs": SQUEUE_COMMAND}
    results = {"jobs": ok(read_fixture("slurm/squeue.txt"))}
    assert cluster.parse_compute(results, TODAY) == []
    assert cluster.parse_storage(results, TODAY) == []
    assert cluster.parse_jobs(results, TODAY) == FIXTURE_JOBS


def test_generic_slurm_cluster_collects_jobs_from_squeue():
    runner = FakeRunner({"jobs": read_fixture("slurm/squeue.txt")})
    cluster = GenericSlurmCluster(make_config("mycluster", type="slurm"), runner=runner)
    report = cluster.collect(today=TODAY, now=NOW)

    # All sections were requested, but only jobs are supported.
    assert runner.calls == [{"jobs": SQUEUE_COMMAND}]
    assert report.sections == ["jobs"]
    assert report.system == "Slurm cluster"
    assert report.jobs == FIXTURE_JOBS
    assert [job.job_id for job in report.running_jobs] == ["1234567", "1234570"]
    assert [job.job_id for job in report.pending_jobs] == ["1234568", "1234569_[1-10]"]
    assert report.budgets == [] and report.storage == []
    assert report.errors == [] and report.warnings == []


def test_generic_slurm_cluster_filters_jobs_by_project():
    runner = FakeRunner({"jobs": read_fixture("slurm/squeue.txt")})
    cluster = GenericSlurmCluster(make_config("c", type="slurm", projects=["Project_465000999"]), runner=runner)
    report = cluster.collect(["jobs"], today=TODAY, now=NOW)
    assert report.jobs == [FIXTURE_JOBS[3]]
    # Jobs-only: the configured projects are not checked against the cluster.
    assert report.warnings == ["4 job(s) of other accounts hidden by the 'projects' setting"]


@pytest.mark.parametrize(
    "result, error",
    [
        (None, f"jobs: no output from `{SQUEUE_COMMAND}`"),
        (failed(1, stderr="slurm_load_jobs error: Unable to contact slurm controller (connect failure)\n"),
         f"jobs: `{SQUEUE_COMMAND}` failed (exit status 1): "
         "slurm_load_jobs error: Unable to contact slurm controller (connect failure)"),
        (failed(None), f"jobs: `{SQUEUE_COMMAND}` did not complete: no exit status"),
        (CommandResult(returncode=None, reason="the SSH session timed out after 75 s"),
         f"jobs: `{SQUEUE_COMMAND}` did not complete: the SSH session timed out after 75 s"),
        (failed(TIMEOUT_EXIT_CODE), f"jobs: `{SQUEUE_COMMAND}` timed out (time limit of 60 s reached)"),
        (ok("Invalid user: jdoe\n"), "jobs: unexpected squeue output: 'Invalid user: jdoe'"),
    ],
    ids=["missing", "failed", "unfinished", "unfinished-reason", "timeout", "garbage"],
)
def test_slurm_cluster_jobs_errors(result, error):
    runner = FakeRunner({} if result is None else {"jobs": result})
    cluster = GenericSlurmCluster(make_config("c", type="slurm", timeout=60.0), runner=runner)
    report = cluster.collect(today=TODAY, now=NOW)
    assert report.errors == [error]
    assert report.jobs is None
    assert report.failed_sections == ["jobs"]


def test_slurm_cluster_with_no_jobs_reports_empty_list():
    cluster = GenericSlurmCluster(make_config("c", type="slurm"), runner=FakeRunner({"jobs": ""}))
    report = cluster.collect(["jobs"], today=TODAY, now=NOW)
    assert report.jobs == []
    assert report.errors == []


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


EXPECTED_TYPES = {
    "lumi": LumiCluster,
    "jupiter": JupiterCluster,
    "leonardo": LeonardoCluster,
    "meluxina": MeluxinaCluster,
    "slurm": GenericSlurmCluster,
}


def test_cluster_types_registry():
    assert set(CLUSTER_TYPES) == {"lumi", "jupiter", "leonardo", "meluxina", "slurm"}
    assert CLUSTER_TYPES == EXPECTED_TYPES
    for type_name, cls in CLUSTER_TYPES.items():
        assert cls.type_name == type_name
        assert cls.display_name
        assert issubclass(cls, SlurmCluster)


@pytest.mark.parametrize("type_name", sorted(EXPECTED_TYPES))
def test_create_cluster_returns_the_right_class_with_the_given_runner(type_name):
    config = make_config(f"my-{type_name}", type=type_name)
    runner = FakeRunner()
    cluster = create_cluster(config, runner=runner)

    assert type(cluster) is EXPECTED_TYPES[type_name]
    assert cluster.runner is runner
    assert cluster.config is config
    assert cluster.name == f"my-{type_name}"


@pytest.mark.parametrize("type_name", sorted(EXPECTED_TYPES))
def test_create_cluster_without_runner_builds_an_ssh_runner(type_name):
    config = make_config("x", type=type_name, ssh=["ssh", "-p", "2222", "host"], timeout=12.0)
    cluster = create_cluster(config)
    assert isinstance(cluster.runner, SSHRunner)
    assert cluster.runner.ssh_command == ["ssh", "-p", "2222", "host"]
    assert cluster.runner.timeout == 12.0


@pytest.mark.parametrize("type_name", ["pbs", "LUMI", "", "dummy"])
def test_create_cluster_unknown_type_raises_key_error(type_name):
    config = make_config("x", type="lumi")
    config.type = type_name  # make_config would replace an empty type by the name
    with pytest.raises(KeyError) as excinfo:
        create_cluster(config, runner=FakeRunner())
    assert excinfo.value.args == (f"unsupported cluster type: {type_name!r}",)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


def test_clusters_package_exports():
    assert sorted(hpcmon.clusters.__all__) == sorted([
        "CLUSTER_TYPES", "Cluster", "CollectionError", "GenericSlurmCluster", "JupiterCluster",
        "LeonardoCluster", "LumiCluster", "MeluxinaCluster", "SlurmCluster", "create_cluster",
    ])
    for name in hpcmon.clusters.__all__:
        assert getattr(hpcmon.clusters, name) is not None
