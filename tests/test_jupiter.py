"""Tests for the JUPITER interface (:mod:`hpcmon.clusters.jupiter`).

Budgets come from ``jutil user cpuquota`` and storage quotas from ``jutil user
dataquota`` / ``jutil project dataquota`` (JSON output).  The expected values
below are derived from the text of the real (anonymised) fixtures in
``tests/fixtures/jupiter``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import stat
import subprocess
from typing import Any, Dict, List, Optional

import pytest

from hpcmon.clusters import CLUSTER_TYPES, create_cluster
from hpcmon.clusters.base import HOME_COMMAND, CollectionError
from hpcmon.clusters.jupiter import (
    FIXED_MODE,
    PROJECT_DATAQUOTA_COMMAND,
    JupiterCluster,
    parse_json_documents,
)
from hpcmon.clusters.slurm import SQUEUE_COMMAND
from hpcmon.models import ComputeBudget, StorageQuota
from hpcmon.runner import CommandResult, RemoteError
from tests.conftest import NOW, TODAY, FakeRunner, failed, make_config, ok, read_fixture

KIB = 1024
PROJECT = "e-dev-2026d00-001"
USER = "jdoe123456789"
HOME = "/p/home/jusers/jdoe123456789/jupiter"

CPUQUOTA_COMMAND = "jutil user cpuquota --currsys --format json"
USERDATA_COMMAND = "jutil user dataquota --format json"

#: The budget described by ``jupiter/cpuquota.json`` (fixed accounting mode).
#: The project is "E-DEV-2026D00-001" and the budget account its lower-case
#: spelling, so there is no separate account.  The remaining amount comes from
#: total - used, so "coreh-avail" is not kept.
FIXTURE_BUDGET = ComputeBudget(
    project=PROJECT,
    resource="jupiter_booster",
    unit="core-h",
    total=864000.0,
    used=62738.88,
    available=None,
    start=dt.date(2026, 6, 1),
    end=dt.date(2027, 5, 31),
    account=None,
)

#: The budget as reported by the cluster, with its GPU-hours rate
#: (288 core-hours per node-hour of 4 GH200, i.e. 72 per GPU-hour).
FIXTURE_REPORTED = dataclasses.replace(FIXTURE_BUDGET, gpu_hours_per_unit=1 / 72)

#: The personal home quota from ``jupiter/userdata.json`` (named by $HOME).
FIXTURE_HOME_QUOTA = StorageQuota(
    name=HOME,
    project=None,
    used_bytes=1187264 * KIB,
    limit_bytes=31457280 * KIB,
    files_used=4128,
    files_limit=120000,
)

#: The project quotas from ``jupiter/projectdata.json``, in file order.
FIXTURE_PROJECT_QUOTAS = [
    StorageQuota(
        name=f"/e/scratch/{PROJECT}",
        project=PROJECT,
        used_bytes=0,
        limit_bytes=107374182400 * KIB,
        files_used=1,
        files_limit=6000000,
    ),
    StorageQuota(
        name=f"/e/project1/{PROJECT}",
        project=PROJECT,
        used_bytes=198366876 * KIB,
        limit_bytes=17179869184 * KIB,
        files_used=18652,
        files_limit=3000000,
    ),
    StorageQuota(
        name=f"/e/fscratch/{PROJECT}",
        project=PROJECT,
        used_bytes=0,
        limit_bytes=32212254720 * KIB,
        files_used=1,
        files_limit=6000000,
    ),
]

#: The user's own usage in the project directory (``jupiter/userdata.json``),
#: shown when the project data quotas cannot be read.
FIXTURE_FALLBACK_QUOTA = StorageQuota(
    name=f"exa_project1 of {PROJECT} (your usage)",
    project=PROJECT,
    used_bytes=198342876 * KIB,
    limit_bytes=None,
    files_used=18634,
    files_limit=None,
)


# -- helpers ------------------------------------------------------------------


def fixture_results(**overrides: Any) -> Dict[str, Any]:
    """Results of all the JUPITER commands, from the fixtures."""
    results: Dict[str, Any] = {
        "cpuquota": read_fixture("jupiter/cpuquota.json"),
        "userdata": read_fixture("jupiter/userdata.json"),
        "projectdata": read_fixture("jupiter/projectdata.json"),
        "home": HOME + "\n",
        "jobs": read_fixture("slurm/squeue.txt"),
    }
    results.update(overrides)
    return {key: ok(value) if isinstance(value, str) else value for key, value in results.items()}


def make_cluster(results: Optional[Dict[str, Any]] = None, **config: Any) -> JupiterCluster:
    return JupiterCluster(make_config("jupiter", **config), runner=FakeRunner(results))


def budget_of(entry: Dict[str, Any]) -> ComputeBudget:
    """Parse a ``cpuquota`` output made of the single ``entry``."""
    budgets = make_cluster().parse_compute({"cpuquota": ok(json.dumps([entry]))}, TODAY)
    assert len(budgets) == 1
    return budgets[0]


def storage_of(
    userdata: List[Dict[str, Any]],
    projectdata: Optional[List[Dict[str, Any]]] = None,
    home: Optional[CommandResult] = None,
) -> List[StorageQuota]:
    """Parse the storage section of the given ``jutil`` entries."""
    results = {
        "userdata": ok(json.dumps(userdata)),
        "projectdata": ok(json.dumps(projectdata or [])),
        "home": home if home is not None else ok(HOME + "\n"),
    }
    return make_cluster().parse_storage(results, TODAY)


def personal_entry(**fields: Any) -> Dict[str, Any]:
    entry = {
        "username": USER,
        "group": USER,
        "filesystem": "exa_home",
        "data-usage": "1KB",
        "data-soft-limit": "2KB",
        "inode-usage": "3",
        "inode-soft-limit": "4",
    }
    entry.update(fields)
    return {key: value for key, value in entry.items() if value is not None}


def project_entry(**fields: Any) -> Dict[str, Any]:
    entry = {
        "project": PROJECT,
        "filesystem": "exa_project1",
        "project-dir": f"/e/project1/{PROJECT}",
        "data-usage": "10KB",
        "data-soft-limit": "20KB",
        "inode-usage": "30",
        "inode-soft-limit": "40",
    }
    entry.update(fields)
    return {key: value for key, value in entry.items() if value is not None}


# -- commands -------------------------------------------------------------------


def test_cluster_is_registered_under_type_jupiter():
    assert CLUSTER_TYPES["jupiter"] is JupiterCluster
    assert JupiterCluster.type_name == "jupiter"
    assert JupiterCluster.display_name == "JUPITER"
    assert isinstance(create_cluster(make_config("jsc", type="jupiter"), runner=FakeRunner()), JupiterCluster)


def test_jupiter_supports_every_section():
    assert JupiterCluster.supported_sections == ("compute", "storage", "jobs")


def test_compute_commands_query_the_cpuquota_of_the_current_system_as_json():
    assert make_cluster().compute_commands() == {"cpuquota": CPUQUOTA_COMMAND}


def test_storage_commands_query_user_and_project_data_quotas_and_home():
    commands = make_cluster().storage_commands()
    assert commands == {
        "userdata": USERDATA_COMMAND,
        "projectdata": PROJECT_DATAQUOTA_COMMAND,
        "home": HOME_COMMAND,
    }
    assert list(commands) == ["userdata", "projectdata", "home"]


def test_jobs_commands_use_squeue():
    assert make_cluster().jobs_commands() == {"jobs": SQUEUE_COMMAND}


def test_commands_run_jobs_then_compute_then_storage():
    assert list(make_cluster().commands()) == ["jobs", "cpuquota", "userdata", "projectdata", "home"]
    assert list(make_cluster().commands(["storage", "compute"])) == ["cpuquota", "userdata", "projectdata", "home"]


def test_fixed_mode_constant():
    assert FIXED_MODE == "fixed"


def test_project_dataquota_command_fails_when_projects_cannot_be_listed():
    command = PROJECT_DATAQUOTA_COMMAND
    assert command.startswith("projects=$(jutil user projects --noheader --format parsable) || exit $?; ")
    assert "rc=0; " in command


def test_project_dataquota_command_loops_over_unique_project_lines_and_keeps_failures():
    command = PROJECT_DATAQUOTA_COMMAND
    assert "for p in $(printf '%s\\n' \"$projects\" | grep '|' | cut -d'|' -f1 | sort -u); do " in command
    assert 'jutil project dataquota --project "$p" --format json || rc=$?; ' in command
    assert command.endswith("done; exit $rc")


STUB_JUTIL = """\
#!/bin/sh
if [ "$*" = "user projects --noheader --format parsable" ]; then
    if [ -n "$STUB_LIST_RC" ]; then
        echo "jutil: cannot list the projects" >&2
        exit "$STUB_LIST_RC"
    fi
    echo "a line without the separator"
    for p in $STUB_PROJECTS; do printf '%s|Some title|PI name\\n' "$p"; done
    exit 0
fi
echo "$4" >> "$STUB_LOG"
case "$*" in
    "project dataquota --project bad --format json")
        echo "jutil: no data quota for bad" >&2
        exit 3 ;;
    "project dataquota --project worse --format json")
        echo "jutil: no data quota for worse" >&2
        exit 5 ;;
    "project dataquota --project "*" --format json")
        printf '[{"project": "%s", "project-dir": "/p/%s", "data-usage": "1KB"}]\\n' "$4" "$4"
        exit 0 ;;
esac
echo "unexpected arguments: $*" >&2
exit 99
"""


@pytest.mark.parametrize(
    "projects, list_rc, returncode, expected_calls, expected_projects",
    [
        ("alpha beta", "", 0, ["alpha", "beta"], ["alpha", "beta"]),
        ("alpha bad beta", "", 3, ["alpha", "bad", "beta"], ["alpha", "beta"]),
        ("alpha bad worse", "", 5, ["alpha", "bad", "worse"], ["alpha"]),
        ("beta alpha beta alpha", "", 0, ["alpha", "beta"], ["alpha", "beta"]),
        ("", "", 0, [], []),
        ("alpha beta", "7", 7, [], []),
    ],
    ids=["ok", "one-failure", "last-failure-wins", "sorted-unique", "no-projects", "listing-fails"],
)
def test_project_dataquota_command_runs_in_a_local_shell(
    tmp_path, projects, list_rc, returncode, expected_calls, expected_projects
):
    """Run the real shell loop against a stub ``jutil`` (no remote host)."""
    jutil = tmp_path / "jutil"
    jutil.write_text(STUB_JUTIL, encoding="utf-8")
    jutil.chmod(jutil.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "calls.log"
    # A minimal environment: nothing from the user's shell setup leaks in.
    env = {
        "PATH": f"{tmp_path}{os.pathsep}{os.defpath}",
        "STUB_PROJECTS": projects,
        "STUB_LIST_RC": list_rc,
        "STUB_LOG": str(log),
        "LC_ALL": "C",
    }

    process = subprocess.run(
        ["bash", "-c", PROJECT_DATAQUOTA_COMMAND], env=env, capture_output=True, text=True, timeout=10
    )

    assert process.returncode == returncode
    assert "unexpected arguments" not in process.stderr
    calls = log.read_text(encoding="utf-8").split() if log.exists() else []
    # Lines without "|" (headers, messages) are not taken for project names.
    assert calls == expected_calls
    entries = parse_json_documents(process.stdout)
    assert [entry["project"] for entry in entries] == expected_projects
    assert [entry["project-dir"] for entry in entries] == [f"/p/{name}" for name in expected_projects]
    if list_rc:
        assert "jutil: cannot list the projects" in process.stderr


# -- parse_json_documents ------------------------------------------------------


@pytest.mark.parametrize("text", ["", "   ", "\n\t \n", "[]", "[]\n[]\n"])
def test_parse_json_documents_without_entries(text):
    assert parse_json_documents(text) == []


@pytest.mark.parametrize(
    "text",
    [
        '[{"a": 1}]\n  [{"b": 2}, {"c": 3}]\n',
        '\n[{"a": 1}]\n\n\t[{"b": 2}]  [{"c": 3}]   ',
        '[{"a": 1}][{"b": 2}, {"c": 3}]',
        '[{"a": 1}]{"b": 2}\n{"c": 3}',
    ],
    ids=["newline-separated", "mixed-whitespace", "adjacent", "arrays-and-objects"],
)
def test_parse_json_documents_concatenated(text):
    assert parse_json_documents(text) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_parse_json_documents_single_object():
    assert parse_json_documents('{"project": "x", "data-usage": "0"}') == [{"project": "x", "data-usage": "0"}]


@pytest.mark.parametrize(
    "text, expected",
    [
        ('[1, "x", null, true, [], [{"nested": 1}], {"a": 1}]', [{"a": 1}]),
        ('42\n"text"\n[{"a": 1}]', [{"a": 1}]),
        ("null", []),
    ],
)
def test_parse_json_documents_ignores_non_object_items(text, expected):
    assert parse_json_documents(text) == expected


def test_parse_json_documents_reads_the_projectdata_fixture():
    entries = parse_json_documents(read_fixture("jupiter/projectdata.json"))
    assert [entry["project-dir"] for entry in entries] == [quota.name for quota in FIXTURE_PROJECT_QUOTAS]


@pytest.mark.parametrize(
    "text",
    [
        "jutil: command not found",
        '[{"a": 1}',
        '[{"a": 1}] trailing garbage',
        '[{"a": 1},]',
    ],
)
def test_parse_json_documents_rejects_invalid_json(text):
    with pytest.raises(CollectionError) as excinfo:
        parse_json_documents(text)
    assert str(excinfo.value).startswith("invalid JSON output from jutil: ")
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


def test_parse_json_documents_reports_the_decoder_message():
    with pytest.raises(CollectionError) as excinfo:
        parse_json_documents("oops")
    assert str(excinfo.value) == "invalid JSON output from jutil: Expecting value: line 1 column 1 (char 0)"


# -- compute budgets ----------------------------------------------------------


def test_parse_compute_fixture_gives_the_fixed_budget():
    results = fixture_results()
    budgets = make_cluster().parse_compute(results, TODAY)

    assert budgets == [FIXTURE_BUDGET]
    budget = budgets[0]
    # The project is lower-cased; the budget account only differs by case.
    assert budget.project == "e-dev-2026d00-001"
    assert budget.account is None
    # Remaining is total - used; "coreh-avail" is not used (available None).
    assert budget.available is None
    assert budget.remaining == pytest.approx(801261.12)
    assert budget.month_total is None and budget.month_used is None and budget.month_remaining is None
    assert budget.note is None
    assert budget.days_left(TODAY) == (dt.date(2027, 5, 31) - TODAY).days


@pytest.mark.parametrize("text", ["", "[]", "  \n"])
def test_parse_compute_without_budgets(text):
    assert make_cluster().parse_compute({"cpuquota": ok(text)}, TODAY) == []


def test_parse_compute_several_concatenated_budgets():
    text = json.dumps([{"budget-account": "a"}]) + "\n" + json.dumps([{"budget-account": "b"}, {"budget-account": "c"}])
    budgets = make_cluster().parse_compute({"cpuquota": ok(text)}, TODAY)
    assert [budget.project for budget in budgets] == ["a", "b", "c"]


def test_parse_compute_invalid_json_raises_collection_error():
    with pytest.raises(CollectionError) as excinfo:
        make_cluster().parse_compute({"cpuquota": ok("<html>")}, TODAY)
    assert str(excinfo.value).startswith("invalid JSON output from jutil: ")


@pytest.mark.parametrize(
    "result, message",
    [
        (failed(1, stderr="jutil: error\nPermission denied\n"), "`cpuquota` failed (exit status 1): Permission denied"),
        (failed(2, stdout="partial\nlast stdout line\n"), "`cpuquota` failed (exit status 2): last stdout line"),
        (failed(2), "`cpuquota` failed (exit status 2)"),
        (failed(124), "`cpuquota` timed out (time limit of 180 s reached)"),
        (failed(None), "`cpuquota` did not complete: no exit status"),
        (
            CommandResult(returncode=None, reason="the SSH session timed out after 195 s"),
            "`cpuquota` did not complete: the SSH session timed out after 195 s",
        ),
    ],
    ids=["stderr-detail", "stdout-detail", "no-detail", "timeout-124", "unfinished", "unfinished-with-reason"],
)
def test_parse_compute_command_failures(result, message):
    with pytest.raises(CollectionError) as excinfo:
        make_cluster().parse_compute({"cpuquota": result}, TODAY)
    assert str(excinfo.value) == message


def test_parse_compute_timeout_names_the_configured_limit():
    with pytest.raises(CollectionError) as excinfo:
        make_cluster(timeout=42.5).parse_compute({"cpuquota": failed(124)}, TODAY)
    assert str(excinfo.value) == "`cpuquota` timed out (time limit of 42.5 s reached)"


def test_parse_compute_missing_output():
    with pytest.raises(CollectionError) as excinfo:
        make_cluster().parse_compute({}, TODAY)
    assert str(excinfo.value) == "no output from `cpuquota`"


@pytest.mark.parametrize(
    "fields, project, account",
    [
        ({"budget-account": "acct", "project": "PROJ"}, "proj", "acct"),
        ({"budget-account": "ACCT", "project": "Proj"}, "proj", "ACCT"),
        ({"budget-account": "E-DEV", "project": "e-dev"}, "e-dev", None),
        ({"budget-account": "e-dev", "project": "E-DEV"}, "e-dev", None),
        ({"budget-account": "", "project": "PROJ"}, "proj", None),
        ({"project": "PROJ"}, "proj", None),
        ({"budget-account": "ACCT"}, "acct", None),
        ({"budget-account": "acct", "project": ""}, "acct", None),
        ({"budget-account": "acct", "project": None}, "acct", None),
        ({"budget-account": 12345}, "12345", None),
        ({"budget-account": None, "project": None}, "?", None),
        ({}, "?", None),
    ],
    ids=[
        "account-differs",
        "account-differs-case-kept",
        "same-ignoring-case",
        "same-ignoring-case-2",
        "empty-account",
        "project-only",
        "account-only",
        "empty-project",
        "null-project",
        "numeric-account",
        "both-null",
        "nothing",
    ],
)
def test_budget_project_and_account(fields, project, account):
    budget = budget_of(fields)
    assert (budget.project, budget.account) == (project, account)


@pytest.mark.parametrize(
    "fields, resource",
    [
        ({"cont-partition": "jupiter_booster", "system": "jupiter"}, "jupiter_booster"),
        ({"cont-partition": "", "system": "jupiter"}, "jupiter"),
        ({"system": "jupiter"}, "jupiter"),
        ({}, "compute"),
    ],
)
def test_budget_resource_fallbacks(fields, resource):
    assert budget_of(fields).resource == resource


def test_budget_with_only_missing_values():
    budget = budget_of({})
    assert budget == ComputeBudget(project="?", resource="compute", unit="core-h")
    assert budget.remaining is None


def test_budget_accepts_numeric_json_values():
    budget = budget_of({"coreh-quota-total": 1000, "coreh-consumed-total": 250.5, "coreh-avail": 749.5})
    assert (budget.total, budget.used, budget.available) == (1000.0, 250.5, None)
    assert budget.remaining == 749.5


def test_budget_without_available_computes_remaining():
    budget = budget_of({"coreh-quota-total": "1,000", "coreh-consumed-total": "250"})
    assert budget.available is None
    assert budget.remaining == 750.0


@pytest.mark.parametrize(
    "fields, available, remaining",
    [
        # Total and used known: the remaining amount is their difference.
        ({"coreh-quota-total": "1000", "coreh-consumed-total": "250", "coreh-avail": "10"}, None, 750.0),
        # Otherwise "coreh-avail" is the best figure there is.
        ({"coreh-quota-total": "1000", "coreh-avail": "40"}, 40.0, 40.0),
        ({"coreh-consumed-total": "250", "coreh-avail": "40"}, 40.0, 40.0),
        ({"coreh-quota-total": "-", "coreh-consumed-total": "250", "coreh-avail": 40}, 40.0, 40.0),
        ({"coreh-avail": "40"}, 40.0, 40.0),
        ({"coreh-quota-total": "1000"}, None, None),
    ],
    ids=["total-and-used", "total-only", "used-only", "dash-total", "avail-only", "nothing-usable"],
)
def test_budget_available_is_used_only_without_total_or_used(fields, available, remaining):
    budget = budget_of(fields)
    assert budget.available == available
    assert budget.remaining == remaining


def test_budget_dash_values_are_missing():
    budget = budget_of({"coreh-quota-total": "-", "coreh-consumed-total": "-", "coreh-avail": "-"})
    assert (budget.total, budget.used, budget.available, budget.remaining) == (None, None, None, None)


@pytest.mark.parametrize(
    "start, end, expected",
    [
        ("2026-06-01", "2027-05-31", (dt.date(2026, 6, 1), dt.date(2027, 5, 31))),
        ("20260601", "31.05.2027", (dt.date(2026, 6, 1), dt.date(2027, 5, 31))),
        ("2026-06-01 00:00:00", "2027-05-31 23:59:59", (dt.date(2026, 6, 1), dt.date(2027, 5, 31))),
        ("2026-06-01T00:00:00", "2027-05-31T23:59:59+01:00", (dt.date(2026, 6, 1), dt.date(2027, 5, 31))),
        (20260601, 20270531, (dt.date(2026, 6, 1), dt.date(2027, 5, 31))),
        ("-", None, (None, None)),
    ],
    ids=["iso", "compact-and-german", "timestamps", "iso-timestamps", "numeric", "missing"],
)
def test_budget_accounting_period(start, end, expected):
    budget = budget_of({"acct-period-start": start, "acct-period-end": end})
    assert (budget.start, budget.end) == expected


def test_budget_with_invalid_date_raises_value_error():
    with pytest.raises(ValueError, match="invalid date"):
        budget_of({"acct-period-end": "end of February"})


@pytest.mark.parametrize("mode", [None, "", "fixed", "FIXED", "Fixed"])
def test_budget_without_monthly_mode_ignores_monthly_figures(mode):
    entry = {
        "coreh-quota-total": "1000",
        "coreh-consumed-total": "100",
        "coreh-avail": "10",  # differs from total - used, but not reported for fixed budgets
        "coreh-quota-curr-month": "84000",
        "coreh-avail-curr-month": "80000",
        "coreh-consumed-curr-month": "4000",
    }
    if mode is not None:
        entry["acct-mode"] = mode
    budget = budget_of(entry)
    assert (budget.month_total, budget.month_used, budget.note) == (None, None, None)
    assert budget.remaining == 900.0


@pytest.mark.parametrize(
    "fields, month_total, month_used",
    [
        ({}, None, None),
        ({"coreh-quota-curr-month": "84000"}, 84000.0, None),
        ({"coreh-avail-curr-month": "84000"}, None, None),
        ({"coreh-consumed-curr-month": "12"}, None, 12.0),
        ({"coreh-quota-curr-month": "-", "coreh-consumed-curr-month": "12"}, None, 12.0),
        ({"coreh-avail-curr-month": "-", "coreh-consumed-curr-month": "12"}, None, 12.0),
        ({"coreh-quota-curr-month": "84000", "coreh-consumed-curr-month": "1200.5"}, 84000.0, 1200.5),
        # Without the monthly quota, it is what is available plus what was used.
        ({"coreh-avail-curr-month": "82799.5", "coreh-consumed-curr-month": "1200.5"}, 84000.0, 1200.5),
        # The quota of the month wins over the derived value.
        (
            {"coreh-quota-curr-month": "84000", "coreh-avail-curr-month": "1", "coreh-consumed-curr-month": "1200.5"},
            84000.0,
            1200.5,
        ),
        ({"coreh-quota-curr-month": 0, "coreh-consumed-curr-month": 0}, 0.0, 0.0),
        ({"coreh-avail-curr-month": 80000, "coreh-consumed-curr-month": 4000}, 84000.0, 4000.0),
    ],
    ids=[
        "no-data",
        "quota-only",
        "avail-only",
        "used-only",
        "dash-quota",
        "dash-avail",
        "quota-and-used",
        "avail-and-used",
        "quota-wins",
        "zero-quota",
        "numeric",
    ],
)
def test_monthly_budget_figures(fields, month_total, month_used):
    budget = budget_of(dict(fields, **{"acct-mode": "Monthly"}))
    assert (budget.month_total, budget.month_used) == (month_total, month_used)
    assert budget.note == "monthly accounting"


def test_monthly_budget_remaining_of_the_month():
    budget = budget_of(
        {"acct-mode": "monthly", "coreh-quota-curr-month": "84000", "coreh-consumed-curr-month": "1200.5"}
    )
    assert budget.month_remaining == 82799.5


@pytest.mark.parametrize(
    "fields, note",
    [
        (
            {"coreh-quota-total": "1000", "coreh-consumed-total": "200", "coreh-avail": "300"},
            "monthly accounting; available now: 300",
        ),
        (
            {"coreh-quota-total": "100000", "coreh-consumed-total": "0", "coreh-avail": "12345.6"},
            "monthly accounting; available now: 12,346",
        ),
        (
            {"coreh-quota-total": 2000000, "coreh-consumed-total": 0, "coreh-avail": 1234567},
            "monthly accounting; available now: 1,234,567",
        ),
        # Nothing to add when it is what is left anyway.
        ({"coreh-quota-total": "1000", "coreh-consumed-total": "250", "coreh-avail": "750"}, "monthly accounting"),
        ({"coreh-quota-total": "1000", "coreh-consumed-total": "250"}, "monthly accounting"),
        # Without total or used, "coreh-avail" is the remaining amount itself.
        ({"coreh-quota-total": "1000", "coreh-avail": "300"}, "monthly accounting"),
        ({"coreh-avail": "-"}, "monthly accounting"),
    ],
    ids=["differs", "thousands", "numeric", "same", "no-avail", "avail-is-remaining", "dash-avail"],
)
def test_monthly_budget_available_now_note(fields, note):
    assert budget_of(dict(fields, **{"acct-mode": "monthly"})).note == note


def test_fixed_budget_never_gets_the_available_now_note():
    budget = budget_of(
        {"acct-mode": "fixed", "coreh-quota-total": "1000", "coreh-consumed-total": "200", "coreh-avail": "300"}
    )
    assert budget.note is None
    assert budget.remaining == 800.0


def test_monthly_budget_no_available_now_note_when_figures_agree_to_the_cent():
    budget = budget_of(
        {
            "acct-mode": "monthly",
            "coreh-quota-total": "500000",
            "coreh-consumed-total": "62738.09",
            "coreh-avail": "437261.91",
        }
    )
    assert budget.note == "monthly accounting"


@pytest.mark.parametrize(
    "fields, note",
    [
        ({"cont-state": "normal"}, None),
        ({"cont-state": "Normal"}, None),
        ({"cont-state": ""}, None),
        ({}, None),
        ({"cont-state": "blocked"}, "state: blocked"),
        ({"cont-state": "EXPIRED"}, "state: expired"),
        ({"acct-mode": "fixed", "cont-state": "blocked"}, "state: blocked"),
        ({"acct-mode": "quarterly", "cont-state": "BLOCKED"}, "quarterly accounting; state: blocked"),
        ({"acct-mode": "quarterly", "cont-state": "normal"}, "quarterly accounting"),
        (
            {
                "acct-mode": "monthly",
                "coreh-quota-curr-month": "10",
                "coreh-consumed-curr-month": "1",
                "cont-state": "low",
            },
            "monthly accounting; state: low",
        ),
    ],
)
def test_budget_notes_for_mode_and_state(fields, note):
    assert budget_of(fields).note == note


@pytest.mark.parametrize(
    "fields, note",
    [
        ({"project-state": "A"}, None),
        ({"project-state": "a"}, None),
        ({"project-state": ""}, None),
        ({}, None),
        ({"project-state": "C"}, "project state: C"),
        ({"project-state": "closed"}, "project state: closed"),
        ({"project-state": "X", "cont-state": "blocked"}, "state: blocked; project state: X"),
    ],
    ids=["active", "active-lower", "empty", "absent", "closed", "word-kept-as-is", "after-state"],
)
def test_budget_project_state_note(fields, note):
    assert budget_of(fields).note == note


def test_budget_notes_are_joined_in_order():
    budget = budget_of(
        {
            "acct-mode": "Quarterly",
            "coreh-quota-total": "1000",
            "coreh-consumed-total": "100",
            "coreh-avail": "50",
            "cont-state": "Blocked",
            "project-state": "C",
        }
    )
    assert budget.note == "quarterly accounting; available now: 50; state: blocked; project state: C"


@pytest.mark.parametrize(
    "fields",
    [{"acct-mode": None}, {"cont-state": None}, {"project-state": None}],
    ids=["acct-mode", "cont-state", "project-state"],
)
def test_budget_null_mode_and_states_are_missing(fields):
    assert budget_of(fields).note is None


# -- storage quotas ---------------------------------------------------------


def test_parse_storage_fixtures():
    quotas = make_cluster().parse_storage(fixture_results(), TODAY)
    # The user's usage in the project directory is covered by the project
    # quota (same project and filesystem), so it is not repeated.
    assert quotas == [FIXTURE_HOME_QUOTA] + FIXTURE_PROJECT_QUOTAS


def test_parse_storage_fixture_home_fractions():
    home = make_cluster().parse_storage(fixture_results(), TODAY)[0]
    assert home.free_bytes == (31457280 - 1187264) * KIB
    assert home.used_fraction == pytest.approx(1187264 / 31457280)
    assert home.files_fraction == pytest.approx(4128 / 120000)


@pytest.mark.parametrize(
    "home",
    [
        failed(1, stderr="printf: error"),
        failed(124),
        failed(None),
        ok(""),
        ok("  \n"),
    ],
    ids=["failed", "timeout-124", "unfinished", "empty", "blank"],
)
def test_home_quota_falls_back_to_the_filesystem_name(home):
    quotas = make_cluster().parse_storage(fixture_results(home=home), TODAY)
    assert quotas[0] == StorageQuota(
        name="exa_home",
        project=None,
        used_bytes=1187264 * KIB,
        limit_bytes=31457280 * KIB,
        files_used=4128,
        files_limit=120000,
    )
    assert quotas[1:] == FIXTURE_PROJECT_QUOTAS


def test_home_quota_without_home_output_uses_the_filesystem_name():
    results = fixture_results()
    del results["home"]
    quotas = make_cluster().parse_storage(results, TODAY)
    assert quotas[0].name == "exa_home"


def test_home_path_is_stripped():
    quotas = storage_of([personal_entry()], home=ok("  /p/home/jusers/x/jupiter  \n\n"))
    assert [quota.name for quota in quotas] == ["/p/home/jusers/x/jupiter"]


@pytest.mark.parametrize(
    "filesystem, home, name",
    [
        ("exa_home", ok("/home/x\n"), "/home/x"),
        ("HOME", ok("/home/x\n"), "HOME"),  # case-sensitive match
        ("exa_scratch", ok("/home/x\n"), "exa_scratch"),
        (None, ok("/home/x\n"), "/home/x"),  # no filesystem: assumed to be the home
        (None, failed(1), "home"),
    ],
)
def test_personal_quota_name(filesystem, home, name):
    quotas = storage_of([personal_entry(filesystem=filesystem)], home=home)
    assert [quota.name for quota in quotas] == [name]
    assert quotas[0].project is None


@pytest.mark.parametrize(
    "fields",
    [
        {"group": None},
        {"group": "", "username": ""},
        {"data-usage": None},
        {"group": PROJECT, "data-usage": None},
    ],
    ids=["no-group", "empty-group", "no-data-usage", "project-group-without-data-usage"],
)
def test_user_entries_without_group_or_usage_are_skipped(fields):
    assert storage_of([personal_entry(**fields)]) == []


def test_project_group_usage_without_project_quota_is_a_fallback_row():
    entry = personal_entry(group=PROJECT, filesystem="exa_scratch", **{"data-usage": "5KB", "inode-usage": "6"})
    assert storage_of([entry]) == [
        StorageQuota(
            name=f"exa_scratch of {PROJECT} (your usage)",
            project=PROJECT,
            used_bytes=5 * KIB,
            limit_bytes=None,
            files_used=6,
            files_limit=None,
        )
    ]


def test_user_entry_without_username_is_a_fallback_row_of_its_group():
    quotas = storage_of([personal_entry(username=None)])
    assert quotas == [
        StorageQuota(name=f"exa_home of {USER} (your usage)", project=USER, used_bytes=KIB, files_used=3)
    ]


@pytest.mark.parametrize(
    "group, filesystem",
    [(PROJECT, "exa_project1"), (PROJECT.upper(), "EXA_PROJECT1"), (PROJECT, "Exa_Project1")],
    ids=["same", "upper-case", "mixed-case-filesystem"],
)
def test_project_group_usage_covered_by_a_project_quota_is_not_repeated(group, filesystem):
    quotas = storage_of([personal_entry(group=group, filesystem=filesystem)], [project_entry()])
    assert [quota.name for quota in quotas] == [f"/e/project1/{PROJECT}"]


def test_project_quota_covers_a_project_group_case_insensitively_on_its_side_too():
    quotas = storage_of(
        [personal_entry(group=PROJECT, filesystem="exa_project1")],
        [project_entry(project=PROJECT.upper(), filesystem="EXA_PROJECT1")],
    )
    assert [quota.name for quota in quotas] == [f"/e/project1/{PROJECT}"]


@pytest.mark.parametrize(
    "project_fields",
    [
        {"filesystem": "exa_scratch"},  # same project, other filesystem
        {"project": "e-other"},  # same filesystem, other project
        {"data-usage": None},  # a project entry without data does not count
    ],
    ids=["other-filesystem", "other-project", "project-entry-without-data"],
)
def test_project_group_usage_not_covered_gives_a_fallback_row(project_fields):
    quotas = storage_of([personal_entry(group=PROJECT, filesystem="exa_project1")], [project_entry(**project_fields)])
    assert quotas[-1] == StorageQuota(
        name=f"exa_project1 of {PROJECT} (your usage)", project=PROJECT, used_bytes=KIB, files_used=3
    )


def test_storage_order_is_personal_then_projects_then_fallback_rows():
    quotas = storage_of(
        [
            personal_entry(group="e-other", filesystem="exa_scratch"),
            personal_entry(filesystem="exa_scratch", **{"data-usage": "5KB"}),
            personal_entry(group=PROJECT),  # covered below: not repeated
            personal_entry(filesystem="exa_home"),
        ],
        [project_entry(filesystem="exa_home", **{"project-dir": "/e/a"})],
    )
    assert [(quota.name, quota.project) for quota in quotas] == [
        ("exa_scratch", None),
        (HOME, None),
        ("/e/a", PROJECT),
        ("exa_scratch of e-other (your usage)", "e-other"),
    ]


@pytest.mark.parametrize(
    "fields, name",
    [
        ({}, f"/e/project1/{PROJECT}"),
        ({"project-dir": ""}, "exa_project1"),
        ({"project-dir": None}, "exa_project1"),
        ({"project-dir": None, "filesystem": None}, "?"),
        ({"project-dir": None, "filesystem": ""}, "?"),
    ],
)
def test_project_quota_name_fallbacks(fields, name):
    quotas = storage_of([], [project_entry(**fields)])
    assert [quota.name for quota in quotas] == [name]
    assert quotas[0].project == PROJECT


def test_project_quota_without_project_field_has_no_project():
    quotas = storage_of([], [project_entry(project=None)])
    assert quotas[0].project is None


def test_project_entries_without_data_usage_are_skipped():
    quotas = storage_of([], [project_entry(**{"data-usage": None}), project_entry(**{"project-dir": "/e/kept"})])
    assert [quota.name for quota in quotas] == ["/e/kept"]


@pytest.mark.parametrize(
    "soft_limit, expected",
    [
        ("17179869184KB", 17179869184 * KIB),
        ("1KB", KIB),
        ("10", 10 * KIB),  # bare numbers are in KB
        ("0", 0),
        ("1.5MB", int(1.5 * KIB * KIB)),
        ("-", None),
        (None, None),
    ],
)
def test_project_quota_data_soft_limit(soft_limit, expected):
    quota = storage_of([], [project_entry(**{"data-soft-limit": soft_limit})])[0]
    assert quota.limit_bytes == expected


@pytest.mark.parametrize(
    "usage, expected",
    [("0", 0), ("198366876KB", 198366876 * KIB), ("7", 7 * KIB)],
)
def test_project_quota_data_usage(usage, expected):
    quota = storage_of([], [project_entry(**{"data-usage": usage})])[0]
    assert quota.used_bytes == expected


@pytest.mark.parametrize(
    "usage, limit, expected",
    [
        ("18652", "3000000", (18652, 3000000)),
        ("0", "0", (0, 0)),
        ("-", "-", (None, None)),
        (None, None, (None, None)),
        ("1,234", "100K", (1234, 100000)),
    ],
)
def test_project_quota_inodes(usage, limit, expected):
    quota = storage_of([], [project_entry(**{"inode-usage": usage, "inode-soft-limit": limit})])[0]
    assert (quota.files_used, quota.files_limit) == expected


def test_project_quota_with_invalid_size_raises_value_error():
    with pytest.raises(ValueError, match="invalid size"):
        storage_of([], [project_entry(**{"data-usage": "lots"})])


def test_project_quotas_from_several_concatenated_documents():
    results = {
        "userdata": ok("[]"),
        "projectdata": ok(
            json.dumps([project_entry(**{"project-dir": "/e/a"})])
            + "\n"
            + json.dumps([project_entry(**{"project-dir": "/e/b"}), project_entry(**{"project-dir": "/e/c"})])
        ),
        "home": ok(HOME),
    }
    quotas = make_cluster().parse_storage(results, TODAY)
    assert [quota.name for quota in quotas] == ["/e/a", "/e/b", "/e/c"]


def test_successful_projectdata_gives_no_warning():
    cluster = make_cluster()
    cluster.parse_storage(fixture_results(), TODAY)
    assert cluster._warnings == []


@pytest.mark.parametrize(
    "stderr, detail",
    [
        ("jutil: error for project a\n  jutil: no data quota for b  \n\n", "jutil: no data quota for b"),
        ("", "exit status 3"),
        ("\n   \n", "exit status 3"),
    ],
)
def test_failed_projectdata_is_still_parsed_with_a_warning(stderr, detail):
    cluster = make_cluster()
    results = fixture_results(
        projectdata=failed(3, stdout=read_fixture("jupiter/projectdata.json"), stderr=stderr)
    )
    quotas = cluster.parse_storage(results, TODAY)
    assert quotas == [FIXTURE_HOME_QUOTA] + FIXTURE_PROJECT_QUOTAS
    assert cluster._warnings == [f"project data quotas may be incomplete: {detail}"]


def test_failed_projectdata_without_output_falls_back_to_the_users_usage():
    cluster = make_cluster()
    quotas = cluster.parse_storage(fixture_results(projectdata=failed(1, stderr="jutil: not found")), TODAY)
    assert quotas == [FIXTURE_HOME_QUOTA, FIXTURE_FALLBACK_QUOTA]
    assert cluster._warnings == ["project data quotas may be incomplete: jutil: not found"]


@pytest.mark.parametrize(
    "returncode, stderr, detail",
    [(1, "first\nlast\n", "last"), (5, "", "exit status 5"), (0, "  only line  ", "only line")],
)
def test_output_detail_is_the_last_stderr_line(returncode, stderr, detail):
    results = {"x": CommandResult(stdout="ignored", stderr=stderr, returncode=returncode)}
    assert JupiterCluster.output_detail(results, "x") == detail


@pytest.mark.parametrize(
    "projectdata, detail",
    [
        (failed(124, stdout="[]"), "`projectdata` timed out (time limit of 180 s reached)"),
        # Partial output of a timed-out loop is not used.
        (
            failed(124, stdout=read_fixture("jupiter/projectdata.json")),
            "`projectdata` timed out (time limit of 180 s reached)",
        ),
        (failed(None, stdout="[]"), "`projectdata` did not complete: no exit status"),
        (
            CommandResult(stdout="[]", returncode=None, reason="the SSH session timed out after 195 s"),
            "`projectdata` did not complete: the SSH session timed out after 195 s",
        ),
    ],
    ids=["timeout-124", "timeout-with-output", "unfinished", "unfinished-with-reason"],
)
def test_timed_out_projectdata_is_a_warning(projectdata, detail):
    cluster = make_cluster()
    quotas = cluster.parse_storage(fixture_results(projectdata=projectdata), TODAY)
    assert quotas == [FIXTURE_HOME_QUOTA, FIXTURE_FALLBACK_QUOTA]
    assert cluster._warnings == [f"project data quotas unavailable: {detail}"]


def test_missing_projectdata_is_a_warning():
    results = fixture_results()
    del results["projectdata"]
    cluster = make_cluster()
    quotas = cluster.parse_storage(results, TODAY)
    assert quotas == [FIXTURE_HOME_QUOTA, FIXTURE_FALLBACK_QUOTA]
    assert cluster._warnings == ["project data quotas unavailable: no output from `projectdata`"]


@pytest.mark.parametrize(
    "userdata, message",
    [
        (failed(1, stderr="jutil: database unavailable"), "`userdata` failed (exit status 1): jutil: database unavailable"),
        (failed(124), "`userdata` timed out (time limit of 180 s reached)"),
        (failed(None), "`userdata` did not complete: no exit status"),
        (ok("not json"), "invalid JSON output from jutil: Expecting value: line 1 column 1 (char 0)"),
    ],
    ids=["failed", "timeout-124", "unfinished", "invalid-json"],
)
def test_unusable_userdata_is_an_error(userdata, message):
    with pytest.raises(CollectionError) as excinfo:
        make_cluster().parse_storage(fixture_results(userdata=userdata), TODAY)
    assert str(excinfo.value) == message


def test_storage_accepts_numeric_json_values():
    entry = personal_entry(**{"data-usage": 1024, "data-soft-limit": 0, "inode-usage": 5, "inode-soft-limit": 120000})
    quotas = storage_of([entry])
    assert quotas == [
        StorageQuota(name=HOME, project=None, used_bytes=1024 * KIB, limit_bytes=0, files_used=5, files_limit=120000)
    ]


def test_numeric_personal_group_is_not_repeated_as_a_fallback_row():
    quotas = storage_of([personal_entry(group=123, username=123)])
    assert quotas == [
        StorageQuota(name=HOME, project=None, used_bytes=KIB, limit_bytes=2 * KIB, files_used=3, files_limit=4)
    ]


def test_fallback_rows_accept_numeric_json_values():
    entry = personal_entry(group=PROJECT, **{"data-usage": 2048, "inode-usage": 7, "inode-soft-limit": 9})
    assert storage_of([entry]) == [
        StorageQuota(name=f"exa_home of {PROJECT} (your usage)", project=PROJECT, used_bytes=2048 * KIB, files_used=7)
    ]


# -- full collection -----------------------------------------------------------


def test_collect_everything_from_the_fixtures():
    runner = FakeRunner(fixture_results())
    cluster = JupiterCluster(make_config("jupiter"), runner=runner)

    report = cluster.collect(today=TODAY, now=NOW)

    assert runner.calls == [
        {
            "cpuquota": CPUQUOTA_COMMAND,
            "userdata": USERDATA_COMMAND,
            "projectdata": PROJECT_DATAQUOTA_COMMAND,
            "home": HOME_COMMAND,
            "jobs": SQUEUE_COMMAND,
        }
    ]
    # The fast commands run first.
    assert list(runner.calls[0]) == ["jobs", "cpuquota", "userdata", "projectdata", "home"]
    assert report.name == "jupiter"
    assert report.system == "JUPITER"
    assert report.collected_at == NOW
    assert report.sections == ["compute", "storage", "jobs"]
    assert report.budgets == [FIXTURE_REPORTED]
    assert report.storage == [FIXTURE_HOME_QUOTA] + FIXTURE_PROJECT_QUOTAS
    assert report.jobs is not None and len(report.jobs) == 5
    assert report.errors == []
    assert report.warnings == []
    assert report.failed_sections == []
    assert report.ok


def test_collect_compute_only_runs_only_cpuquota():
    runner = FakeRunner(fixture_results())
    report = JupiterCluster(make_config("jupiter"), runner=runner).collect(["compute"], today=TODAY, now=NOW)
    assert runner.calls == [{"cpuquota": CPUQUOTA_COMMAND}]
    assert report.sections == ["compute"]
    assert report.budgets == [FIXTURE_REPORTED]
    assert report.storage == []
    assert report.jobs is None


def test_collect_storage_only_runs_the_storage_commands():
    runner = FakeRunner(fixture_results())
    report = JupiterCluster(make_config("jupiter"), runner=runner).collect(["storage"], today=TODAY, now=NOW)
    assert runner.calls == [{"userdata": USERDATA_COMMAND, "projectdata": PROJECT_DATAQUOTA_COMMAND, "home": HOME_COMMAND}]
    assert list(runner.calls[0]) == ["userdata", "projectdata", "home"]
    assert report.storage == [FIXTURE_HOME_QUOTA] + FIXTURE_PROJECT_QUOTAS
    assert report.budgets == []


def test_collect_serialises_the_fixture_budget():
    report = make_cluster(fixture_results()).collect(["compute"], today=TODAY, now=NOW)
    data = report.to_dict()["budgets"][0]
    assert data["project"] == PROJECT
    assert data["account"] is None
    assert data["remaining"] == pytest.approx(801261.12)
    assert data["days_left"] == (dt.date(2027, 5, 31) - NOW.date()).days


def test_collect_filters_projects_but_keeps_personal_quotas():
    cluster = make_cluster(fixture_results(), projects=["e-other"])
    report = cluster.collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.budgets == []
    assert report.storage == [FIXTURE_HOME_QUOTA]
    assert report.warnings == ["project(s) not found on this cluster: e-other"]


def test_collect_selected_project_is_case_insensitive():
    cluster = make_cluster(fixture_results(), projects=["E-DEV-2026D00-001"])
    report = cluster.collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.budgets == [FIXTURE_REPORTED]
    assert report.storage == [FIXTURE_HOME_QUOTA] + FIXTURE_PROJECT_QUOTAS
    assert report.warnings == []


@pytest.mark.parametrize("selected", ["acct-b", "ACCT-B", "proj-a"])
def test_collect_keeps_a_budget_selected_by_project_or_by_account(selected):
    entry = {"project": "Proj-A", "budget-account": "acct-b", "acct-mode": "fixed"}
    cluster = make_cluster({"cpuquota": json.dumps([entry])}, projects=[selected])
    report = cluster.collect(["compute"], today=TODAY, now=NOW)
    assert [(budget.project, budget.account) for budget in report.budgets] == [("proj-a", "acct-b")]
    assert report.warnings == []


def test_collect_hides_fallback_rows_of_other_projects():
    results = fixture_results(projectdata=failed(1, stderr="jutil: denied"))
    report = make_cluster(results, projects=["e-other"]).collect(["storage"], today=TODAY, now=NOW)
    assert report.storage == [FIXTURE_HOME_QUOTA]
    assert report.warnings == [
        "project data quotas may be incomplete: jutil: denied",
        "project(s) not found on this cluster: e-other",
    ]


def test_collect_fallback_rows_make_their_project_known():
    results = fixture_results(projectdata=failed(1, stderr="jutil: denied"))
    report = make_cluster(results, projects=[PROJECT.upper()]).collect(["storage"], today=TODAY, now=NOW)
    assert report.storage == [FIXTURE_HOME_QUOTA, FIXTURE_FALLBACK_QUOTA]
    assert report.warnings == ["project data quotas may be incomplete: jutil: denied"]


def test_collect_does_not_report_unknown_projects_when_a_section_failed():
    results = fixture_results(cpuquota=failed(1, stderr="denied"))
    report = make_cluster(results, projects=["e-other"]).collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.failed_sections == ["compute"]
    assert report.warnings == []


def test_collect_keeps_the_reported_end_date_over_the_configured_one():
    cluster = make_cluster(fixture_results(), project_end={PROJECT: dt.date(2030, 1, 1)})
    report = cluster.collect(["compute"], today=TODAY, now=NOW)
    assert report.budgets[0].end == dt.date(2027, 5, 31)


def test_collect_uses_the_configured_end_date_when_jutil_reports_none():
    entry = {"budget-account": PROJECT, "acct-mode": "fixed"}
    cluster = make_cluster({"cpuquota": json.dumps([entry])}, project_end={PROJECT.upper(): dt.date(2030, 1, 1)})
    report = cluster.collect(["compute"], today=TODAY, now=NOW)
    assert report.budgets[0].end == dt.date(2030, 1, 1)


def test_collect_reports_the_projectdata_warning():
    results = fixture_results(
        projectdata=failed(2, stdout=read_fixture("jupiter/projectdata.json"), stderr="jutil: timeout for e-x\n")
    )
    report = make_cluster(results).collect(["storage"], today=TODAY, now=NOW)
    assert report.storage == [FIXTURE_HOME_QUOTA] + FIXTURE_PROJECT_QUOTAS
    assert report.errors == []
    assert report.failed_sections == []
    assert report.warnings == ["project data quotas may be incomplete: jutil: timeout for e-x"]


def test_collect_timed_out_projectdata_names_the_shell_loop_and_the_limit():
    results = fixture_results(projectdata=failed(124))
    report = make_cluster(results, timeout=42.5).collect(["storage"], today=TODAY, now=NOW)
    assert report.storage == [FIXTURE_HOME_QUOTA, FIXTURE_FALLBACK_QUOTA]
    assert report.errors == []
    assert report.failed_sections == []
    assert report.warnings == [
        f"project data quotas unavailable: `{PROJECT_DATAQUOTA_COMMAND}` timed out (time limit of 42.5 s reached)"
    ]


def test_collect_failed_projectdata_with_invalid_json_gives_warning_and_error():
    results = fixture_results(projectdata=failed(1, stdout="[{broken", stderr="jutil: crashed"))
    report = make_cluster(results).collect(["storage"], today=TODAY, now=NOW)
    assert report.storage == []
    assert report.warnings == ["project data quotas may be incomplete: jutil: crashed"]
    assert len(report.errors) == 1
    assert report.errors[0].startswith("storage: invalid JSON output from jutil: ")
    assert report.failed_sections == ["storage"]


def test_collect_errors_name_the_full_commands():
    results = fixture_results(cpuquota=failed(124), userdata=failed(1, stderr="denied"))
    report = make_cluster(results).collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.errors == [
        f"compute: `{CPUQUOTA_COMMAND}` timed out (time limit of 180 s reached)",
        f"storage: `{USERDATA_COMMAND}` failed (exit status 1): denied",
    ]
    assert report.failed_sections == ["compute", "storage"]
    assert report.budgets == [] and report.storage == []


def test_collect_reports_unexpected_values_without_raising():
    results = fixture_results(cpuquota=json.dumps([{"acct-period-end": "end of February"}]))
    report = make_cluster(results).collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.errors == ["compute: unexpected output (ValueError: invalid date: 'end of February')"]
    assert report.failed_sections == ["compute"]
    assert report.storage == [FIXTURE_HOME_QUOTA] + FIXTURE_PROJECT_QUOTAS


def test_collect_missing_projectdata_names_the_shell_loop():
    results = fixture_results()
    del results["projectdata"]
    report = make_cluster(results).collect(["storage"], today=TODAY, now=NOW)
    assert report.errors == []
    assert report.warnings == [f"project data quotas unavailable: no output from `{PROJECT_DATAQUOTA_COMMAND}`"]
    assert report.storage == [FIXTURE_HOME_QUOTA, FIXTURE_FALLBACK_QUOTA]


def test_collect_remote_error_fails_every_section():
    runner = FakeRunner(error=RemoteError("SSH connection failed: Connection refused"))
    report = JupiterCluster(make_config("jupiter", projects=["e-other"]), runner=runner).collect(
        today=TODAY, now=NOW
    )
    assert report.errors == ["SSH connection failed: Connection refused"]
    assert report.failed_sections == ["compute", "storage", "jobs"]
    assert report.warnings == []
    assert report.budgets == [] and report.storage == [] and report.jobs is None
