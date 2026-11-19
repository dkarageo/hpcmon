"""Tests for :mod:`hpcmon.models`: derived properties and JSON representations."""

from __future__ import annotations

import datetime as dt
import json

import pytest

from hpcmon.models import PENDING_STATES, RUNNING_STATES, ClusterReport, ComputeBudget, Job, StorageQuota
from tests.conftest import NOW, TODAY


def budget(**kwargs) -> ComputeBudget:
    """A GPU budget of a test project with the given amounts."""
    return ComputeBudget(project="project_465000123", resource="GPU", unit="GPU-h", **kwargs)


# --------------------------------------------------------------------------- #
# ComputeBudget
# --------------------------------------------------------------------------- #


def test_compute_budget_defaults_are_all_unknown():
    b = budget()
    assert (b.total, b.used, b.available) == (None, None, None)
    assert (b.month_total, b.month_used) == (None, None)
    assert (b.start, b.end, b.note, b.account) == (None, None, None, None)
    assert b.remaining is None
    assert b.remaining_fraction is None
    assert b.month_remaining is None
    assert b.month_remaining_fraction is None
    assert b.days_left(TODAY) is None
    assert b.has_ended(TODAY) is False


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"total": 1000.0, "used": 250.0}, 750.0),
        ({"total": 1000.0, "used": 1000.0}, 0.0),
        ({"total": 100.0, "used": 150.0}, -50.0),
        ({"total": 0.0, "used": 0.0}, 0.0),
        # A reported "available" amount wins over total - used ...
        ({"total": 1000.0, "used": 250.0, "available": 300.0}, 300.0),
        # ... even when it is zero ...
        ({"total": 1000.0, "used": 250.0, "available": 0.0}, 0.0),
        # ... and when total/used are unknown.
        ({"available": 42.0}, 42.0),
        ({"total": 1000.0, "available": 900.0}, 900.0),
        # Without "available", both total and used are needed.
        ({"total": 1000.0}, None),
        ({"used": 250.0}, None),
        ({}, None),
    ],
)
def test_compute_budget_remaining(kwargs, expected):
    assert budget(**kwargs).remaining == expected


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"total": 1000.0, "used": 250.0}, 0.75),
        ({"total": 1000.0, "used": 1000.0}, 0.0),
        ({"total": 100.0, "used": 150.0}, -0.5),
        ({"total": 1000.0, "available": 300.0}, 0.3),
        ({"total": 1000.0, "used": 250.0, "available": 0.0}, 0.0),
        # Unknown or zero totals have no meaningful fraction.
        ({"available": 42.0}, None),
        ({"total": 0.0, "used": 0.0}, None),
        ({"total": 0.0, "available": 5.0}, None),
        # Unknown remaining amount.
        ({"total": 1000.0}, None),
        ({}, None),
    ],
)
def test_compute_budget_remaining_fraction(kwargs, expected):
    value = budget(**kwargs).remaining_fraction
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"month_total": 100.0, "month_used": 30.0}, 70.0),
        ({"month_total": 100.0, "month_used": 0.0}, 100.0),
        ({"month_total": 100.0, "month_used": 130.0}, -30.0),
        ({"month_total": 0.0, "month_used": 0.0}, 0.0),
        ({"month_total": 100.0}, None),
        ({"month_used": 30.0}, None),
        # The overall "available" amount does not leak into the monthly figures.
        ({"available": 500.0, "total": 1000.0, "used": 500.0}, None),
    ],
)
def test_compute_budget_month_remaining(kwargs, expected):
    assert budget(**kwargs).month_remaining == expected


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"month_total": 100.0, "month_used": 30.0}, 0.7),
        ({"month_total": 100.0, "month_used": 100.0}, 0.0),
        ({"month_total": 200.0, "month_used": 250.0}, -0.25),
        ({"month_total": 0.0, "month_used": 0.0}, None),
        ({"month_total": 100.0}, None),
        ({"month_used": 30.0}, None),
        ({}, None),
    ],
)
def test_compute_budget_month_remaining_fraction(kwargs, expected):
    value = budget(**kwargs).month_remaining_fraction
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)


@pytest.mark.parametrize(
    "end, expected",
    [
        (None, None),
        (dt.date(2026, 11, 18), 0),
        (dt.date(2026, 11, 19), 1),
        (dt.date(2026, 12, 31), 43),
        (dt.date(2027, 5, 19), 182),
        (dt.date(2026, 11, 17), -1),
        (dt.date(2025, 11, 18), -365),
    ],
)
def test_compute_budget_days_left(end, expected):
    assert budget(end=end).days_left(TODAY) == expected


def test_compute_budget_days_left_does_not_depend_on_start():
    b = budget(start=dt.date(2026, 1, 1), end=dt.date(2026, 12, 1))
    assert b.days_left(TODAY) == 13
    assert b.days_left(dt.date(2026, 12, 1)) == 0


def test_compute_budget_days_left_round_trips_lumi_days_left():
    # LUMI reports "N day(s) left"; the cluster code stores today + N as end.
    b = budget(end=TODAY + dt.timedelta(days=182))
    assert b.days_left(TODAY) == 182


@pytest.mark.parametrize(
    "end, expected",
    [
        (None, False),
        (dt.date(2027, 5, 19), False),
        (dt.date(2026, 11, 19), False),
        # The last day itself still belongs to the period.
        (dt.date(2026, 11, 18), False),
        (dt.date(2026, 11, 17), True),
        (dt.date(2020, 1, 1), True),
    ],
)
def test_compute_budget_has_ended(end, expected):
    assert budget(end=end).has_ended(TODAY) is expected


def test_compute_budget_has_ended_depends_on_the_given_day():
    b = budget(start=dt.date(2026, 1, 1), end=dt.date(2026, 12, 31))
    assert b.has_ended(dt.date(2026, 12, 31)) is False
    assert b.has_ended(dt.date(2027, 1, 1)) is True
    # The start of the period is irrelevant.
    assert b.has_ended(dt.date(2025, 6, 1)) is False


def test_compute_budget_account_is_stored_separately_from_the_project():
    b = ComputeBudget(project="exampleproj", resource="CPU", unit="core-h", account="examplebudget")
    assert b.project == "exampleproj"
    assert b.account == "examplebudget"
    assert b.to_dict()["account"] == "examplebudget"
    assert b.to_dict()["project"] == "exampleproj"


def test_compute_budget_to_dict_with_all_fields():
    b = ComputeBudget(
        project="IscrB_EXAMPLE",
        resource="Booster",
        unit="core-h",
        total=1000.0,
        used=250.5,
        month_total=100.0,
        month_used=40.0,
        start=dt.date(2026, 5, 1),
        end=dt.date(2027, 4, 30),
        note="monthly split",
        account="IscrB_ACCOUNT",
        gpu_hours_per_unit=0.125,
    )
    expected = {
        "project": "IscrB_EXAMPLE",
        "account": "IscrB_ACCOUNT",
        "resource": "Booster",
        "unit": "core-h",
        "total": 1000.0,
        "used": 250.5,
        "remaining": 749.5,
        "remaining_fraction": 0.7495,
        "month_total": 100.0,
        "month_used": 40.0,
        "month_remaining": 60.0,
        "month_remaining_fraction": 0.6,
        "start": "2026-05-01",
        "end": "2027-04-30",
        "days_left": None,
        "note": "monthly split",
        "gpu_hours_per_unit": 0.125,
        "gpu_hours_total": 125.0,
        "gpu_hours_remaining": 93.6875,
        "month_gpu_hours_total": 12.5,
        "month_gpu_hours_remaining": 7.5,
    }
    assert b.to_dict() == expected
    assert b.to_dict(None) == expected
    # With a reference day, the days left until the end are included.
    assert b.to_dict(TODAY) == dict(expected, days_left=163)
    assert list(b.to_dict(TODAY)) == list(expected)


@pytest.mark.parametrize(
    "end, today, expected",
    [
        (dt.date(2027, 5, 19), TODAY, 182),
        (dt.date(2026, 11, 18), TODAY, 0),
        (dt.date(2026, 11, 10), TODAY, -8),
        (None, TODAY, None),
        (dt.date(2027, 5, 19), None, None),
    ],
)
def test_compute_budget_to_dict_days_left(end, today, expected):
    assert budget(end=end).to_dict(today)["days_left"] == expected


def test_compute_budget_to_dict_reports_available_as_remaining():
    data = budget(total=1000.0, used=250.0, available=300.0).to_dict()
    assert "available" not in data
    assert data["remaining"] == 300.0
    assert data["remaining_fraction"] == 0.3


def test_compute_budget_to_dict_with_unknown_values():
    expected = {
        "project": "project_465000123",
        "account": None,
        "resource": "GPU",
        "unit": "GPU-h",
        "total": None,
        "used": None,
        "remaining": None,
        "remaining_fraction": None,
        "month_total": None,
        "month_used": None,
        "month_remaining": None,
        "month_remaining_fraction": None,
        "start": None,
        "end": None,
        "days_left": None,
        "note": None,
        "gpu_hours_per_unit": None,
        "gpu_hours_total": None,
        "gpu_hours_remaining": None,
        "month_gpu_hours_total": None,
        "month_gpu_hours_remaining": None,
    }
    assert budget().to_dict() == expected
    assert budget().to_dict(TODAY) == expected


def test_compute_budget_to_dict_is_json_serialisable():
    b = budget(total=10.0, used=4.0, month_total=2.0, month_used=1.0, start=dt.date(2026, 1, 1),
               end=dt.date(2026, 12, 31), account="acc")
    for data in (b.to_dict(), b.to_dict(TODAY)):
        assert json.loads(json.dumps(data)) == data
    assert b.to_dict(TODAY)["days_left"] == 43


# --------------------------------------------------------------------------- #
# StorageQuota
# --------------------------------------------------------------------------- #


def test_storage_quota_defaults():
    q = StorageQuota(name="/home/users/u100001")
    assert q.project is None
    assert (q.used_bytes, q.limit_bytes, q.files_used, q.files_limit, q.note) == (None, None, None, None, None)
    assert q.free_bytes is None
    assert q.used_fraction is None
    assert q.files_fraction is None


@pytest.mark.parametrize(
    "used, limit, expected",
    [
        (30, 100, 70),
        (0, 100, 100),
        (100, 100, 0),
        (150, 100, -50),
        (10, 0, -10),
        (0, 0, 0),
        (None, 100, None),
        (30, None, None),
        (None, None, None),
    ],
)
def test_storage_quota_free_bytes(used, limit, expected):
    assert StorageQuota(name="x", used_bytes=used, limit_bytes=limit).free_bytes == expected


@pytest.mark.parametrize(
    "used, limit, expected",
    [
        (25, 100, 0.25),
        (0, 100, 0.0),
        (100, 100, 1.0),
        (150, 100, 1.5),
        (10, 0, None),
        (0, 0, None),
        (None, 100, None),
        (30, None, None),
    ],
)
def test_storage_quota_used_fraction(used, limit, expected):
    value = StorageQuota(name="x", used_bytes=used, limit_bytes=limit).used_fraction
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)


@pytest.mark.parametrize(
    "used, limit, expected",
    [
        (4512, 150000, 0.03008),
        (187342, 2000000, 0.093671),
        (0, 120000, 0.0),
        (120000, 120000, 1.0),
        (5, 0, None),
        (None, 120000, None),
        (4128, None, None),
    ],
)
def test_storage_quota_files_fraction(used, limit, expected):
    value = StorageQuota(name="x", files_used=used, files_limit=limit).files_fraction
    if expected is None:
        assert value is None
    else:
        assert value == pytest.approx(expected)


def test_storage_quota_to_dict_with_all_fields():
    q = StorageQuota(
        name="/project/home/p200001",
        project="p200001",
        used_bytes=213 * 1024**3,
        limit_bytes=5000 * 1024**3,
        files_used=187342,
        files_limit=2000000,
        note="project home",
    )
    assert q.to_dict() == {
        "name": "/project/home/p200001",
        "project": "p200001",
        "used_bytes": 228707008512,
        "limit_bytes": 5368709120000,
        "free_bytes": 5140002111488,
        "used_fraction": 0.0426,
        "files_used": 187342,
        "files_limit": 2000000,
        "files_fraction": 0.093671,
        "note": "project home",
    }


def test_storage_quota_to_dict_with_unknown_values():
    data = StorageQuota(name="/scratch/project_465000123").to_dict()
    assert data == {
        "name": "/scratch/project_465000123",
        "project": None,
        "used_bytes": None,
        "limit_bytes": None,
        "free_bytes": None,
        "used_fraction": None,
        "files_used": None,
        "files_limit": None,
        "files_fraction": None,
        "note": None,
    }
    assert json.loads(json.dumps(data)) == data


@pytest.mark.parametrize(
    "used, limit, expected",
    [(4512, 150000, 0.03008), (0, 120000, 0.0), (5, 0, None), (None, 120000, None), (4128, None, None)],
)
def test_storage_quota_to_dict_includes_files_fraction(used, limit, expected):
    data = StorageQuota(name="x", files_used=used, files_limit=limit).to_dict()
    assert data["files_fraction"] == expected
    assert (data["files_used"], data["files_limit"]) == (used, limit)


# --------------------------------------------------------------------------- #
# Job
# --------------------------------------------------------------------------- #


def test_job_defaults():
    job = Job(job_id="1234567")
    assert job.to_dict() == {
        "job_id": "1234567",
        "name": "",
        "partition": "",
        "account": "",
        "state": "",
        "elapsed": "",
        "time_limit": "",
        "nodes": None,
        "reason": "",
    }
    assert job.is_running is False
    assert job.is_pending is False


@pytest.mark.parametrize(
    "state",
    [
        "RUNNING",
        "COMPLETING",
        "CONFIGURING",
        "STAGE_OUT",
        "SIGNALING",
        "R",
        "CG",
        "CF",
        "SO",
        "running",
        "Running",
        "r",
        "cg",
        "configuring",
        "stage_out",
    ],
)
def test_job_running_states(state):
    job = Job(job_id="1", state=state)
    assert job.is_running is True
    assert job.is_pending is False


@pytest.mark.parametrize(
    "state",
    [
        "PENDING",
        "REQUEUED",
        "REQUEUE_HOLD",
        "REQUEUE_FED",
        "RESV_DEL_HOLD",
        "PD",
        "RQ",
        "RH",
        "pending",
        "Pending",
        "pd",
        "rq",
        "requeue_hold",
    ],
)
def test_job_pending_states(state):
    job = Job(job_id="1", state=state)
    assert job.is_pending is True
    assert job.is_running is False


@pytest.mark.parametrize(
    "state",
    [
        "",
        "SUSPENDED",
        "S",
        "COMPLETED",
        "CD",
        "CANCELLED",
        "CA",
        "FAILED",
        "F",
        "TIMEOUT",
        "TO",
        "PREEMPTED",
        "PR",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "BOOT_FAIL",
        "STOPPED",
        "RUN",
        "PEND",
    ],
)
def test_job_other_states_are_neither_running_nor_pending(state):
    job = Job(job_id="1", state=state)
    assert job.is_running is False
    assert job.is_pending is False


def test_job_short_code_of_signaling_is_running():
    assert Job(job_id="1", state="SI").is_running is True


@pytest.mark.parametrize("state", ["RF", "RD"])
def test_job_short_codes_of_requeue_fed_and_resv_del_hold_are_pending(state):
    assert Job(job_id="1", state=state).is_pending is True


def test_state_sets_contain_the_documented_codes():
    assert {"SI", "SIGNALING"} <= RUNNING_STATES
    assert {"RF", "RD", "REQUEUE_FED", "RESV_DEL_HOLD"} <= PENDING_STATES
    assert RUNNING_STATES.isdisjoint(PENDING_STATES)


@pytest.mark.parametrize("state", [" R ", "\tRUNNING\n", "si ", "  cg", "SIGNALING\r"])
def test_job_running_state_ignores_surrounding_whitespace(state):
    job = Job(job_id="1", state=state)
    assert job.is_running is True
    assert job.is_pending is False


@pytest.mark.parametrize("state", [" PD ", "pending\n", "\tRF", "rd  ", " REQUEUED "])
def test_job_pending_state_ignores_surrounding_whitespace(state):
    job = Job(job_id="1", state=state)
    assert job.is_pending is True
    assert job.is_running is False


@pytest.mark.parametrize("state", ["   ", "R D", "P D", " S "])
def test_job_whitespace_only_or_inner_spaces_are_neither(state):
    job = Job(job_id="1", state=state)
    assert job.is_running is False
    assert job.is_pending is False


def test_job_to_dict_with_all_fields():
    job = Job(
        job_id="1234567",
        name="train|model [v2]",
        partition="standard-g",
        account="project_465000123",
        state="RUNNING",
        elapsed="1:02:03",
        time_limit="2-00:00:00",
        nodes=4,
        reason="nid[005000-005003]",
    )
    data = job.to_dict()
    assert data == {
        "job_id": "1234567",
        "name": "train|model [v2]",
        "partition": "standard-g",
        "account": "project_465000123",
        "state": "RUNNING",
        "elapsed": "1:02:03",
        "time_limit": "2-00:00:00",
        "nodes": 4,
        "reason": "nid[005000-005003]",
    }
    assert json.loads(json.dumps(data)) == data


# --------------------------------------------------------------------------- #
# ClusterReport
# --------------------------------------------------------------------------- #


def squeue_jobs():
    """The jobs of ``tests/fixtures/slurm/squeue.txt``, in file order."""
    return [
        Job("1234567", "train|model [v2]", "standard-g", "project_465000123", "RUNNING", "1:02:03", "2-00:00:00", 4,
            "nid[005000-005003]"),
        Job("1234570", "cleanup", "dev-g", "project_465000123", "COMPLETING", "10:00", "30:00", 1, "nid005100"),
        Job("1234568", "eval", "small-g", "project_465000123", "PENDING", "0:00", "1:00:00", 1, "(Priority)"),
        Job("1234569_[1-10]", "sweep", "small-g", "project_465000999", "PENDING", "0:00", "30:00", 1,
            "(AssocGrpGRESMinutes)"),
        Job("1234571", "suspended-job", "small", "project_465000123", "SUSPENDED", "5:00", "30:00", 1, "nid001001"),
    ]


def test_cluster_report_defaults():
    report = ClusterReport(name="lumi", system="LUMI")
    assert report.budgets == []
    assert report.storage == []
    assert report.jobs is None
    assert report.errors == []
    assert report.warnings == []
    assert report.sections == []
    assert report.failed_sections == []
    assert report.collected_at is None
    assert report.ok is True
    assert report.running_jobs == []
    assert report.pending_jobs == []


def test_cluster_report_default_lists_are_not_shared():
    first = ClusterReport(name="a", system="A")
    second = ClusterReport(name="b", system="B")
    first.budgets.append(budget())
    first.storage.append(StorageQuota(name="x"))
    first.errors.append("boom")
    first.warnings.append("hmm")
    first.sections.append("compute")
    first.failed_sections.append("compute")
    assert second.budgets == []
    assert second.storage == []
    assert second.errors == []
    assert second.warnings == []
    assert second.sections == []
    assert second.failed_sections == []


def test_cluster_report_ok_depends_on_errors_only():
    assert ClusterReport(name="lumi", system="LUMI", warnings=["slow"]).ok is True
    assert ClusterReport(name="lumi", system="LUMI", errors=["ssh: connection refused"]).ok is False
    assert ClusterReport(name="lumi", system="LUMI", errors=["a"], warnings=["b"]).ok is False


def test_cluster_report_running_and_pending_jobs_keep_order():
    jobs = squeue_jobs()
    report = ClusterReport(name="lumi", system="LUMI", jobs=jobs)
    assert report.running_jobs == [jobs[0], jobs[1]]
    assert [job.job_id for job in report.running_jobs] == ["1234567", "1234570"]
    assert [job.job_id for job in report.pending_jobs] == ["1234568", "1234569_[1-10]"]
    # The SUSPENDED job is neither.
    assert jobs[4] not in report.running_jobs + report.pending_jobs
    assert report.running_jobs[0] is jobs[0]


def test_cluster_report_job_lists_with_no_jobs():
    report = ClusterReport(name="lumi", system="LUMI", jobs=[])
    assert report.running_jobs == []
    assert report.pending_jobs == []


def test_cluster_report_job_lists_without_matching_jobs():
    report = ClusterReport(name="lumi", system="LUMI", jobs=[Job("1", state="COMPLETED"), Job("2", state="FAILED")])
    assert report.running_jobs == []
    assert report.pending_jobs == []


def test_cluster_report_to_dict_full():
    b = budget(total=1000.0, used=250.0, end=dt.date(2027, 5, 19), gpu_hours_per_unit=1.0)
    q = StorageQuota(name="/project/project_465000123", project="project_465000123", used_bytes=4096,
                     limit_bytes=53687091200, files_used=1, files_limit=100000)
    jobs = squeue_jobs()[:1]
    report = ClusterReport(
        name="lumi",
        system="LUMI",
        gpus="4× AMD MI250X 128 GB (2 GCDs each)",
        budgets=[b],
        storage=[q],
        jobs=jobs,
        errors=["projectinfo failed"],
        warnings=["cached data"],
        failed_sections=["storage"],
        sections=["compute", "storage", "jobs"],
        collected_at=NOW,
    )
    assert report.to_dict() == {
        "name": "lumi",
        "system": "LUMI",
        "gpus": "4× AMD MI250X 128 GB (2 GCDs each)",
        "collected_at": "2026-11-18T20:30:00",
        "sections": ["compute", "storage", "jobs"],
        "failed_sections": ["storage"],
        "budgets": [
            {
                "project": "project_465000123",
                "account": None,
                "resource": "GPU",
                "unit": "GPU-h",
                "total": 1000.0,
                "used": 250.0,
                "remaining": 750.0,
                "remaining_fraction": 0.75,
                "month_total": None,
                "month_used": None,
                "month_remaining": None,
                "month_remaining_fraction": None,
                "start": None,
                "end": "2027-05-19",
                # Counted from the day of the collection.
                "days_left": 182,
                "note": None,
                "gpu_hours_per_unit": 1.0,
                "gpu_hours_total": 1000.0,
                "gpu_hours_remaining": 750.0,
                "month_gpu_hours_total": None,
                "month_gpu_hours_remaining": None,
            }
        ],
        "storage": [
            {
                "name": "/project/project_465000123",
                "project": "project_465000123",
                "used_bytes": 4096,
                "limit_bytes": 53687091200,
                "free_bytes": 53687087104,
                "used_fraction": 7.62939453125e-08,
                "files_used": 1,
                "files_limit": 100000,
                "files_fraction": 1e-05,
                "note": None,
            }
        ],
        "jobs": [
            {
                "job_id": "1234567",
                "name": "train|model [v2]",
                "partition": "standard-g",
                "account": "project_465000123",
                "state": "RUNNING",
                "elapsed": "1:02:03",
                "time_limit": "2-00:00:00",
                "nodes": 4,
                "reason": "nid[005000-005003]",
            }
        ],
        "errors": ["projectinfo failed"],
        "warnings": ["cached data"],
    }


def test_cluster_report_to_dict_minimal():
    data = ClusterReport(name="jupiter", system="JUPITER").to_dict()
    assert data == {
        "name": "jupiter",
        "system": "JUPITER",
        "gpus": None,
        "collected_at": None,
        "sections": [],
        "failed_sections": [],
        "budgets": [],
        "storage": [],
        "jobs": None,
        "errors": [],
        "warnings": [],
    }
    assert json.loads(json.dumps(data)) == data


def test_cluster_report_to_dict_without_collection_time_has_no_days_left():
    report = ClusterReport(name="x", system="X", budgets=[budget(end=dt.date(2027, 5, 19))])
    [data] = report.to_dict()["budgets"]
    assert data["end"] == "2027-05-19"
    assert data["days_left"] is None


def test_cluster_report_to_dict_days_left_uses_the_date_of_collected_at():
    stamp = dt.datetime(2026, 11, 19, 0, 5, tzinfo=dt.timezone(dt.timedelta(hours=-3)))
    report = ClusterReport(name="x", system="X", collected_at=stamp,
                           budgets=[budget(end=dt.date(2026, 11, 20)), budget(end=dt.date(2026, 11, 18))])
    assert [data["days_left"] for data in report.to_dict()["budgets"]] == [1, -1]


def test_cluster_report_to_dict_drops_microseconds_of_collected_at():
    stamp = dt.datetime(2026, 11, 18, 20, 30, 15, 987654, tzinfo=dt.timezone(dt.timedelta(hours=1)))
    data = ClusterReport(name="x", system="X", collected_at=stamp).to_dict()
    assert data["collected_at"] == "2026-11-18T20:30:15+01:00"
    naive = dt.datetime(2026, 11, 18, 20, 30, 15, 1)
    assert ClusterReport(name="x", system="X", collected_at=naive).to_dict()["collected_at"] == "2026-11-18T20:30:15"


def test_cluster_report_to_dict_distinguishes_no_jobs_from_not_collected():
    assert ClusterReport(name="x", system="X", jobs=[]).to_dict()["jobs"] == []
    assert ClusterReport(name="x", system="X", jobs=None).to_dict()["jobs"] is None


def test_cluster_report_to_dict_keeps_timezone_of_collected_at():
    stamp = dt.datetime(2026, 11, 18, 20, 30, tzinfo=dt.timezone.utc)
    data = ClusterReport(name="x", system="X", collected_at=stamp).to_dict()
    assert data["collected_at"] == "2026-11-18T20:30:00+00:00"


def test_cluster_report_to_dict_midnight_timestamp_is_not_dropped():
    stamp = dt.datetime(2026, 11, 18, 0, 0, 0)
    data = ClusterReport(name="x", system="X", collected_at=stamp).to_dict()
    assert data["collected_at"] == "2026-11-18T00:00:00"


def test_cluster_report_to_dict_returns_copies_of_lists():
    report = ClusterReport(name="x", system="X", jobs=squeue_jobs(), errors=["e"], warnings=["w"],
                           sections=["jobs"], failed_sections=["jobs"])
    data = report.to_dict()
    data["errors"].append("new")
    data["warnings"].append("new")
    data["sections"].append("new")
    data["failed_sections"].append("new")
    data["jobs"].clear()
    assert report.errors == ["e"]
    assert report.warnings == ["w"]
    assert report.sections == ["jobs"]
    assert report.failed_sections == ["jobs"]
    assert len(report.jobs) == 5


def test_cluster_report_to_dict_full_is_json_serialisable():
    report = ClusterReport(
        name="lumi",
        system="LUMI",
        budgets=[budget(total=1.0, used=0.5, start=dt.date(2026, 1, 1), end=dt.date(2027, 1, 1), account="a")],
        storage=[StorageQuota(name="/home", used_bytes=1, limit_bytes=2, files_used=1, files_limit=4)],
        jobs=squeue_jobs(),
        sections=["compute", "storage", "jobs"],
        failed_sections=["jobs"],
        collected_at=dt.datetime(2026, 11, 18, 20, 30, 0, 5, tzinfo=dt.timezone.utc),
    )
    data = report.to_dict()
    assert json.loads(json.dumps(data)) == data
    assert len(data["jobs"]) == 5
