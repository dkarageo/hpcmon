"""Tests of the Leonardo interface (:mod:`hpcmon.clusters.leonardo`).

The parsers are checked against real (anonymised) outputs of ``saldo -b``,
``saldo -b --dcgp`` and ``cindata`` and against inline variants of these
formats.  The whole collection is exercised through
:class:`tests.conftest.FakeRunner`, never through SSH.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Dict, List, Union

import pytest

from hpcmon.clusters import create_cluster
from hpcmon.clusters.base import CollectionError
from hpcmon.clusters.leonardo import LeonardoCluster, describe_freshness, parse_cindata, parse_saldo
from hpcmon.clusters.slurm import SQUEUE_COMMAND
from hpcmon.models import ComputeBudget, StorageQuota
from hpcmon.runner import CommandResult
from tests.conftest import NOW, TODAY, FakeRunner, failed, make_config, ok, read_fixture

KiB = 1024
GiB = 1024**3
TiB = 1024**4

ACCOUNT = "EUHPC_D00_001"
#: Note of every area of the cindata fixture (FRESH = "3hou").
FRESH_NOTE = "as of 3 hours ago"

EXPECTED_COMMANDS = {
    "saldo": "saldo -b",
    "saldo_dcgp": "saldo -b --dcgp",
    "cindata": "cindata",
    "jobs": SQUEUE_COMMAND,
}

BOOSTER_BUDGET = ComputeBudget(
    project=ACCOUNT,
    resource="Booster",
    unit="core-h",
    total=96000.0,
    used=3187.0,
    month_total=8250.0,
    month_used=0.0,
    start=dt.date(2026, 7, 1),
    end=dt.date(2027, 6, 30),
)

FIXTURE_QUOTAS = [
    StorageQuota(
        name="/leonardo_work/EUHPC_D00_001", project=ACCOUNT, used_bytes=87 * GiB, limit_bytes=2 * TiB, note=FRESH_NOTE
    ),
    StorageQuota(
        name="/leonardo_scratch/fast/EUHPC_D00_001",
        project=ACCOUNT,
        used_bytes=4 * KiB,
        limit_bytes=2 * TiB,
        note=FRESH_NOTE,
    ),
    StorageQuota(
        name="/leonardo_scratch/large/userexternal/jdoe5678", project=None, used_bytes=4 * KiB, note=FRESH_NOTE
    ),
    StorageQuota(
        name="/leonardo/pub/userexternal/jdoe5678",
        project=None,
        used_bytes=4 * KiB,
        limit_bytes=60 * GiB,
        note=FRESH_NOTE,
    ),
    StorageQuota(
        name="/leonardo/home/userexternal/jdoe5678",
        project=None,
        used_bytes=round(3.4 * GiB),
        limit_bytes=60 * GiB,
        note=FRESH_NOTE,
    ),
]

SALDO_HEADER = """\
-----------------------------------------------------------------------------------------------------------------------------------------
account                start         end         total        localCluster   totConsumed     totConsumed     monthTotal     monthConsumed
                                             (local h)   Consumed(local h)     (local h)               %      (local h)         (local h)
-----------------------------------------------------------------------------------------------------------------------------------------
"""

#: Two accounts; the local consumption differs from the total consumption.
SALDO_TWO_ACCOUNTS = SALDO_HEADER + (
    "EUHPC_D00_001       20260701    20270630        96000                1200          3187             3.3"
    "         8250                150\n"
    "EUHPC_A01_002       20260101    20261231     1,500,000            250000.5      300000.5            20.0"
    "        125000           12000.25    extra columns are ignored\n"
)

SALDO_DCGP = SALDO_HEADER + (
    "EUHPC_D00_001       20261001    20270331         50000                 100           120             0.2"
    "          8333                 20\n"
)

#: The Booster budget as reported by the cluster: with its GPU-hours rate
#: (8 core-hours per A100-hour).
BOOSTER_REPORTED = dataclasses.replace(BOOSTER_BUDGET, gpu_hours_per_unit=1 / 8)

DCGP_BUDGET = ComputeBudget(
    project=ACCOUNT,
    resource="DCGP",
    unit="core-h",
    total=50000.0,
    used=120.0,
    month_total=8333.0,
    month_used=20.0,
    start=dt.date(2026, 10, 1),
    end=dt.date(2027, 3, 31),
)

CINDATA_HEADER = (
    "USER       AREADESCR                                     AREAID                                   FRESH"
    "     USED      QTA    USED%    aUSED     aQTA   aUSED%\n"
)


def fixture_results(**overrides: Union[str, CommandResult, None]) -> Dict[str, Union[str, CommandResult]]:
    """The fixture outputs of all Leonardo commands; ``None`` removes a command."""
    results: Dict[str, Union[str, CommandResult, None]] = {
        "saldo": read_fixture("leonardo/saldo.txt"),
        "saldo_dcgp": read_fixture("leonardo/saldo_dcgp.txt"),
        "cindata": read_fixture("leonardo/cindata.txt"),
        "jobs": read_fixture("slurm/squeue.txt"),
    }
    results.update(overrides)
    return {key: value for key, value in results.items() if value is not None}


def make_leonardo(results=None, **config) -> LeonardoCluster:
    return LeonardoCluster(make_config("leonardo", **config), runner=FakeRunner(results))


def collecting_leonardo(results=None, **config):
    """A Leonardo cluster whose :meth:`warn` messages go to the returned list."""
    cluster = make_leonardo(results, **config)
    warnings: List[str] = []
    cluster.warn = warnings.append  # type: ignore[method-assign]
    return cluster, warnings


# -- cluster definition -------------------------------------------------------


def test_cluster_identity_and_registry():
    runner = FakeRunner()
    cluster = create_cluster(make_config("leo", type="leonardo"), runner=runner)

    assert isinstance(cluster, LeonardoCluster)
    assert cluster.runner is runner
    assert cluster.name == "leo"
    assert LeonardoCluster.type_name == "leonardo"
    assert LeonardoCluster.display_name == "Leonardo"


def test_commands_of_every_section():
    cluster = make_leonardo()

    assert cluster.compute_commands() == {"saldo": "saldo -b", "saldo_dcgp": "saldo -b --dcgp"}
    assert cluster.storage_commands() == {"cindata": "cindata"}
    assert cluster.commands() == EXPECTED_COMMANDS
    # The commands run in the order jobs -> compute -> storage.
    assert list(cluster.commands()) == ["jobs", "saldo", "saldo_dcgp", "cindata"]


# -- saldo --------------------------------------------------------------------


def test_parse_saldo_real_output():
    assert parse_saldo(read_fixture("leonardo/saldo.txt"), resource="Booster") == [BOOSTER_BUDGET]


def test_parse_saldo_uses_total_consumption_and_first_nine_columns():
    assert parse_saldo(SALDO_TWO_ACCOUNTS, resource="DCGP") == [
        ComputeBudget(
            project="EUHPC_D00_001",
            resource="DCGP",
            unit="core-h",
            total=96000.0,
            used=3187.0,
            month_total=8250.0,
            month_used=150.0,
            start=dt.date(2026, 7, 1),
            end=dt.date(2027, 6, 30),
        ),
        ComputeBudget(
            project="EUHPC_A01_002",
            resource="DCGP",
            unit="core-h",
            total=1_500_000.0,
            used=300000.5,
            month_total=125000.0,
            month_used=12000.25,
            start=dt.date(2026, 1, 1),
            end=dt.date(2026, 12, 31),
        ),
    ]


def test_parse_saldo_skips_lines_without_two_compact_dates():
    text = SALDO_HEADER + (
        "EUHPC_D00_001  2026-07-01  2027-06-30  96000  3187  3187  3.3  8250  0\n"
        "EUHPC_D00_001  20260701    2027/06/30  96000  3187  3187  3.3  8250  0\n"
        "EUHPC_D00_001  20260701    20270630    96000  3187  3187  3.3  8250\n"
        "EUHPC_D00_001  202607010   20270630    96000  3187  3187  3.3  8250  0\n"
    )

    assert parse_saldo(text, resource="Booster") == []


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n\n",
        SALDO_HEADER,
        "  Account   start   end   total\n",
        "username not existing\n",
        "Username NOT EXISTING\n",
        "saldo: no account for user jdoe5678\n",
        "User jdoe5678 not found in the accounting database\n",
    ],
)
def test_parse_saldo_without_budgets(text):
    assert parse_saldo(text, resource="Booster") == []


@pytest.mark.parametrize(
    "text, first_line",
    [
        ("saldo: internal error\n", "saldo: internal error"),
        ("\n\n  Permission denied\nsecond line\n", "Permission denied"),
    ],
)
def test_parse_saldo_unexpected_output_is_an_error(text, first_line):
    with pytest.raises(CollectionError) as excinfo:
        parse_saldo(text, resource="Booster")

    assert str(excinfo.value) == f"unexpected output of saldo: {first_line!r}"


def test_parse_compute_combines_booster_and_dcgp_budgets():
    cluster, warnings = collecting_leonardo()
    results = {"saldo": ok(read_fixture("leonardo/saldo.txt")), "saldo_dcgp": ok(SALDO_DCGP)}

    budgets = cluster.parse_compute(results, TODAY)

    assert budgets == [BOOSTER_BUDGET, DCGP_BUDGET]
    assert warnings == []


def test_parse_compute_failing_booster_saldo_is_a_warning():
    cluster, warnings = collecting_leonardo()
    results = {"saldo": failed(3, stdout="saldo: internal error"), "saldo_dcgp": ok(SALDO_DCGP)}

    budgets = cluster.parse_compute(results, TODAY)

    assert budgets == [DCGP_BUDGET]
    assert warnings == ["Booster budgets unavailable: `saldo` failed (exit status 3): saldo: internal error"]


def test_parse_compute_failing_dcgp_saldo_is_a_warning():
    cluster, warnings = collecting_leonardo()
    results = {"saldo": ok(read_fixture("leonardo/saldo.txt")), "saldo_dcgp": ok("Segmentation fault\n")}

    budgets = cluster.parse_compute(results, TODAY)

    assert budgets == [BOOSTER_BUDGET]
    assert warnings == ["DCGP budgets unavailable: unexpected output of saldo: 'Segmentation fault'"]


@pytest.mark.parametrize(
    "results, message",
    [
        (
            {"saldo": failed(3, stdout="saldo: internal error"), "saldo_dcgp": ok("Segmentation fault\n")},
            "Booster: `saldo` failed (exit status 3): saldo: internal error; "
            "DCGP: unexpected output of saldo: 'Segmentation fault'",
        ),
        ({}, "Booster: no output from `saldo`; DCGP: no output from `saldo_dcgp`"),
    ],
)
def test_parse_compute_both_partitions_failing_is_an_error(results, message):
    cluster, warnings = collecting_leonardo()

    with pytest.raises(CollectionError) as excinfo:
        cluster.parse_compute(results, TODAY)

    assert str(excinfo.value) == message
    assert warnings == []


def test_collect_malformed_dcgp_row_does_not_hide_booster_budgets():
    dcgp = SALDO_DCGP.replace("20270331", "20270230")
    cluster = make_leonardo(fixture_results(saldo_dcgp=dcgp))

    report = cluster.collect(sections=("compute",), today=TODAY, now=NOW)

    assert report.errors == []
    assert report.budgets == [BOOSTER_REPORTED]
    assert len(report.warnings) == 1
    assert report.warnings[0].startswith("DCGP budgets unavailable:")


# -- cindata ------------------------------------------------------------------


@pytest.mark.parametrize(
    "fresh, note",
    [
        ("3hou", "as of 3 hours ago"),
        ("1hou", "as of 1 hour ago"),
        ("1day", "as of 1 day ago"),
        ("2DAY", "as of 2 days ago"),
        ("10sec", "as of 10 seconds ago"),
        ("1min", "as of 1 minute ago"),
        ("2wee", "as of 2 weeks ago"),
        ("1mon", "as of 1 month ago"),
        ("5yea", "as of 5 years ago"),
        ("  3hours  ", "as of 3 hours ago"),
        ("-4min", "as of 4 minutes ago"),
        ("0hou", "as of 0 hours ago"),
        ("soon", None),
        ("hou3", None),
        ("3", None),
        ("3fortnights", None),
        ("--", None),
        ("", None),
    ],
)
def test_describe_freshness(fresh, note):
    assert describe_freshness(fresh) == note


def test_parse_cindata_real_output():
    # Project areas (aUSED set) use the area occupancy and size; personal areas
    # use the user's own occupancy and quota.
    warnings: List[str] = []

    assert parse_cindata(read_fixture("leonardo/cindata.txt"), warn=warnings.append) == FIXTURE_QUOTAS
    assert warnings == []


def test_parse_cindata_lower_case_prefixes_are_decimal():
    text = CINDATA_HEADER + (
        "jdoe   /leonardo_work/EUHPC_D00_002   leonardo_work-1   3hou   --   --   --%   1.5t   10t   15.0%\n"
        "jdoe   /leonardo/home/userexternal/jdoe   leonardo_home-1   3hou   500m   2g   25.0%   --   --   --%\n"
        "jdoe   /leonardo/pub/userexternal/jdoe   leonardo_pub-1   3hou   4k   50G   0.0%   --   --   --%\n"
    )

    assert parse_cindata(text) == [
        StorageQuota(
            name="/leonardo_work/EUHPC_D00_002",
            project="EUHPC_D00_002",
            used_bytes=1_500_000_000_000,
            limit_bytes=10_000_000_000_000,
            note=FRESH_NOTE,
        ),
        StorageQuota(
            name="/leonardo/home/userexternal/jdoe",
            used_bytes=500_000_000,
            limit_bytes=2_000_000_000,
            note=FRESH_NOTE,
        ),
        StorageQuota(name="/leonardo/pub/userexternal/jdoe", used_bytes=4000, limit_bytes=50 * GiB, note=FRESH_NOTE),
    ]


def test_parse_cindata_classifies_personal_and_project_areas():
    text = CINDATA_HEADER + (
        # Personal (userexternal / userinternal): the user's own figures, even
        # when the area figures are filled in.
        "jdoe   /leonardo/home/userexternal/jdoe        a-1   1day   2G     50G    4.0%    9T    10T   90.0%\n"
        "jdoe   /leonardo_scratch/large/userinternal/x  a-2   1day   1G     --     --%     5T    --    --%\n"
        # Personal (last path component = USER), with area figures as well.
        "jdoe   /g100_work/jdoe/                        a-3   1day   3G     10G    30.0%   8T    9T    88.0%\n"
        "jdoe   /leonardo_scratch/jdoe                  a-4   1day   --     --     --%     --    --    --%\n"
        # Project area of a fileset: aUSED set, aQTA unknown.
        "jdoe   /leonardo_work/EUHPC_D00_001/fset01     a-5   1day   --     --     --%     10G   --    --%\n"
        # Project area without area figures: USED / QTA are used.
        "jdoe   /leonardo_work/EUHPC_B02_003            a-6   1day   7G     1T     0.7%    --    --    --%\n"
        # A user name contained in a project path is not enough.
        "jdoe   /leonardo_work/jdoe_project             a-7   1day   --     --     --%     1T    2T    50.0%\n"
    )
    warnings: List[str] = []

    quotas = parse_cindata(text, warn=warnings.append)

    note = "as of 1 day ago"
    assert warnings == []
    assert quotas == [
        StorageQuota(name="/leonardo/home/userexternal/jdoe", used_bytes=2 * GiB, limit_bytes=50 * GiB, note=note),
        StorageQuota(name="/leonardo_scratch/large/userinternal/x", used_bytes=GiB, limit_bytes=None, note=note),
        StorageQuota(name="/g100_work/jdoe/", used_bytes=3 * GiB, limit_bytes=10 * GiB, note=note),
        StorageQuota(name="/leonardo_scratch/jdoe", used_bytes=None, limit_bytes=None, note=note),
        StorageQuota(
            name="/leonardo_work/EUHPC_D00_001/fset01",
            project="fset01",
            used_bytes=10 * GiB,
            limit_bytes=None,
            note=note,
        ),
        StorageQuota(
            name="/leonardo_work/EUHPC_B02_003",
            project="EUHPC_B02_003",
            used_bytes=7 * GiB,
            limit_bytes=TiB,
            note=note,
        ),
        StorageQuota(
            name="/leonardo_work/jdoe_project",
            project="jdoe_project",
            used_bytes=TiB,
            limit_bytes=2 * TiB,
            note=note,
        ),
    ]
    assert all(quota.project is None for quota in quotas[:4])


def test_parse_cindata_unknown_freshness_gives_no_note():
    text = CINDATA_HEADER + (
        "jdoe   /leonardo/home/userexternal/jdoe   leonardo_home-1   unknown   3.4G   50G   6.8%   --   --   --%\n"
        "jdoe   /leonardo/pub/userexternal/jdoe    leonardo_pub-1    --        4K     50G   0.0%   --   --   --%\n"
    )

    assert parse_cindata(text) == [
        StorageQuota(name="/leonardo/home/userexternal/jdoe", used_bytes=round(3.4 * GiB), limit_bytes=50 * GiB),
        StorageQuota(name="/leonardo/pub/userexternal/jdoe", used_bytes=4 * KiB, limit_bytes=50 * GiB),
    ]


def test_parse_cindata_warns_about_rows_with_wrong_field_count():
    lines = [
        "jdoe   /leonardo/home/userexternal/jdoe   leonardo_home-1   3 hours   3.4G   50G   6.8%   --   --   --%",
        "jdoe   /leonardo/pub/userexternal/jdoe   leonardo_pub-1   3hou   4K   50G   0.0%   --   --",
        "------------------------------------------------------------------------------------------",
    ]
    text = CINDATA_HEADER + (
        f"{lines[0]}\n{lines[1]}\n"
        "jdoe   /leonardo_work/EUHPC_D00_001   leonardo_work-1   3hou   --   --   --%   87G   1T   8.5%\n"
        f"{lines[2]}\n"
    )
    warnings: List[str] = []

    assert parse_cindata(text, warn=warnings.append) == [
        StorageQuota(
            name="/leonardo_work/EUHPC_D00_001", project=ACCOUNT, used_bytes=87 * GiB, limit_bytes=TiB, note=FRESH_NOTE
        ),
    ]
    assert warnings == [f"cannot parse cindata line: {line!r}" for line in lines]


def test_parse_cindata_finds_columns_by_name_after_banner_lines():
    text = (
        "Note: occupancy data are refreshed every 3 hours\n"
        "\n"
        "USER   aQTA   aUSED   QTA   USED   AREADESCR\n"
        "jdoe   1T     87G     --    --     /leonardo_work/EUHPC_D00_001/\n"
        "jdoe   --     --      50G   3.4G   /leonardo/home/userexternal/jdoe\n"
        "jdoe   2T     1T      --    --     /\n"
    )

    # Without a FRESH column, no note.
    assert parse_cindata(text) == [
        StorageQuota(name="/leonardo_work/EUHPC_D00_001/", project=ACCOUNT, used_bytes=87 * GiB, limit_bytes=TiB),
        StorageQuota(
            name="/leonardo/home/userexternal/jdoe", used_bytes=round(3.4 * GiB), limit_bytes=50 * GiB
        ),
        StorageQuota(name="/", project=None, used_bytes=TiB, limit_bytes=2 * TiB),
    ]


@pytest.mark.parametrize("text", ["", "\n  \n\t\n", CINDATA_HEADER, "\n" + CINDATA_HEADER + "\n"])
def test_parse_cindata_without_areas(text):
    assert parse_cindata(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "bash: cindata: command not found\n",
        "USERS   AREADESCR   USED   QTA   aUSED   aQTA\n",
        "  user   AREADESCR   USED   QTA   aUSED   aQTA\njdoe /x 1K 2K -- --\n",
        "AREADESCR   USER   USED   QTA   aUSED   aQTA\n/x jdoe 1K 2K -- --\n",
    ],
)
def test_parse_cindata_without_header_is_an_error(text):
    with pytest.raises(CollectionError) as excinfo:
        parse_cindata(text)

    assert str(excinfo.value) == "unexpected output of cindata (no header)"


@pytest.mark.parametrize(
    "header, missing",
    [
        ("USER AREADESCR AREAID FRESH USED QTA USED%", "aUSED, aQTA"),
        ("USER AREADESCR AREAID FRESH used QTA USED% aUSED aQTA aUSED%", "USED"),
        ("USER AREAID FRESH USED QTA USED% aUSED aQTA aUSED%", "AREADESCR"),
        ("USER", "AREADESCR, USED, QTA, aUSED, aQTA"),
    ],
)
def test_parse_cindata_header_missing_columns_is_an_error(header, missing):
    text = header + "\njdoe /leonardo/home/userexternal/jdoe leonardo_home-1 3hou 3.4G 50G 6.8%\n"

    with pytest.raises(CollectionError) as excinfo:
        parse_cindata(text)

    assert str(excinfo.value) == f"unexpected output of cindata (missing columns: {missing})"


def test_parse_cindata_warns_about_and_skips_rows_with_invalid_sizes():
    text = CINDATA_HEADER + (
        "jdoe /leonardo/home/x leonardo_home-1 3hou 3.4X 50G 6.8% -- -- --%\n"
        "jdoe /leonardo/home/userexternal/jdoe leonardo_home-1 3hou 2G 50Q 6.8% -- -- --%\n"
        "jdoe /leonardo_work/EUHPC_D00_001 leonardo_work-1 3hou -- -- --% 87Q 1T 8.5%\n"
        "jdoe /leonardo_work/EUHPC_D00_002 leonardo_work-2 3hou -- -- --% 87G lots 8.5%\n"
        "jdoe /leonardo_work/EUHPC_D00_003 leonardo_work-3 3hou -- -- --% 87G 1T 8.5%\n"
    )
    warnings: List[str] = []

    quotas = parse_cindata(text, warn=warnings.append)

    assert quotas == [
        StorageQuota(
            name="/leonardo_work/EUHPC_D00_003",
            project="EUHPC_D00_003",
            used_bytes=87 * GiB,
            limit_bytes=TiB,
            note=FRESH_NOTE,
        ),
    ]
    assert warnings == [
        "cannot parse the quota of /leonardo/home/x: invalid size unit in '3.4X'",
        "cannot parse the quota of /leonardo/home/userexternal/jdoe: invalid size unit in '50Q'",
        "cannot parse the quota of /leonardo_work/EUHPC_D00_001: invalid size unit in '87Q'",
        "cannot parse the quota of /leonardo_work/EUHPC_D00_002: invalid size: 'lots'",
    ]


def test_parse_cindata_drops_unparsable_rows_without_warn_callback():
    text = CINDATA_HEADER + (
        "jdoe /leonardo/home/x leonardo_home-1 3hou 3.4X 50G 6.8% -- -- --%\n"
        "jdoe /leonardo/pub/userexternal/jdoe leonardo_pub-1 3hou 4K 50G\n"
        "jdoe /leonardo/home/userexternal/jdoe leonardo_home-1 3hou 2G 50G 4.0% -- -- --%\n"
    )

    assert parse_cindata(text) == [
        StorageQuota(name="/leonardo/home/userexternal/jdoe", used_bytes=2 * GiB, limit_bytes=50 * GiB, note=FRESH_NOTE),
    ]


def test_parse_cindata_ignores_the_areaid_column():
    fixture = read_fixture("leonardo/cindata.txt")
    renamed = fixture.replace("leonardo_work-20000001", "anything-else-999")

    assert renamed != fixture
    assert parse_cindata(renamed) == FIXTURE_QUOTAS


def test_parse_storage_reports_unparsable_rows_through_warn():
    cluster, warnings = collecting_leonardo()
    cindata = read_fixture("leonardo/cindata.txt").replace("      87G       2T", "      87G       2Z")

    quotas = cluster.parse_storage({"cindata": ok(cindata)}, TODAY)

    assert quotas == FIXTURE_QUOTAS[1:]
    assert warnings == ["cannot parse the quota of /leonardo_work/EUHPC_D00_001: invalid size unit in '2Z'"]


def test_parse_storage_failing_cindata_is_an_error():
    cluster = make_leonardo()

    with pytest.raises(CollectionError, match=r"^no output from `cindata`$"):
        cluster.parse_storage({}, TODAY)


# -- collect() end to end -----------------------------------------------------


def test_collect_end_to_end_with_real_outputs():
    runner = FakeRunner(fixture_results())
    cluster = LeonardoCluster(make_config("leonardo"), runner=runner)

    report = cluster.collect(today=TODAY, now=NOW)

    assert runner.calls == [EXPECTED_COMMANDS]
    assert report.name == "leonardo"
    assert report.system == "Leonardo"
    assert report.collected_at == NOW
    assert report.sections == ["compute", "storage", "jobs"]
    # "username not existing" for DCGP: no DCGP budget and no warning.
    assert report.errors == []
    assert report.failed_sections == []
    assert report.warnings == []
    assert report.budgets == [BOOSTER_REPORTED]
    assert report.budgets[0].days_left(TODAY) == 224
    assert report.budgets[0].month_remaining == 8250.0
    assert report.storage == FIXTURE_QUOTAS
    assert {quota.note for quota in report.storage} == {"as of 3 hours ago"}
    assert [job.job_id for job in report.jobs] == ["1234567", "1234570", "1234568", "1234569_[1-10]", "1234571"]


def test_collect_with_dcgp_budgets():
    cluster = make_leonardo(fixture_results(saldo=SALDO_TWO_ACCOUNTS, saldo_dcgp=SALDO_DCGP))

    report = cluster.collect(sections=("compute",), today=TODAY, now=NOW)

    assert report.errors == []
    assert report.warnings == []
    assert [(b.project, b.resource, b.used, b.end) for b in report.budgets] == [
        ("EUHPC_D00_001", "Booster", 3187.0, dt.date(2027, 6, 30)),
        ("EUHPC_A01_002", "Booster", 300000.5, dt.date(2026, 12, 31)),
        ("EUHPC_D00_001", "DCGP", 120.0, dt.date(2027, 3, 31)),
    ]


@pytest.mark.parametrize(
    "dcgp, warning",
    [
        (
            failed(1, stderr="saldo: cannot connect to the accounting database\n"),
            "DCGP budgets unavailable: `saldo -b --dcgp` failed (exit status 1): "
            "saldo: cannot connect to the accounting database",
        ),
        (failed(2), "DCGP budgets unavailable: `saldo -b --dcgp` failed (exit status 2)"),
        (failed(None), "DCGP budgets unavailable: `saldo -b --dcgp` did not complete: no exit status"),
        (
            CommandResult(returncode=None, reason="the SSH session ended unexpectedly (exit status 255)"),
            "DCGP budgets unavailable: `saldo -b --dcgp` did not complete: "
            "the SSH session ended unexpectedly (exit status 255)",
        ),
        (failed(124), "DCGP budgets unavailable: `saldo -b --dcgp` timed out (time limit of 180 s reached)"),
        (None, "DCGP budgets unavailable: no output from `saldo -b --dcgp`"),
        (ok("Segmentation fault\n"), "DCGP budgets unavailable: unexpected output of saldo: 'Segmentation fault'"),
    ],
)
def test_collect_dcgp_problems_are_only_warnings(dcgp, warning):
    cluster = make_leonardo(fixture_results(saldo_dcgp=dcgp))

    report = cluster.collect(sections=("compute",), today=TODAY, now=NOW)

    assert report.errors == []
    assert report.failed_sections == []
    assert report.warnings == [warning]
    assert report.budgets == [BOOSTER_REPORTED]


@pytest.mark.parametrize(
    "saldo, warning",
    [
        (
            failed(1, stderr="saldo: error\n"),
            "Booster budgets unavailable: `saldo -b` failed (exit status 1): saldo: error",
        ),
        (failed(None), "Booster budgets unavailable: `saldo -b` did not complete: no exit status"),
        (failed(124), "Booster budgets unavailable: `saldo -b` timed out (time limit of 180 s reached)"),
        (ok("Service unavailable\n"), "Booster budgets unavailable: unexpected output of saldo: 'Service unavailable'"),
    ],
)
def test_collect_booster_problems_are_only_warnings_when_dcgp_works(saldo, warning):
    cluster = make_leonardo(fixture_results(saldo=saldo, saldo_dcgp=SALDO_DCGP))

    report = cluster.collect(today=TODAY, now=NOW)

    assert report.errors == []
    assert report.failed_sections == []
    assert report.warnings == [warning]
    assert report.budgets == [DCGP_BUDGET]
    assert report.storage == FIXTURE_QUOTAS
    assert len(report.jobs) == 5


@pytest.mark.parametrize(
    "saldo, dcgp, error",
    [
        (
            failed(1, stderr="saldo: error\n"),
            failed(2),
            "compute: Booster: `saldo -b` failed (exit status 1): saldo: error; "
            "DCGP: `saldo -b --dcgp` failed (exit status 2)",
        ),
        (
            ok("Service unavailable\n"),
            failed(124),
            "compute: Booster: unexpected output of saldo: 'Service unavailable'; "
            "DCGP: `saldo -b --dcgp` timed out (time limit of 180 s reached)",
        ),
        (
            None,
            None,
            "compute: Booster: no output from `saldo -b`; DCGP: no output from `saldo -b --dcgp`",
        ),
    ],
)
def test_collect_both_partitions_failing_is_a_compute_error_only(saldo, dcgp, error):
    cluster = make_leonardo(fixture_results(saldo=saldo, saldo_dcgp=dcgp))

    report = cluster.collect(today=TODAY, now=NOW)

    assert report.errors == [error]
    assert report.failed_sections == ["compute"]
    assert report.warnings == []
    assert report.budgets == []
    assert report.storage == FIXTURE_QUOTAS
    assert len(report.jobs) == 5


@pytest.mark.parametrize(
    "cindata, error",
    [
        (failed(1, stderr="cindata: error\n"), "storage: `cindata` failed (exit status 1): cindata: error"),
        (failed(124), "storage: `cindata` timed out (time limit of 180 s reached)"),
        (failed(None), "storage: `cindata` did not complete: no exit status"),
        (ok("Unexpected banner\n"), "storage: unexpected output of cindata (no header)"),
        (ok("USER AREADESCR USED QTA\n"), "storage: unexpected output of cindata (missing columns: aUSED, aQTA)"),
    ],
)
def test_collect_cindata_problems_are_storage_errors_only(cindata, error):
    cluster = make_leonardo(fixture_results(cindata=cindata))

    report = cluster.collect(sections=("compute", "storage"), today=TODAY, now=NOW)

    assert report.errors == [error]
    assert report.failed_sections == ["storage"]
    assert report.storage == []
    assert report.budgets == [BOOSTER_REPORTED]


def test_collect_cindata_invalid_sizes_are_warnings():
    cindata = read_fixture("leonardo/cindata.txt") + (
        "jdoe5678   /leonardo/home/x   leonardo_home-1   3hou   4X   50G   0.0%   --   --   --%\n"
    )
    cluster = make_leonardo(fixture_results(cindata=cindata))

    report = cluster.collect(sections=("compute", "storage"), today=TODAY, now=NOW)

    assert report.errors == []
    assert report.failed_sections == []
    assert report.warnings == ["cannot parse the quota of /leonardo/home/x: invalid size unit in '4X'"]
    assert report.storage == FIXTURE_QUOTAS
    assert report.budgets == [BOOSTER_REPORTED]


def test_collect_storage_only_runs_cindata():
    runner = FakeRunner(fixture_results())
    cluster = LeonardoCluster(make_config("leonardo"), runner=runner)

    report = cluster.collect(sections=("storage",), today=TODAY, now=NOW)

    assert runner.calls == [{"cindata": "cindata"}]
    assert report.ok
    assert report.sections == ["storage"]
    assert report.budgets == []
    assert report.storage == FIXTURE_QUOTAS


@pytest.mark.parametrize(
    "projects, budgets, areas, warnings",
    [
        (
            ["euhpc_d00_001"],
            [ACCOUNT],
            [quota.name for quota in FIXTURE_QUOTAS],
            [],
        ),
        (
            ["EUHPC_OTHER"],
            [],
            [
                "/leonardo_scratch/large/userexternal/jdoe5678",
                "/leonardo/pub/userexternal/jdoe5678",
                "/leonardo/home/userexternal/jdoe5678",
            ],
            ["project(s) not found on this cluster: EUHPC_OTHER"],
        ),
    ],
)
def test_collect_filters_projects_case_insensitively(projects, budgets, areas, warnings):
    cluster = make_leonardo(fixture_results(), projects=projects)

    report = cluster.collect(sections=("compute", "storage"), today=TODAY, now=NOW)

    assert report.errors == []
    assert [budget.project for budget in report.budgets] == budgets
    assert [quota.name for quota in report.storage] == areas
    assert report.warnings == warnings
