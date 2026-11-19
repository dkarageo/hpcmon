"""Tests for the MeluXina interface (:mod:`hpcmon.clusters.meluxina`).

Everything comes from a single run of LuxProvide's ``myquota``, whose
fixed-width, ANSI-coloured tables are parsed for both the compute and the
storage sections.  The raw fixture ``meluxina/myquota.txt`` keeps the colour
codes of the real output.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any, List

import pytest

from hpcmon.clusters import CLUSTER_TYPES
from hpcmon.clusters.base import CollectionError
from hpcmon.clusters.meluxina import (
    MONTHLY_NOTE,
    MeluxinaCluster,
    _check_percent,
    parse_compute_table,
    parse_data_table,
    strip_ansi,
)
from hpcmon.clusters.slurm import SQUEUE_COMMAND
from hpcmon.models import ComputeBudget, StorageQuota
from hpcmon.runner import CommandResult
from tests.conftest import NOW, TODAY, FakeRunner, failed, make_config, ok, read_fixture

GIB = 1024**3
TIB = 1024**4

RATE_LIMITED = "myquota refused to run (it can only run once every 10 seconds); retry later"
RATE_LIMITED_LINE = "[ERROR] Wait at least 10 seconds before executing myquota again"

#: The only allocation in the fixture: 27 of 210 GPU node-hours this month.
FIXTURE_BUDGET = ComputeBudget(
    project="p200001",
    resource="GPU",
    unit="node-h",
    month_total=210.0,
    month_used=27.0,
    note=MONTHLY_NOTE,
)

#: The budget as reported by the cluster, with its GPU-hours rate
#: (a GPU node-hour is 4 A100-hours).
FIXTURE_REPORTED = dataclasses.replace(FIXTURE_BUDGET, gpu_hours_per_unit=4.0)

FIXTURE_QUOTAS = [
    StorageQuota(
        name="/home/users/u100001",
        project=None,
        used_bytes=5 * GIB,
        limit_bytes=150 * GIB,
        files_used=4512,
        files_limit=150000,
    ),
    StorageQuota(
        name="/project/home/p200001",
        project="p200001",
        used_bytes=213 * GIB,
        limit_bytes=5000 * GIB,
        files_used=187342,
        files_limit=2000000,
    ),
]

#: Two partitions, three projects; the table ends without a blank line.
SEVERAL_PROJECTS_TABLE = "\n".join(
    [
        "Project          CPU node-hours        GPU node-hours",
        "                Used   Max   %used    Used   Max   %used",
        "p200001           10    100   10.0%     0",
        "p200002                                 5     50   10.0%",
        "p200003            7",
    ]
)

#: A single partition, for testing the cells of one row.
SINGLE_PARTITION_HEADER = [
    "Project      CPU node-hours",
    "            Used   Max   %used",
]

#: The start of a data table, without a unit (rows default to GiB).
DATA_TABLE_START = "                          DATA ALLOCATIONS"


def make_cluster(result: Any = None, **config: Any) -> MeluxinaCluster:
    results = {} if result is None else {"myquota": result}
    return MeluxinaCluster(make_config("meluxina", **config), runner=FakeRunner(results))


def results_of(result: Any) -> dict:
    return {"myquota": ok(result) if isinstance(result, str) else result}


def fixture_text() -> str:
    return read_fixture("meluxina/myquota.txt")


def single_partition(*rows: str) -> str:
    """A one-partition compute table with ``rows``."""
    return "\n".join(SINGLE_PARTITION_HEADER + list(rows))


def data_table(*rows: str) -> str:
    """A data table (without a unit header) with ``rows``."""
    return "\n".join([DATA_TABLE_START] + list(rows))


def compute_only_text() -> str:
    """The fixture without its data table."""
    text = strip_ansi(fixture_text())
    return text[: text.index("DATA ALLOCATIONS")].rstrip() + "\n"


def data_only_text() -> str:
    """The fixture without its compute table."""
    text = strip_ansi(fixture_text())
    start = text.index("DATA ALLOCATIONS")
    return text[text.rindex("\n", 0, start) + 1 :]


# -- strip_ansi -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("\x1b[1mProject\x1b[0m", "Project"),
        ("\x1b[92m 12.9%  \x1b[0m", " 12.9%  "),
        ("\x1b[1;31mred\x1b[0m and \x1b[38;5;208morange\x1b[m", "red and orange"),
        ("\x1b[?25lhidden cursor\x1b[?25h", "hidden cursor"),
        ("\x1b[2K\x1b[1A", ""),
        ("plain | text 100%", "plain | text 100%"),
        ("", ""),
    ],
)
def test_strip_ansi(text, expected):
    assert strip_ansi(text) == expected


def test_strip_ansi_removes_every_escape_of_the_fixture_and_keeps_the_layout():
    raw = fixture_text()
    assert "\x1b" in raw
    text = strip_ansi(raw)
    assert "\x1b" not in text
    assert len(text.splitlines()) == len(raw.splitlines())
    assert "/project/home/p200001                   213 |    5000 |   4%" in text


# -- commands -------------------------------------------------------------------


def test_cluster_is_registered_under_type_meluxina():
    assert CLUSTER_TYPES["meluxina"] is MeluxinaCluster
    assert MeluxinaCluster.type_name == "meluxina"
    assert MeluxinaCluster.display_name == "MeluXina"
    assert MeluxinaCluster.supported_sections == ("compute", "storage", "jobs")


def test_monthly_note():
    assert MONTHLY_NOTE == "current month only"


def test_compute_and_storage_share_the_single_myquota_command():
    cluster = make_cluster()
    assert cluster.compute_commands() == {"myquota": "myquota"}
    assert cluster.storage_commands() == {"myquota": "myquota"}
    assert cluster.commands(["compute", "storage"]) == {"myquota": "myquota"}
    assert cluster.commands() == {"myquota": "myquota", "jobs": SQUEUE_COMMAND}
    # squeue (fast) runs before myquota.
    assert list(cluster.commands()) == ["jobs", "myquota"]


# -- compute table ------------------------------------------------------------


def test_parse_compute_table_of_the_fixture():
    budgets = parse_compute_table(strip_ansi(fixture_text()))
    assert budgets == [FIXTURE_BUDGET]
    budget = budgets[0]
    assert budget.note == "current month only"
    assert (budget.total, budget.used, budget.available, budget.remaining) == (None, None, None, None)
    assert budget.month_remaining == 183.0
    assert budget.end is None


def test_parse_compute_table_with_several_projects_and_partitions():
    assert parse_compute_table(SEVERAL_PROJECTS_TABLE) == [
        ComputeBudget(project="p200001", resource="CPU", unit="node-h", month_total=100.0, month_used=10.0, note=MONTHLY_NOTE),
        ComputeBudget(project="p200002", resource="GPU", unit="node-h", month_total=50.0, month_used=5.0, note=MONTHLY_NOTE),
        # Usage without an allocation is still reported (without a total).
        ComputeBudget(project="p200003", resource="CPU", unit="node-h", month_total=None, month_used=7.0, note=MONTHLY_NOTE),
    ]


@pytest.mark.parametrize(
    "separator",
    ["\n\n", "\n   \n", "\n\n\n"],
    ids=["one-blank-line", "whitespace-line", "two-blank-lines"],
)
def test_parse_compute_table_skips_blank_lines_between_rows(separator):
    text = SEVERAL_PROJECTS_TABLE.replace("\np200002", separator + "p200002").replace("\np200003", separator + "p200003")
    assert [budget.project for budget in parse_compute_table(text)] == ["p200001", "p200002", "p200003"]


@pytest.mark.parametrize(
    "data_header",
    [
        "                          DATA ALLOCATIONS",
        " Datapath                Data GiB              No. Files",
    ],
    ids=["data-allocations", "datapath"],
)
@pytest.mark.parametrize("blank", ["", "\n"], ids=["directly", "after-blank-line"])
def test_parse_compute_table_stops_at_the_data_table(data_header, blank):
    text = "\n".join(
        [
            SEVERAL_PROJECTS_TABLE + blank,
            data_header,
            "                Used |   Max   | Use%        Used |     Max    | Use%",
            "/home/users/u1      5 |     150 |   3%        4512 |     150000 |   3%",
        ]
    )
    assert [budget.project for budget in parse_compute_table(text)] == ["p200001", "p200002", "p200003"]


def test_parse_compute_table_header_without_percent_columns():
    text = "\n".join(
        [
            "Project    CPU node-hours   GPU node-hours",
            "           Used   Max       Used   Max",
            "p1          3     10         4      20",
        ]
    )
    budgets = parse_compute_table(text)
    assert [(b.resource, b.month_used, b.month_total) for b in budgets] == [("CPU", 3.0, 10.0), ("GPU", 4.0, 20.0)]


def test_parse_compute_table_header_without_percent_column_of_the_last_partition():
    text = "\n".join(
        [
            "Project      CPU node-hours        GPU node-hours",
            "            Used   Max   %used    Used   Max",
            "p1             1     2    50%       3     4",
        ]
    )
    budgets = parse_compute_table(text)
    assert [(b.resource, b.month_used, b.month_total) for b in budgets] == [("CPU", 1.0, 2.0), ("GPU", 3.0, 4.0)]


@pytest.mark.parametrize(
    "row, expected",
    [
        ("p1           0    100    0.0%", (0.0, 100.0)),
        ("p1          27    210   12.9%", (27.0, 210.0)),
        ("p1       1,234  2,000   61.7%", (1234.0, 2000.0)),
        ("p1        12.5", (12.5, None)),
        ("p1           -     50       -", (None, 50.0)),
        ("p1           0", None),
        ("p1           0      -       -", None),
        ("p1           -      -       -", None),
        ("p1", None),
    ],
)
def test_parse_compute_table_cells(row, expected):
    budgets = parse_compute_table(single_partition(row))
    if expected is None:
        assert budgets == []
    else:
        assert [(b.project, b.resource, b.month_used, b.month_total) for b in budgets] == [("p1", "CPU") + expected]


@pytest.mark.parametrize(
    "text",
    [
        "Project      CPU node-hours",
        "intro\n\nProject      CPU node-hours        GPU node-hours",
    ],
    ids=["header-only", "header-last"],
)
def test_parse_compute_table_without_a_complete_header(text):
    assert parse_compute_table(text) == []


@pytest.mark.parametrize(
    "text, first_line",
    [
        ("", ""),
        ("  \n\n", ""),
        ("no compute table here\nat all\n", "no compute table here"),
        ("\n   \n  something odd  \nmore\n", "something odd"),
        (RATE_LIMITED_LINE + "\n", RATE_LIMITED_LINE),
    ],
    ids=["empty", "blank", "no-header", "leading-blank-lines", "rate-limited"],
)
def test_parse_compute_table_without_any_table_is_an_error(text, first_line):
    with pytest.raises(CollectionError) as excinfo:
        parse_compute_table(text)
    assert str(excinfo.value) == f"unexpected output of myquota (no compute table): {first_line!r}"


def test_parse_compute_table_of_the_rate_limited_fixture_names_its_first_line():
    with pytest.raises(CollectionError) as excinfo:
        parse_compute_table(strip_ansi(read_fixture("meluxina/myquota_rate_limited.txt")))
    assert str(excinfo.value) == (
        "unexpected output of myquota (no compute table): "
        "'[ERROR] Wait at least 10 seconds before executing myquota again'"
    )


@pytest.mark.parametrize(
    "text",
    [
        DATA_TABLE_START + "\n/home/users/u1   3 |  100 |  3%   10 |  100 |  10%",
        " Datapath    Data GiB    No. Files\n",
    ],
    ids=["data-allocations", "datapath"],
)
def test_parse_compute_table_with_only_a_data_table_has_no_budget(text):
    assert parse_compute_table(text) == []


def test_parse_compute_table_of_the_fixture_without_compute_table():
    assert parse_compute_table(data_only_text()) == []


def test_parse_compute_table_header_without_rows():
    assert parse_compute_table(single_partition()) == []
    assert parse_compute_table(single_partition("", "   ")) == []


def test_parse_compute_table_row_after_a_blank_line_after_the_header():
    budgets = parse_compute_table(single_partition("", "p1  5  10  50%"))
    assert [(b.project, b.month_used, b.month_total) for b in budgets] == [("p1", 5.0, 10.0)]


@pytest.mark.parametrize(
    "subheader",
    [
        "            Used   Max   %used",
        "            Used   Max   %used    Used   Max   %used    Used",
        "",
    ],
    ids=["too-few", "too-many", "blank"],
)
def test_parse_compute_table_rejects_a_mismatched_header(subheader):
    text = "\n".join(["Project      CPU node-hours        GPU node-hours", subheader, "p1  1  2  50%"])
    with pytest.raises(CollectionError) as excinfo:
        parse_compute_table(text)
    assert str(excinfo.value) == "unexpected output of myquota (compute table header)"


def test_parse_compute_table_with_garbage_cell_raises_value_error():
    with pytest.raises(ValueError):
        parse_compute_table(single_partition("p1         n/a  lots"))


# -- %used cross-check ------------------------------------------------------------


@pytest.mark.parametrize(
    "row, expected",
    [
        ("p1          10    100   11.4%", (10.0, 100.0)),  # within 0.5 + 100/100 = 1.5 points
        ("p1          10    100    8.6%", (10.0, 100.0)),
        ("p1          10    100   10.0%", (10.0, 100.0)),
        ("p1           1      4   35.0%", (1.0, 4.0)),  # small Max: whole hours, large tolerance
        ("p1        1000  10000   10.4%", (1000.0, 10000.0)),
        ("p1           0    100    0.0%", (0.0, 100.0)),
    ],
    ids=["above", "below", "exact", "small-max", "large-max", "unused"],
)
def test_parse_compute_table_accepts_consistent_percentages(row, expected):
    budgets = parse_compute_table(single_partition(row))
    assert [(b.month_used, b.month_total) for b in budgets] == [expected]


@pytest.mark.parametrize(
    "row, expected",
    [
        ("p1           0      -    0.0%", []),
        ("p1           -      -    0.0%", []),
        # A zero allocation is still an allocation (shown, but not checked).
        ("p1           -      0    0.0%", [(None, 0.0)]),
        ("p1           0      0    0.0%", [(0.0, 0.0)]),
    ],
    ids=["no-max", "dashes", "zero-max", "zero-max-and-used"],
)
def test_parse_compute_table_does_not_check_rows_without_allocation_nor_usage(row, expected):
    budgets = parse_compute_table(single_partition(row))
    assert [(b.month_used, b.month_total) for b in budgets] == expected


@pytest.mark.parametrize(
    "row",
    [
        "p1          10    100   11.7%",  # just outside the tolerance
        "p1          10    100    8.3%",
        "p1          27    210   50.0%",
        "p1        1000  10000   11.0%",  # large Max: tolerance close to 0.5 points
        "p1     100000210          12.9%",  # Used and Max merged into one cell
        "p1           5      -    5.0%",  # no Max
        "p1           5      0    5.0%",  # zero Max
        "p1           -     50    2.0%",  # no Used
    ],
    ids=["above", "below", "inconsistent", "large-max", "merged-cells", "no-max", "zero-max", "no-used"],
)
def test_parse_compute_table_rejects_inconsistent_percentages(row):
    with pytest.raises(CollectionError) as excinfo:
        parse_compute_table(single_partition(row))
    assert str(excinfo.value) == f"cannot parse the CPU allocation of p1: {row.strip()!r}"


def test_parse_compute_table_checks_every_partition():
    text = SEVERAL_PROJECTS_TABLE.replace("5     50   10.0%", "5     50   40.0%")
    with pytest.raises(CollectionError) as excinfo:
        parse_compute_table(text)
    assert str(excinfo.value) == (
        "cannot parse the GPU allocation of p200002: 'p200002                                 5     50   40.0%'"
    )


def test_parse_compute_table_detects_merged_cells_in_the_fixture():
    line = next(line for line in strip_ansi(fixture_text()).splitlines() if line.startswith("p200001"))
    merged = line.replace("27    210", "100000210")
    assert len(merged) == len(line)
    with pytest.raises(CollectionError) as excinfo:
        parse_compute_table(strip_ansi(fixture_text()).replace(line, merged))
    assert str(excinfo.value) == f"cannot parse the GPU allocation of p200001: {merged.strip()!r}"


@pytest.mark.parametrize(
    "used, limit, percent",
    [
        (10.0, 100.0, None),  # no percentage: nothing to compare with
        (None, None, 50.0),  # no allocation and no usage
        (0.0, None, 50.0),
        (None, 0.0, 50.0),
        (0.0, 0.0, 0.0),
        (10.0, 100.0, 11.4),
        (27.0, 210.0, 12.9),
    ],
)
def test_check_percent_accepts(used, limit, percent):
    assert _check_percent("p1", "GPU", used, limit, percent, "  the line  ") is None


@pytest.mark.parametrize(
    "used, limit, percent",
    [
        (10.0, None, 10.0),
        (10.0, 0.0, 10.0),
        (None, 100.0, 10.0),
        (10.0, 100.0, 11.6),
        (10.0, 100.0, 8.4),
    ],
)
def test_check_percent_rejects(used, limit, percent):
    with pytest.raises(CollectionError) as excinfo:
        _check_percent("p1", "GPU", used, limit, percent, "  the line  ")
    assert str(excinfo.value) == "cannot parse the GPU allocation of p1: 'the line'"


# -- data table ---------------------------------------------------------------


def test_parse_data_table_of_the_fixture():
    quotas = parse_data_table(strip_ansi(fixture_text()))
    assert quotas == FIXTURE_QUOTAS
    assert quotas[0].used_fraction == pytest.approx(5 / 150)
    assert quotas[1].files_fraction == pytest.approx(0.093671)


def test_parse_data_table_of_the_fixture_gives_no_warning():
    warnings: List[str] = []
    parse_data_table(strip_ansi(fixture_text()), warn=warnings.append)
    assert warnings == []


@pytest.mark.parametrize(
    "header, multiplier",
    [
        ("Data TiB", TIB),
        ("Data MiB", 1024**2),
        ("Data GiB", GIB),
        ("Data TB", TIB),
        ("Data KiB", 1024),
    ],
)
def test_parse_data_table_unit_from_the_header(header, multiplier):
    text = "\n".join(
        [
            f" Datapath            {header}            No. Files",
            "                Used |   Max   | Use%        Used |     Max    | Use%",
            "/project/home/p1     2 |      10 |  20%           5 |        100 |   5%",
        ]
    )
    assert parse_data_table(text) == [
        StorageQuota(
            name="/project/home/p1",
            project="p1",
            used_bytes=2 * multiplier,
            limit_bytes=10 * multiplier,
            files_used=5,
            files_limit=100,
        )
    ]


def test_parse_data_table_unit_applies_to_the_following_rows_only():
    text = data_table(
        "/home/users/u1   1 |  2 |  50%   3 |  4 |  75%",
        "Data TiB",
        "/project/home/p1   1 |  2 |  50%   3 |  4 |  75%",
    )
    quotas = parse_data_table(text)
    assert [(q.used_bytes, q.limit_bytes) for q in quotas] == [(GIB, 2 * GIB), (TIB, 2 * TIB)]


def test_parse_data_table_fractional_sizes():
    quota = parse_data_table(data_table("/home/users/u1   1.5 |  2.25 |  67%   3 |  4 |  75%"))[0]
    assert (quota.used_bytes, quota.limit_bytes) == (int(1.5 * GIB), int(2.25 * GIB))


@pytest.mark.parametrize(
    "row, expected",
    [
        (
            "/project/home/p200002     - |       - |   -        - |        - |   -",
            StorageQuota(name="/project/home/p200002", project="p200002"),
        ),
        (
            "/project/home/p200002    -- |     --- |   -       -- |      --- |   -",
            StorageQuota(name="/project/home/p200002", project="p200002"),
        ),
        (
            "/project/home/p200003     5 |         |              12 |            |     ",
            StorageQuota(name="/project/home/p200003", project="p200003", used_bytes=5 * GIB, files_used=12),
        ),
        (
            "/project/home/p4/   1 | 2 | 50%   3 | 4 | 75%",
            StorageQuota(
                name="/project/home/p4/", project="p4", used_bytes=GIB, limit_bytes=2 * GIB, files_used=3, files_limit=4
            ),
        ),
        (
            "/scratch/u1   1,024 | 2,048 | 50%  1,000 | 2,000 | 50%",
            StorageQuota(
                name="/scratch/u1", used_bytes=1024 * GIB, limit_bytes=2048 * GIB, files_used=1000, files_limit=2000
            ),
        ),
        (
            "/home/users/project/x   0 | 100 | 0%   0 | 10 | 0%",
            StorageQuota(name="/home/users/project/x", used_bytes=0, limit_bytes=100 * GIB, files_used=0, files_limit=10),
        ),
    ],
    ids=["dashes", "longer-dashes", "empty-max", "trailing-slash", "thousands", "not-a-project"],
)
def test_parse_data_table_rows(row, expected):
    assert parse_data_table(data_table(row)) == [expected]


def test_parse_data_table_row_with_four_dashes_is_missing():
    row = "/project/home/p200002  ---- |  ---- |  -   ---- |  ---- |  -"
    assert parse_data_table(data_table(row)) == [StorageQuota(name="/project/home/p200002", project="p200002")]


@pytest.mark.parametrize(
    "line",
    [
        "",
        " Datapath                                      Data                           No. Files",
        "                                       Used |   Max   | Use%             Used |     Max    | Use%",
        "home/users/u1   1 |  2 |  50%   3 |  4 |  75%",
        "p200001                             0                          27    210    12.9%",
    ],
    ids=["blank", "header-without-unit", "subheader", "relative-path", "compute-row"],
)
def test_parse_data_table_ignores_other_lines(line):
    warnings: List[str] = []
    assert parse_data_table(data_table(line), warn=warnings.append) == []
    assert warnings == []


@pytest.mark.parametrize(
    "line",
    [
        "/home/users/u1   1 |  2",
        "/home/users/u1   one |  2 |  50%   3 |  4 |  75%",
        "   /project/home/p1   (unavailable)   ",
    ],
    ids=["truncated", "words", "indented-message"],
)
def test_parse_data_table_warns_about_unparsable_path_lines(line):
    warnings: List[str] = []
    assert parse_data_table(data_table(line), warn=warnings.append) == []
    assert warnings == [f"cannot parse myquota data line: {line.strip()!r}"]


def test_parse_data_table_keeps_the_other_rows_of_a_bad_line():
    warnings: List[str] = []
    text = data_table(
        "/home/users/u1   1 |  2 |  50%   3 |  4 |  75%",
        "/project/home/p1   ??",
        "/project/home/p2   1 |  2 |  50%   3 |  4 |  75%",
    )
    quotas = parse_data_table(text, warn=warnings.append)
    assert [quota.name for quota in quotas] == ["/home/users/u1", "/project/home/p2"]
    assert warnings == ["cannot parse myquota data line: '/project/home/p1   ??'"]


def test_parse_data_table_ignores_bad_lines_by_default():
    assert parse_data_table(data_table("/home/users/u1   1 |  2")) == []


@pytest.mark.parametrize(
    "text, first_line",
    [
        ("", ""),
        ("/home/users/u1   1 |  2 |  50%   3 |  4 |  75%", "/home/users/u1   1 |  2 |  50%   3 |  4 |  75%"),
        ("\n  Data TiB  \n", "Data TiB"),
        (RATE_LIMITED_LINE + "\n", RATE_LIMITED_LINE),
    ],
    ids=["empty", "rows-without-header", "unit-only", "rate-limited"],
)
def test_parse_data_table_without_any_table_is_an_error(text, first_line):
    with pytest.raises(CollectionError) as excinfo:
        parse_data_table(text)
    assert str(excinfo.value) == f"unexpected output of myquota (no data table): {first_line!r}"


def test_parse_data_table_of_the_rate_limited_fixture_is_an_error():
    with pytest.raises(CollectionError) as excinfo:
        parse_data_table(strip_ansi(read_fixture("meluxina/myquota_rate_limited.txt")))
    assert str(excinfo.value) == f"unexpected output of myquota (no data table): {RATE_LIMITED_LINE!r}"


@pytest.mark.parametrize(
    "text",
    [SEVERAL_PROJECTS_TABLE, "\n".join(SINGLE_PARTITION_HEADER)],
    ids=["compute-table", "compute-header"],
)
def test_parse_data_table_with_only_a_compute_table_has_no_quota(text):
    assert parse_data_table(text) == []


def test_parse_data_table_of_the_fixture_without_data_table():
    assert parse_data_table(compute_only_text()) == []


# -- parsing through the cluster ----------------------------------------------


def test_parse_compute_and_storage_strip_the_colours():
    cluster = make_cluster()
    results = results_of(fixture_text())
    assert cluster.parse_compute(results, TODAY) == [FIXTURE_BUDGET]
    assert cluster.parse_storage(results, TODAY) == FIXTURE_QUOTAS


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize("method", ["parse_compute", "parse_storage"])
def test_rate_limited_myquota_is_an_error(method, returncode):
    result = CommandResult(stdout=read_fixture("meluxina/myquota_rate_limited.txt"), returncode=returncode)
    with pytest.raises(CollectionError) as excinfo:
        getattr(make_cluster(), method)(results_of(result), TODAY)
    assert str(excinfo.value) == RATE_LIMITED


def test_rate_limit_message_is_detected_through_colours_and_case():
    result = ok("\x1b[91m[ERROR]\x1b[0m WAIT \x1b[1mAT LEAST\x1b[0m 10 seconds\n")
    with pytest.raises(CollectionError) as excinfo:
        make_cluster().parse_compute(results_of(result), TODAY)
    assert str(excinfo.value) == RATE_LIMITED


@pytest.mark.parametrize("returncode", [0, 1])
@pytest.mark.parametrize("method", ["parse_compute", "parse_storage"])
@pytest.mark.parametrize(
    "stdout",
    ["", "\n\n", fixture_text()],
    ids=["no-stdout", "blank-stdout", "with-tables"],
)
def test_rate_limit_message_on_stderr_is_an_error(stdout, method, returncode):
    result = CommandResult(
        stdout=stdout,
        stderr="\x1b[91m[ERROR]\x1b[0m Wait \x1b[1mat least\x1b[0m 10 seconds before executing myquota again\n",
        returncode=returncode,
    )
    with pytest.raises(CollectionError) as excinfo:
        getattr(make_cluster(), method)(results_of(result), TODAY)
    assert str(excinfo.value) == RATE_LIMITED


@pytest.mark.parametrize(
    "stdout, stderr, message",
    [
        ("", "", "`myquota` printed nothing"),
        ("  \n\n", "", "`myquota` printed nothing"),
        ("\x1b[0m\n", "\n \n", "`myquota` printed nothing"),
        ("", "note: no project found\n", "`myquota` printed nothing: note: no project found"),
        ("", "first\n  \x1b[93mWARNING\x1b[0m: empty database  \n\n", "`myquota` printed nothing: WARNING: empty database"),
    ],
    ids=["empty", "blank", "only-colour-codes", "stderr-detail", "last-stderr-line-without-colours"],
)
def test_successful_myquota_without_output_is_an_error(stdout, stderr, message):
    result = CommandResult(stdout=stdout, stderr=stderr, returncode=0)
    for method in ("parse_compute", "parse_storage"):
        with pytest.raises(CollectionError) as excinfo:
            getattr(make_cluster(), method)(results_of(result), TODAY)
        assert str(excinfo.value) == message


@pytest.mark.parametrize(
    "stdout, stderr, message",
    [
        ("", "Traceback (most recent call last):\n  ...\nPermissionError: denied\n\n", "`myquota` failed (exit status 2): PermissionError: denied"),
        ("  \n\n", "  bash: myquota: command not found  ", "`myquota` failed (exit status 2): bash: myquota: command not found"),
        ("\x1b[0m\n", "", "`myquota` failed (exit status 2)"),
        ("", "\n \n", "`myquota` failed (exit status 2)"),
        ("", "\x1b[91merror\x1b[0m: no access\n", "`myquota` failed (exit status 2): error: no access"),
    ],
    ids=["traceback", "blank-stdout", "only-colour-codes", "blank-stderr", "coloured-stderr"],
)
def test_failed_myquota_without_output_is_an_error(stdout, stderr, message):
    result = failed(2, stdout=stdout, stderr=stderr)
    for method in ("parse_compute", "parse_storage"):
        with pytest.raises(CollectionError) as excinfo:
            getattr(make_cluster(), method)(results_of(result), TODAY)
        assert str(excinfo.value) == message


def test_failed_myquota_with_output_is_still_parsed():
    result = failed(1, stdout=fixture_text(), stderr="warning: slow database\n")
    cluster = make_cluster()
    assert cluster.parse_compute(results_of(result), TODAY) == [FIXTURE_BUDGET]
    assert cluster.parse_storage(results_of(result), TODAY) == FIXTURE_QUOTAS


@pytest.mark.parametrize(
    "result, message",
    [
        (failed(124, stdout=fixture_text()), "`myquota` timed out (time limit of 180 s reached)"),
        (failed(None, stdout=fixture_text()), "`myquota` did not complete: no exit status"),
        (
            CommandResult(stdout=fixture_text(), returncode=None, reason="the SSH session ended unexpectedly (exit status 255)"),
            "`myquota` did not complete: the SSH session ended unexpectedly (exit status 255)",
        ),
        # The time limit is checked before the rate-limit message.
        (failed(124, stdout=RATE_LIMITED_LINE), "`myquota` timed out (time limit of 180 s reached)"),
    ],
    ids=["timeout-124", "unfinished", "unfinished-with-reason", "timeout-before-rate-limit"],
)
@pytest.mark.parametrize("method", ["parse_compute", "parse_storage"])
def test_timed_out_myquota_is_an_error(method, result, message):
    with pytest.raises(CollectionError) as excinfo:
        getattr(make_cluster(), method)(results_of(result), TODAY)
    assert str(excinfo.value) == message


def test_timed_out_myquota_names_the_configured_limit():
    with pytest.raises(CollectionError) as excinfo:
        make_cluster(timeout=42.5).parse_storage(results_of(failed(124)), TODAY)
    assert str(excinfo.value) == "`myquota` timed out (time limit of 42.5 s reached)"


def test_missing_myquota_output_is_an_error():
    with pytest.raises(CollectionError) as excinfo:
        make_cluster().parse_compute({}, TODAY)
    assert str(excinfo.value) == "no output from `myquota`"


def test_parse_storage_reports_unparsable_lines_as_warnings():
    cluster = make_cluster()
    text = fixture_text() + "/scratch/users/u100001   (quota unavailable)\n"
    assert cluster.parse_storage(results_of(text), TODAY) == FIXTURE_QUOTAS
    assert cluster._warnings == ["cannot parse myquota data line: '/scratch/users/u100001   (quota unavailable)'"]


# -- full collection -----------------------------------------------------------


def test_collect_runs_myquota_once_for_compute_and_storage():
    cluster = make_cluster(fixture_text())
    report = cluster.collect(["compute", "storage"], today=TODAY, now=NOW)

    assert cluster.runner.calls == [{"myquota": "myquota"}]
    assert report.name == "meluxina"
    assert report.system == "MeluXina"
    assert report.sections == ["compute", "storage"]
    assert report.collected_at == NOW
    assert report.budgets == [FIXTURE_REPORTED]
    assert report.storage == FIXTURE_QUOTAS
    assert report.jobs is None
    assert report.errors == []
    assert report.warnings == []
    assert report.failed_sections == []


def test_collect_all_sections():
    runner = FakeRunner({"myquota": fixture_text(), "jobs": read_fixture("slurm/squeue.txt")})
    report = MeluxinaCluster(make_config("meluxina"), runner=runner).collect(today=TODAY, now=NOW)

    assert runner.calls == [{"myquota": "myquota", "jobs": SQUEUE_COMMAND}]
    assert list(runner.calls[0]) == ["jobs", "myquota"]
    assert report.sections == ["compute", "storage", "jobs"]
    assert report.budgets == [FIXTURE_REPORTED]
    assert report.storage == FIXTURE_QUOTAS
    assert report.jobs is not None and len(report.jobs) == 5
    assert report.ok


@pytest.mark.parametrize("returncode", [0, 1])
def test_collect_rate_limited_reports_both_sections(returncode):
    result = CommandResult(stdout=read_fixture("meluxina/myquota_rate_limited.txt"), returncode=returncode)
    cluster = make_cluster(result)
    report = cluster.collect(["compute", "storage"], today=TODAY, now=NOW)

    assert cluster.runner.calls == [{"myquota": "myquota"}]
    assert report.errors == [f"compute: {RATE_LIMITED}", f"storage: {RATE_LIMITED}"]
    assert report.failed_sections == ["compute", "storage"]
    assert report.budgets == [] and report.storage == []
    assert not report.ok


def test_collect_rate_limited_on_stderr():
    result = CommandResult(stderr=RATE_LIMITED_LINE + "\n", returncode=1)
    report = make_cluster(result).collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.errors == [f"compute: {RATE_LIMITED}", f"storage: {RATE_LIMITED}"]
    assert report.failed_sections == ["compute", "storage"]


@pytest.mark.parametrize(
    "result, message",
    [
        (failed(124), "`myquota` timed out (time limit of 30 s reached)"),
        (failed(None), "`myquota` did not complete: no exit status"),
    ],
    ids=["timeout-124", "unfinished"],
)
def test_collect_timed_out_myquota(result, message):
    report = make_cluster(result, timeout=30).collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.errors == [f"compute: {message}", f"storage: {message}"]
    assert report.failed_sections == ["compute", "storage"]


def test_collect_failed_myquota_without_output():
    report = make_cluster(failed(1, stderr="myquota: service unavailable\n")).collect(
        ["compute", "storage"], today=TODAY, now=NOW
    )
    assert report.errors == [
        "compute: `myquota` failed (exit status 1): myquota: service unavailable",
        "storage: `myquota` failed (exit status 1): myquota: service unavailable",
    ]
    assert report.failed_sections == ["compute", "storage"]


def test_collect_myquota_without_output():
    report = make_cluster(ok("")).collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.errors == ["compute: `myquota` printed nothing", "storage: `myquota` printed nothing"]


def test_collect_malformed_compute_header_keeps_the_storage_section():
    text = strip_ansi(fixture_text()).replace("%used         Used", "%used         Usage", 1)
    report = make_cluster(text).collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.errors == ["compute: unexpected output of myquota (compute table header)"]
    assert report.failed_sections == ["compute"]
    assert report.budgets == []
    assert report.storage == FIXTURE_QUOTAS


def test_collect_merged_cells_fail_the_compute_section_only():
    text = strip_ansi(fixture_text()).replace("27    210", "100000210")
    report = make_cluster(text, projects=["p999999"]).collect(["compute", "storage"], today=TODAY, now=NOW)
    assert len(report.errors) == 1
    assert report.errors[0].startswith("compute: cannot parse the GPU allocation of p200001: 'p200001")
    assert report.failed_sections == ["compute"]
    assert report.storage == FIXTURE_QUOTAS[:1]
    # Projects are not reported as missing when a section could not be read.
    assert report.warnings == []


def test_collect_output_with_only_a_data_table():
    report = make_cluster(data_only_text()).collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.errors == []
    assert report.budgets == []
    assert report.storage == FIXTURE_QUOTAS


def test_collect_output_with_only_a_compute_table():
    report = make_cluster(compute_only_text()).collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.errors == []
    assert report.budgets == [FIXTURE_REPORTED]
    assert report.storage == []


def test_collect_unexpected_output_fails_both_sections():
    report = make_cluster("Service temporarily unavailable\n").collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.errors == [
        "compute: unexpected output of myquota (no compute table): 'Service temporarily unavailable'",
        "storage: unexpected output of myquota (no data table): 'Service temporarily unavailable'",
    ]
    assert report.failed_sections == ["compute", "storage"]


def test_collect_reports_unparsable_data_lines_as_warnings():
    text = fixture_text() + "/scratch/users/u100001   (quota unavailable)\n"
    report = make_cluster(text).collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.errors == []
    assert report.storage == FIXTURE_QUOTAS
    assert report.warnings == ["cannot parse myquota data line: '/scratch/users/u100001   (quota unavailable)'"]


def test_collect_uses_the_configured_end_date_case_insensitively():
    end = dt.date(2027, 6, 30)
    cluster = make_cluster(fixture_text(), project_end={"P200001": end})
    report = cluster.collect(["compute"], today=TODAY, now=NOW)
    assert report.budgets[0].end == end
    assert report.budgets[0].days_left(TODAY) == (end - TODAY).days


def test_collect_filters_projects_but_keeps_the_home_directory():
    cluster = make_cluster(fixture_text(), projects=["p999999"])
    report = cluster.collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.budgets == []
    assert report.storage == FIXTURE_QUOTAS[:1]
    assert report.warnings == ["project(s) not found on this cluster: p999999"]


def test_collect_selected_project_is_case_insensitive():
    cluster = make_cluster(fixture_text(), projects=["P200001"])
    report = cluster.collect(["compute", "storage"], today=TODAY, now=NOW)
    assert report.budgets == [FIXTURE_REPORTED]
    assert report.storage == FIXTURE_QUOTAS
    assert report.warnings == []
