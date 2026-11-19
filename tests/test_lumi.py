"""Tests of the LUMI interface (:mod:`hpcmon.clusters.lumi`).

The parsers are checked against real (anonymised) outputs of
``lumi-allocations``, ``lumi-ldap-projectinfo`` and ``lumi-workspaces`` and
against inline variants of these formats (several projects, decimals,
thousands separators, unusual rows).  The whole collection is exercised
through :class:`tests.conftest.FakeRunner`, never through SSH.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Union

import pytest

from hpcmon.clusters import create_cluster
from hpcmon.clusters.base import HOME_COMMAND, CollectionError
from hpcmon.clusters.lumi import (
    OVER_QUOTA_NOTE,
    PERIOD_OVER_NOTE,
    STORAGE_BILLING_NOTE,
    LumiCluster,
    parse_allocations,
    parse_days_left,
    parse_refresh_date,
    parse_workspaces,
)
from hpcmon.clusters.slurm import SQUEUE_COMMAND
from hpcmon.models import ComputeBudget, StorageQuota
from hpcmon.runner import CommandResult
from tests.conftest import NOW, TODAY, FakeRunner, failed, make_config, ok, read_fixture

KiB = 1024
MiB = 1024**2
GiB = 1024**3
TiB = 1024**4

PROJECT = "project_465000123"
OTHER_PROJECT = "project_465000456"
HOME = "/users/jdoe1234"
#: "Data was last refreshed at Wed Nov 18 19:24:19 2026" in the projectinfo fixture.
REFRESHED = dt.date(2026, 11, 18)
#: REFRESHED + 137 days ("137 day(s) of compute left" in the projectinfo fixture).
END = dt.date(2027, 4, 4)

EXPECTED_COMMANDS = {
    "allocations": "lumi-allocations",
    "projectinfo": "lumi-ldap-projectinfo",
    "workspaces": "lumi-workspaces",
    "home": HOME_COMMAND,
    "jobs": SQUEUE_COMMAND,
}


def fixture_budgets(end: Optional[dt.date] = None, note: Optional[str] = None, reported: bool = False):
    """The budgets of the allocations fixture; ``note`` is added to every budget.

    :param reported: Return the budgets as reported by the cluster, with the
        GPU-hours rate of the GPU budget (a LUMI GPU-hour is an MI250X-hour).
    """

    def join(*notes: Optional[str]) -> Optional[str]:
        return "; ".join(n for n in notes if n) or None

    return [
        ComputeBudget(project=PROJECT, resource="CPU", unit="core-h", total=50000.0, used=0.0, end=end, note=note),
        ComputeBudget(
            project=PROJECT,
            resource="GPU",
            unit="GPU-h",
            total=12000.0,
            used=0.0,
            end=end,
            note=note,
            gpu_hours_per_unit=1.0 if reported else None,
        ),
        ComputeBudget(
            project=PROJECT,
            resource="Storage",
            unit="TB-h",
            total=200000.0,
            used=0.0,
            end=end,
            note=join(STORAGE_BILLING_NOTE, note),
        ),
    ]


def fixture_quotas(home_name: str = HOME):
    return [
        StorageQuota(
            name=home_name, project=None, used_bytes=20 * KiB, limit_bytes=25 * GiB, files_used=5, files_limit=150_000
        ),
        StorageQuota(
            name="/project/project_465000123",
            project=PROJECT,
            used_bytes=4 * KiB,
            limit_bytes=100 * GiB,
            files_used=1,
            files_limit=200_000,
        ),
        StorageQuota(
            name="/scratch/project_465000123",
            project=PROJECT,
            used_bytes=4 * KiB,
            limit_bytes=20 * TiB,
            files_used=1,
            files_limit=4_000_000,
        ),
        StorageQuota(
            name="/flash/project_465000123",
            project=PROJECT,
            used_bytes=4 * KiB,
            limit_bytes=1 * TiB,
            files_used=1,
            files_limit=3_000_000,
        ),
    ]


def fixture_results(**overrides: Union[str, CommandResult, None]) -> Dict[str, Union[str, CommandResult]]:
    """The fixture outputs of all LUMI commands; ``None`` removes a command."""
    results: Dict[str, Union[str, CommandResult, None]] = {
        "allocations": read_fixture("lumi/allocations.txt"),
        "projectinfo": read_fixture("lumi/projectinfo.txt"),
        "workspaces": read_fixture("lumi/workspaces.txt"),
        "home": HOME + "\n",
        "jobs": read_fixture("slurm/squeue.txt"),
    }
    results.update(overrides)
    return {key: value for key, value in results.items() if value is not None}


def make_lumi(results=None, **config) -> LumiCluster:
    return LumiCluster(make_config("lumi", **config), runner=FakeRunner(results))


def collecting_lumi(results=None, **config):
    """A LUMI cluster whose :meth:`warn` messages go to the returned list."""
    cluster = make_lumi(results, **config)
    warnings: List[str] = []
    cluster.warn = warnings.append  # type: ignore[method-assign]
    return cluster, warnings


# -- inline variants of the real formats --------------------------------------

ALLOCATIONS_TWO_PROJECTS = """\
Data updated: 2026-11-18 19:24:19
Project             |                    CPU (used/allocated)|               GPU (used/allocated)|           Storage (used/allocated)
--------------------------------------------------------------------------------------------------------------------------------------
project_465000123   |     1,234.5/100,000   (1.2%) core/hours|      12.25/12000   (0.1%) gpu/hours|          0/0   (0.0%) TB/hours
project_465000456   |              0/0   (0.0%) core/hours|       500/1,000   (50.0%) gpu/hours|     10.5/200,000   (0.0%) TB/hours
"""

PROJECTINFO_TWO_PROJECTS = """\
    15 day(s) of compute left

Information for project_465000123:
  - General information:
    - 31% of the project time has passed
    137 day(s) of compute left
  - 227 day(s) left until data removal
Information for project_465000456
    -3 DAY(S) OF COMPUTE LEFT
Information for project_465000789:
  - 227 day(s) left until data removal
"""

WORKSPACES_PROJECT_THEN_USER = """\
Information for project project_465000456 - EHPC-DEV-D / Another Project

- Storage information:
  - Disk quota (live from `lfs quota`):
                                  file quota:    0.0% used (    9    of   100K  /    101K   soft/hard)
    - /project/project_465000456: block quota:  12.5% used ( 6.25GiB of    50GiB/     51GiB soft/hard)
                                  file quota:    1.2% used (  1.2K    of   100K  /    101K   soft/hard)
    - /scratch/project_465000456: block quota:   0.0% used (      0 of    50TiB/50.0010TiB soft/hard)

--------------------------------------------------------------------------------------------------------

Information for userid jdoe1234:
  - Disk quota home directory (live from `lfs quota`):
    - /users/jdoe1234: BLOCK QUOTA:   0.0% used ( 1.5MiB of    20GiB/     22GiB soft/hard)
    - File quota:    0.3% used (  321    of   100K  /    101K   soft/hard)
    - Block quota:   0.0% used (   20KiB of    20GiB/     22GiB soft/hard)
"""


def one_cell_table(cell: str) -> str:
    """A ``lumi-allocations`` table with a single project and resource."""
    return (
        "Project             |               GPU (used/allocated)\n"
        "---------------------------------------------------------\n"
        f"project_465000123   |{cell}\n"
    )


def projectinfo_with_days(days: int, refreshed: Optional[str] = None) -> str:
    """A ``lumi-ldap-projectinfo`` output for :data:`PROJECT` with ``days`` left."""
    refresh_line = f"- Data was last refreshed at {refreshed}\n" if refreshed else ""
    return f"Information for {PROJECT}:\n\n{refresh_line}- General information:\n    {days} day(s) of compute left\n"


# -- cluster definition -------------------------------------------------------


def test_cluster_identity_and_registry():
    runner = FakeRunner()
    cluster = create_cluster(make_config("mylumi", type="lumi"), runner=runner)

    assert isinstance(cluster, LumiCluster)
    assert cluster.runner is runner
    assert cluster.name == "mylumi"
    assert LumiCluster.type_name == "lumi"
    assert LumiCluster.display_name == "LUMI"


def test_notes_wording():
    assert STORAGE_BILLING_NOTE == "storage billing"
    assert PERIOD_OVER_NOTE == "no compute days left"
    assert OVER_QUOTA_NOTE == "over quota"


def test_commands_of_every_section():
    cluster = make_lumi()

    assert cluster.compute_commands() == {"allocations": "lumi-allocations", "projectinfo": "lumi-ldap-projectinfo"}
    assert cluster.storage_commands() == {"workspaces": "lumi-workspaces", "home": HOME_COMMAND}
    assert cluster.jobs_commands() == {"jobs": SQUEUE_COMMAND}
    assert cluster.commands() == EXPECTED_COMMANDS
    # The commands run in the order jobs -> compute -> storage.
    assert list(cluster.commands()) == ["jobs", "allocations", "projectinfo", "workspaces", "home"]


# -- lumi-allocations ---------------------------------------------------------


def test_parse_allocations_real_output():
    warnings: List[str] = []

    assert parse_allocations(read_fixture("lumi/allocations.txt"), warn=warnings.append) == fixture_budgets()
    assert warnings == []


def test_parse_allocations_several_projects_with_decimals_and_thousands_separators():
    assert parse_allocations(ALLOCATIONS_TWO_PROJECTS) == [
        ComputeBudget(project=PROJECT, resource="CPU", unit="core-h", total=100_000.0, used=1234.5),
        ComputeBudget(project=PROJECT, resource="GPU", unit="GPU-h", total=12000.0, used=12.25),
        ComputeBudget(project=OTHER_PROJECT, resource="GPU", unit="GPU-h", total=1000.0, used=500.0),
        ComputeBudget(
            project=OTHER_PROJECT,
            resource="Storage",
            unit="TB-h",
            total=200000.0,
            used=10.5,
            note=STORAGE_BILLING_NOTE,
        ),
    ]


def test_parse_allocations_skips_empty_allocations_but_keeps_usage_without_allocation():
    text = (
        "Project | CPU (used/allocated) | GPU (used/allocated) | Storage (used/allocated)\n"
        "project_1 | 0/0 (0.0%) core/hours | 0.0/0.00 gpu/hours | 5/0 TB/hours\n"
    )
    warnings: List[str] = []

    assert parse_allocations(text, warn=warnings.append) == [
        ComputeBudget(project="project_1", resource="Storage", unit="TB-h", total=0.0, used=5.0, note="storage billing"),
    ]
    assert warnings == []


def test_parse_allocations_warns_about_cells_that_are_not_amounts():
    text = (
        "Project | CPU (used/allocated) | GPU (used/allocated) | Storage (used/allocated) | Other\n"
        "project_1 | n/a | - |    3/10   (30.0%) TB/hours |   \n"
        "project_2 | pending | 1 of 2 gpu/hours | 7/ | 2 core hours\n"
        "project_3 | -- |  --  | - |\n"
    )
    warnings: List[str] = []

    assert parse_allocations(text, warn=warnings.append) == [
        ComputeBudget(
            project="project_1", resource="Storage", unit="TB-h", total=10.0, used=3.0, note=STORAGE_BILLING_NOTE
        ),
    ]
    # Empty cells and the '-' / '--' placeholders are not worth a warning.
    assert warnings == [
        "cannot parse the CPU allocation of project_1: 'n/a'",
        "cannot parse the CPU allocation of project_2: 'pending'",
        "cannot parse the GPU allocation of project_2: '1 of 2 gpu/hours'",
        "cannot parse the Storage allocation of project_2: '7/'",
        "cannot parse the Other allocation of project_2: '2 core hours'",
    ]


def test_parse_allocations_ignores_unparsable_cells_without_warn_callback():
    text = "Project | CPU (used/allocated) | GPU (used/allocated)\nproject_1 | n/a | 1/2 gpu/hours\n"

    assert parse_allocations(text) == [
        ComputeBudget(project="project_1", resource="GPU", unit="GPU-h", total=2.0, used=1.0),
    ]


def test_parse_allocations_malformed_number_in_a_cell_is_a_warning():
    text = (
        "Project | CPU (used/allocated) | GPU (used/allocated)\n"
        "project_1 | 1.2.3/10 core/hours | 1/2 gpu/hours\n"
    )
    warnings: List[str] = []

    budgets = parse_allocations(text, warn=warnings.append)

    assert budgets == [ComputeBudget(project="project_1", resource="GPU", unit="GPU-h", total=2.0, used=1.0)]
    assert len(warnings) == 1
    assert warnings[0].startswith("cannot parse the CPU allocation of project_1")


def test_parse_allocations_skips_separator_blank_and_projectless_rows():
    text = (
        "Data updated: 2026-11-18 19:24:19\n"
        "PROJECT | CPU (used/allocated) | GPU (used/allocated)\n"
        "==========|==========|==========\n"
        "----------|----------|----------\n"
        "-=-=-=-=-=|\n"
        "          | 5/10 core/hours | 1/2 gpu/hours\n"
        "\n"
        "project_1 | 5/10 core/hours | 1/2 gpu/hours\n"
        "project_2 | 3/4 core/hours\n"
        "Footer without any table delimiter\n"
    )
    warnings: List[str] = []

    assert parse_allocations(text, warn=warnings.append) == [
        ComputeBudget(project="project_1", resource="CPU", unit="core-h", total=10.0, used=5.0),
        ComputeBudget(project="project_1", resource="GPU", unit="GPU-h", total=2.0, used=1.0),
        ComputeBudget(project="project_2", resource="CPU", unit="core-h", total=4.0, used=3.0),
    ]
    assert warnings == []


def test_parse_allocations_header_without_rows_means_no_budgets():
    text = "Project | CPU (used/allocated) | GPU (used/allocated)\n-------------------------\n"

    assert parse_allocations(text) == []


@pytest.mark.parametrize(
    "cell, unit, note",
    [
        ("   10/100   (10.0%) core/hours", "core-h", None),
        ("   10/100   (10.0%) gpu/hours", "GPU-h", None),
        ("   10/100   (10.0%) GPU/Hours", "GPU-h", None),
        ("   10/100   (10.0%) TB/hours", "TB-h", STORAGE_BILLING_NOTE),
        ("   10/100   (10.0%) tb/HOURS", "TB-h", STORAGE_BILLING_NOTE),
        ("   10/100   (10.0%) foo/hours", "foo-h", None),
        ("   10/100   (10.0%) Node/hours", "node-h", None),
        ("   10/100   (10.0%) core/days", "core/days", None),
        ("   10/100   (10.0%)", "h", None),
        ("   10/100", "h", None),
        ("10 / 100 ( 10.0 % ) core/hours  ", "core-h", None),
        # The percentage group may hold anything but a closing parenthesis.
        ("   10/100   () core/hours", "core-h", None),
        ("   10/100   (n/a) gpu/hours", "GPU-h", None),
        ("   10/100 gpu/hours", "GPU-h", None),
    ],
)
def test_parse_allocations_unit_names(cell, unit, note):
    assert parse_allocations(one_cell_table(cell)) == [
        ComputeBudget(project=PROJECT, resource="GPU", unit=unit, total=100.0, used=10.0, note=note),
    ]


@pytest.mark.parametrize(
    "cell",
    [
        "   10/100   (10.0%) khours",  # the unit must look like "<what>/<time>"
        "   10/100   (10.0%) gpu/",
        "   10/100   (10.0%) gpu/hours extra",
        "   10/100   (10.0% gpu/hours",
        "   unavailable",
    ],
)
def test_parse_allocations_rows_without_any_parsable_cell_are_an_error(cell):
    warnings: List[str] = []

    with pytest.raises(CollectionError) as excinfo:
        parse_allocations(one_cell_table(cell), warn=warnings.append)

    assert str(excinfo.value) == "unexpected output of lumi-allocations (no parsable allocation)"
    assert warnings == [f"cannot parse the GPU allocation of {PROJECT}: {cell.strip()!r}"]


@pytest.mark.parametrize("cell", ["   -", " -- ", "    "])
def test_parse_allocations_rows_with_only_placeholders_are_an_error_without_warning(cell):
    warnings: List[str] = []

    with pytest.raises(CollectionError, match=r"^unexpected output of lumi-allocations \(no parsable allocation\)$"):
        parse_allocations(one_cell_table(cell), warn=warnings.append)

    assert warnings == []


@pytest.mark.parametrize(
    "text",
    [
        "",
        "  \n\n\t\n",
        "No projects found for user jdoe1234\n",
        "Data updated: 2026-11-18 19:24:19\nYou are not a member of any project (NO PROJECT)\n",
    ],
)
def test_parse_allocations_without_any_project(text):
    assert parse_allocations(text) == []


@pytest.mark.parametrize(
    "text",
    [
        "bash: lumi-allocations: command not found\n",
        "Account | CPU (used/allocated)\nproject_1 | 5/10 core/hours\n",
    ],
)
def test_parse_allocations_unexpected_output_is_an_error(text):
    with pytest.raises(CollectionError) as excinfo:
        parse_allocations(text)

    assert str(excinfo.value) == "unexpected output of lumi-allocations (no table header)"


# -- lumi-ldap-projectinfo ----------------------------------------------------


def test_parse_days_left_real_output():
    assert parse_days_left(read_fixture("lumi/projectinfo.txt")) == {PROJECT: 137}


def test_parse_days_left_several_projects():
    # Days before any project header are ignored; a project without the line
    # ("project_465000789") is absent; negative values are kept.
    assert parse_days_left(PROJECTINFO_TWO_PROJECTS) == {PROJECT: 137, OTHER_PROJECT: -3}


@pytest.mark.parametrize(
    "header",
    [
        f"Information for {PROJECT}:",
        f"  Information for {PROJECT}:  ",
        f"Information for project {PROJECT} - EHPC-DEV-D / An Example Project Title",
        f"Information for project   {PROJECT}:",
        f"Information for project {PROJECT}",
    ],
)
def test_parse_days_left_header_variants(header):
    assert parse_days_left(f"{header}\n    12 day(s) of compute left\n") == {PROJECT: 12}


@pytest.mark.parametrize(
    "other_header",
    [
        "Information for: all users",
        "Information for jdoe-1234:",
        "Information for",
        "Information for project-x:",
    ],
)
def test_parse_days_left_other_information_block_is_ignored(other_header):
    # An unrecognised "Information for" header resets the project: its days
    # are not attributed to the previous project.
    text = f"Information for {PROJECT}:\n    137 day(s) of compute left\n{other_header}\n    5 day(s) of compute left\n"

    assert parse_days_left(text) == {PROJECT: 137}


@pytest.mark.parametrize(
    "text",
    [
        "",
        "    137 day(s) of compute left\n",
        pytest.param(
            "Information for userid jdoe1234:\n    137 day(s) of compute left\n",
            id="userid-block",
        ),
        "Information for project_465000123:\n  - 227 day(s) left until data removal\n",
    ],
)
def test_parse_days_left_without_usable_information(text):
    assert parse_days_left(text) == {}


@pytest.mark.parametrize(
    "text, expected",
    [
        (read_fixture("lumi/projectinfo.txt"), REFRESHED),
        ("- Data was last refreshed at Wed Nov  4 01:02:03 2026\n", dt.date(2026, 11, 4)),
        ("Data was last refreshed at Thu Dec 31 23:59:59 2026   \nmore\n", dt.date(2026, 12, 31)),
        ("", None),
        ("Information for project_465000123:\n    137 day(s) of compute left\n", None),
        ("- Data was last refreshed at yesterday\n", None),
        ("- Data was last refreshed at 2026-11-18 19:24:19\n", None),
        ("- Data was last refreshed at Wed Nov 31 19:24:19 2026\n", None),
    ],
)
def test_parse_refresh_date(text, expected):
    assert parse_refresh_date(text) == expected


# -- lumi-workspaces ----------------------------------------------------------


def test_parse_workspaces_real_output_names_home_quota_after_home_directory():
    warnings: List[str] = []

    assert parse_workspaces(read_fixture("lumi/workspaces.txt"), home=HOME, warn=warnings.append) == fixture_quotas()
    assert warnings == []


def test_parse_workspaces_without_home_directory_names_it_home():
    assert parse_workspaces(read_fixture("lumi/workspaces.txt")) == fixture_quotas(home_name="home")


def test_parse_workspaces_project_section_followed_by_user_section():
    quotas = parse_workspaces(WORKSPACES_PROJECT_THEN_USER, home="/home/other")

    assert quotas == [
        # The file quota printed before any block quota is not attached.
        StorageQuota(
            name="/project/project_465000456",
            project=OTHER_PROJECT,
            used_bytes=int(6.25 * GiB),
            limit_bytes=50 * GiB,
            files_used=1200,
            files_limit=100_000,
        ),
        StorageQuota(
            name="/scratch/project_465000456", project=OTHER_PROJECT, used_bytes=0, limit_bytes=50 * TiB
        ),
        # The user section resets the project; an explicit path wins over home.
        StorageQuota(
            name="/users/jdoe1234",
            project=None,
            used_bytes=int(1.5 * MiB),
            limit_bytes=20 * GiB,
            files_used=321,
            files_limit=100_000,
        ),
        StorageQuota(name="/home/other", project=None, used_bytes=20 * KiB, limit_bytes=20 * GiB),
    ]


def test_parse_workspaces_two_project_sections_back_to_back():
    text = """\
Information for project project_465000123 - EHPC-DEV-D / First Project

  - Disk quota (live from `lfs quota`):
    - /project/project_465000123: block quota:   2.0% used (    1GiB of    50GiB/     51GiB soft/hard)
                                  file quota:    0.1% used (  100    of   100K  /    101K   soft/hard)
    - /scratch/project_465000123: block quota:   0.0% used (  512KiB of    50TiB/50.0010TiB soft/hard)
                                  file quota:    0.0% used (   20    of     2M  /2.00100M   soft/hard)

Information for project project_465000456 - EHPC-DEV-D / Second Project

  - Disk quota (live from `lfs quota`):
    - /project/project_465000456: block quota:   0.0% used (    4KiB of    20GiB/     21GiB soft/hard)
                                  file quota:    0.0% used (    1    of    50K  /     51K   soft/hard)
"""

    assert parse_workspaces(text, home=HOME) == [
        StorageQuota(
            name="/project/project_465000123",
            project=PROJECT,
            used_bytes=GiB,
            limit_bytes=50 * GiB,
            files_used=100,
            files_limit=100_000,
        ),
        StorageQuota(
            name="/scratch/project_465000123",
            project=PROJECT,
            used_bytes=512 * KiB,
            limit_bytes=50 * TiB,
            files_used=20,
            files_limit=2_000_000,
        ),
        StorageQuota(
            name="/project/project_465000456",
            project=OTHER_PROJECT,
            used_bytes=4 * KiB,
            limit_bytes=20 * GiB,
            files_used=1,
            files_limit=50_000,
        ),
    ]


WORKSPACES_OVER_QUOTA = """\
Information for userid jdoe1234:
  - Disk quota home directory (live from `lfs quota`):
    - Block quota: 110.0% used (    22GiB* of    20GiB/     22GiB soft/hard)
    - File quota:    0.0% used (    5    of   100K  /    101K   soft/hard)

Information for project project_465000123 - EHPC-DEV-D / Example

  - Disk quota (live from `lfs quota`):
    - /project/project_465000123: block quota: 102.0% used (   51GiB* of    50GiB/     51GiB soft/hard)
                                  file quota: 200.0% used ( 200K*   of   100K  /    101K   soft/hard)
    - /scratch/project_465000123: block quota:   0.0% used (  512KiB of    50TiB/50.0010TiB soft/hard)
                                  file quota: 150.0% used (  3M*    of     2M  /2.00100M   soft/hard)
    - /flash/project_465000123:   block quota:   0.0% used (    4KiB of     2TiB/2.00098TiB soft/hard)
                                  file quota:    0.0% used (    1    of     1M  /1.00100M   soft/hard)
"""


def test_parse_workspaces_marks_quotas_over_their_limit():
    warnings: List[str] = []

    quotas = parse_workspaces(WORKSPACES_OVER_QUOTA, home=HOME, warn=warnings.append)

    assert warnings == []
    assert quotas == [
        # The size over quota: the asterisk is dropped from the figure.
        StorageQuota(
            name=HOME,
            project=None,
            used_bytes=22 * GiB,
            limit_bytes=20 * GiB,
            files_used=5,
            files_limit=100_000,
            note=OVER_QUOTA_NOTE,
        ),
        # Both the size and the file count over quota: the note is not repeated.
        StorageQuota(
            name="/project/project_465000123",
            project=PROJECT,
            used_bytes=51 * GiB,
            limit_bytes=50 * GiB,
            files_used=200_000,
            files_limit=100_000,
            note=OVER_QUOTA_NOTE,
        ),
        # Only the file count over quota.
        StorageQuota(
            name="/scratch/project_465000123",
            project=PROJECT,
            used_bytes=512 * KiB,
            limit_bytes=50 * TiB,
            files_used=3_000_000,
            files_limit=2_000_000,
            note=OVER_QUOTA_NOTE,
        ),
        StorageQuota(
            name="/flash/project_465000123",
            project=PROJECT,
            used_bytes=4 * KiB,
            limit_bytes=2 * TiB,
            files_used=1,
            files_limit=1_000_000,
        ),
    ]


def test_parse_workspaces_asterisk_on_the_limit_only_is_not_over_quota():
    text = (
        "    - /project/project_465000123: block quota:   1.0% used (    1GiB of  50GiB*/  51GiB soft/hard)\n"
        "                                  file quota:    0.0% used (    1    of  100K* /  101K soft/hard)\n"
    )

    assert parse_workspaces(text) == [
        StorageQuota(
            name="/project/project_465000123",
            used_bytes=GiB,
            limit_bytes=50 * GiB,
            files_used=1,
            files_limit=100_000,
        ),
    ]


def test_parse_workspaces_skips_quotas_with_unparsable_sizes():
    text = """\
Information for project project_465000123 - EHPC-DEV-D / Example

  - Disk quota (live from `lfs quota`):
    - /project/project_465000123: block quota:   0.0% used (    4KiB of    50GiB/     51GiB soft/hard)
                                  file quota:    0.0% used (    1    of   100K  /    101K   soft/hard)
    - /scratch/project_465000123: block quota:   0.0% used (    4XiB of    50TiB/50.0010TiB soft/hard)
                                  file quota:    0.0% used (    7    of     2M  /2.00100M   soft/hard)
    - /flash/project_465000123:   block quota:   0.0% used (    4KiB of    lots/2.00098TiB soft/hard)
                                  file quota:    0.0% used (    8    of     1M  /1.00100M   soft/hard)
"""
    warnings: List[str] = []

    quotas = parse_workspaces(text, warn=warnings.append)

    # The file lines following a skipped quota are not attached to any quota.
    assert quotas == [
        StorageQuota(
            name="/project/project_465000123",
            project=PROJECT,
            used_bytes=4 * KiB,
            limit_bytes=50 * GiB,
            files_used=1,
            files_limit=100_000,
        ),
    ]
    assert warnings == [
        "cannot parse the quota of /scratch/project_465000123: invalid size unit in '4XiB'",
        "cannot parse the quota of /flash/project_465000123: invalid size: 'lots'",
    ]


def test_parse_workspaces_unparsable_home_size_names_the_home_quota():
    text = "    - Block quota:   0.0% used (   20KiB of    20QiB/     22GiB soft/hard)\n"
    warnings: List[str] = []

    assert parse_workspaces(text, home=HOME, warn=warnings.append) == []
    assert warnings == [f"cannot parse the quota of {HOME}: invalid size unit in '20QiB'"]


def test_parse_workspaces_keeps_quotas_with_unparsable_file_counts():
    text = (
        "    - /project/project_465000123: block quota:   0.0% used (    4KiB of    50GiB/     51GiB soft/hard)\n"
        "                                  file quota:    0.0% used (    5Z    of   100K  /    101K   soft/hard)\n"
        "    - /scratch/project_465000123: block quota:   0.0% used (    4KiB of    50TiB/50.0010TiB soft/hard)\n"
        "                                  file quota:    0.0% used (    5    of   many  /    101K   soft/hard)\n"
    )
    warnings: List[str] = []

    quotas = parse_workspaces(text, warn=warnings.append)

    assert quotas == [
        StorageQuota(name="/project/project_465000123", used_bytes=4 * KiB, limit_bytes=50 * GiB),
        StorageQuota(name="/scratch/project_465000123", used_bytes=4 * KiB, limit_bytes=50 * TiB),
    ]
    assert warnings == [
        "cannot parse the file quota of /project/project_465000123: invalid count unit in '5Z'",
        "cannot parse the file quota of /scratch/project_465000123: invalid count: 'many'",
    ]


def test_parse_workspaces_unparsable_lines_without_warn_callback_are_dropped():
    text = (
        "    - /project/project_465000123: block quota:   0.0% used (    4XiB of    50GiB/     51GiB soft/hard)\n"
        "    - /scratch/project_465000123: block quota:   0.0% used (    4KiB of    50TiB/50.0010TiB soft/hard)\n"
        "                                  file quota:    0.0% used (    5Z    of   100K  /    101K   soft/hard)\n"
    )

    assert parse_workspaces(text) == [
        StorageQuota(name="/scratch/project_465000123", used_bytes=4 * KiB, limit_bytes=50 * TiB),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Information for userid jdoe1234:\n  - No disk quota information available\n",
        "    - File quota:    0.0% used (    5    of   100K  /    101K   soft/hard)\n",
    ],
)
def test_parse_workspaces_without_block_quotas(text):
    assert parse_workspaces(text, home=HOME) == []


def test_parse_workspaces_projectinfo_style_paths_have_no_project():
    # lumi-ldap-projectinfo prints the same quota lines, but under a header
    # that is not a lumi-workspaces section header.
    quotas = parse_workspaces(read_fixture("lumi/projectinfo.txt"))

    assert [(quota.name, quota.project) for quota in quotas] == [
        ("/project/project_465000123", None),
        ("/scratch/project_465000123", None),
        ("/flash/project_465000123", None),
    ]


# -- parse_compute / parse_storage --------------------------------------------


def test_parse_compute_counts_days_from_the_refresh_date_not_from_today():
    cluster, warnings = collecting_lumi()
    results = {
        "allocations": ok(read_fixture("lumi/allocations.txt")),
        "projectinfo": ok(read_fixture("lumi/projectinfo.txt")),
    }

    budgets = cluster.parse_compute(results, dt.date(2026, 1, 1))

    assert END == REFRESHED + dt.timedelta(days=137)
    assert budgets == fixture_budgets(end=END)
    assert warnings == []


@pytest.mark.parametrize(
    "refreshed",
    [None, "yesterday", "Wed Nov 31 19:24:19 2026"],
    ids=["no-refresh-line", "invalid-refresh-date", "impossible-refresh-date"],
)
def test_parse_compute_without_usable_refresh_date_counts_days_from_today(refreshed):
    cluster, warnings = collecting_lumi()
    today = dt.date(2026, 1, 1)
    results = {
        "allocations": ok(read_fixture("lumi/allocations.txt")),
        "projectinfo": ok(projectinfo_with_days(92, refreshed)),
    }

    budgets = cluster.parse_compute(results, today)

    assert budgets == fixture_budgets(end=dt.date(2026, 4, 3))
    assert warnings == []


@pytest.mark.parametrize(
    "days, note",
    [(1, None), (0, PERIOD_OVER_NOTE), (-3, PERIOD_OVER_NOTE)],
)
def test_parse_compute_notes_projects_without_compute_days_left(days, note):
    cluster = make_lumi()
    results = {
        "allocations": ok(read_fixture("lumi/allocations.txt")),
        "projectinfo": ok(projectinfo_with_days(days, "Wed Nov 18 19:24:19 2026")),
    }

    budgets = cluster.parse_compute(results, dt.date(2030, 1, 1))

    assert budgets == fixture_budgets(end=REFRESHED + dt.timedelta(days=days), note=note)
    if note is not None:
        assert budgets[2].note == "storage billing; no compute days left"


def test_parse_compute_failing_projectinfo_warns_and_returns_budgets_without_end():
    cluster, warnings = collecting_lumi()
    results = {"allocations": ok(read_fixture("lumi/allocations.txt")), "projectinfo": failed(1)}

    assert cluster.parse_compute(results, TODAY) == fixture_budgets()
    assert warnings == ["end of projects unknown: `projectinfo` failed (exit status 1)"]


def test_parse_compute_reports_unparsable_allocations_through_warn():
    cluster, warnings = collecting_lumi()
    allocations = (
        "Project | CPU (used/allocated) | GPU (used/allocated)\n"
        "project_465000123 | 0/50000 (0.0%) core/hours | soon\n"
    )
    results = {"allocations": ok(allocations), "projectinfo": ok(read_fixture("lumi/projectinfo.txt"))}

    budgets = cluster.parse_compute(results, TODAY)

    assert budgets == [fixture_budgets(end=END)[0]]
    assert warnings == ["cannot parse the GPU allocation of project_465000123: 'soon'"]


def test_parse_compute_missing_allocations_is_an_error():
    cluster = make_lumi()

    with pytest.raises(CollectionError, match="no output from `allocations`"):
        cluster.parse_compute({"projectinfo": ok(read_fixture("lumi/projectinfo.txt"))}, TODAY)


@pytest.mark.parametrize(
    "home, name",
    [
        (ok(HOME + "\n"), HOME),
        (ok("  /users/jdoe1234  \n"), HOME),
        (ok(""), "home"),
        (ok(" \n"), "home"),
        (failed(1, stderr="printf: error"), "home"),
        (failed(None), "home"),
        (None, "home"),
    ],
)
def test_parse_storage_names_home_quota(home, name):
    cluster = make_lumi()
    results = {"workspaces": ok(read_fixture("lumi/workspaces.txt"))}
    if home is not None:
        results["home"] = home

    assert cluster.parse_storage(results, TODAY) == fixture_quotas(home_name=name)


def test_parse_storage_reports_unparsable_quotas_through_warn():
    cluster, warnings = collecting_lumi()
    workspaces = read_fixture("lumi/workspaces.txt").replace("4KiB of     1TiB", "4KiB of     1YiB")
    results = {"workspaces": ok(workspaces), "home": ok(HOME)}

    assert cluster.parse_storage(results, TODAY) == fixture_quotas()[:3]
    assert warnings == ["cannot parse the quota of /flash/project_465000123: invalid size unit in '1YiB'"]


def test_parse_storage_failing_workspaces_is_an_error():
    cluster = make_lumi()
    results = {"workspaces": failed(1, stderr="lfs: error"), "home": ok(HOME)}

    with pytest.raises(CollectionError, match=r"^`workspaces` failed \(exit status 1\): lfs: error$"):
        cluster.parse_storage(results, TODAY)


# -- collect() end to end -----------------------------------------------------


def test_collect_end_to_end_with_real_outputs():
    runner = FakeRunner(fixture_results())
    cluster = LumiCluster(make_config("lumi"), runner=runner)

    report = cluster.collect(today=TODAY, now=NOW)

    assert runner.calls == [EXPECTED_COMMANDS]
    assert report.name == "lumi"
    assert report.system == "LUMI"
    assert report.collected_at == NOW
    assert report.sections == ["compute", "storage", "jobs"]
    assert report.errors == []
    assert report.failed_sections == []
    assert report.warnings == []
    assert report.budgets == fixture_budgets(end=END, reported=True)
    assert report.budgets[2].note == "storage billing"
    assert [budget.days_left(TODAY) for budget in report.budgets] == [137, 137, 137]
    assert report.storage == fixture_quotas()
    assert [job.job_id for job in report.jobs] == ["1234567", "1234570", "1234568", "1234569_[1-10]", "1234571"]


def test_collect_end_date_does_not_depend_on_the_day_of_collection():
    cluster = make_lumi(fixture_results())

    report = cluster.collect(sections=("compute",), today=dt.date(2026, 12, 1), now=NOW)

    assert report.budgets == fixture_budgets(end=END, reported=True)
    assert [budget.days_left(dt.date(2026, 12, 1)) for budget in report.budgets] == [124, 124, 124]


def test_collect_storage_only_runs_storage_commands():
    runner = FakeRunner(fixture_results())
    cluster = LumiCluster(make_config("lumi"), runner=runner)

    report = cluster.collect(sections=("storage",), today=TODAY, now=NOW)

    assert runner.calls == [{"workspaces": "lumi-workspaces", "home": HOME_COMMAND}]
    assert report.ok
    assert report.sections == ["storage"]
    assert report.budgets == []
    assert report.jobs is None
    assert report.storage == fixture_quotas()


@pytest.mark.parametrize(
    "projectinfo, warning",
    [
        (
            failed(1, stderr="Error: cannot contact LDAP server\n"),
            "end of projects unknown: `lumi-ldap-projectinfo` failed (exit status 1): Error: cannot contact LDAP server",
        ),
        (
            failed(1, stdout="partial output\nldap: timeout\n\n"),
            "end of projects unknown: `lumi-ldap-projectinfo` failed (exit status 1): ldap: timeout",
        ),
        (failed(2), "end of projects unknown: `lumi-ldap-projectinfo` failed (exit status 2)"),
        (failed(None), "end of projects unknown: `lumi-ldap-projectinfo` did not complete: no exit status"),
        (
            CommandResult(returncode=None, reason="the SSH session timed out after 195 s"),
            "end of projects unknown: `lumi-ldap-projectinfo` did not complete: "
            "the SSH session timed out after 195 s",
        ),
        (
            failed(124),
            "end of projects unknown: `lumi-ldap-projectinfo` timed out (time limit of 180 s reached)",
        ),
        (None, "end of projects unknown: no output from `lumi-ldap-projectinfo`"),
    ],
)
def test_collect_without_projectinfo_warns_and_keeps_budgets(projectinfo, warning):
    cluster = make_lumi(fixture_results(projectinfo=projectinfo))

    report = cluster.collect(sections=("compute",), today=TODAY, now=NOW)

    assert report.errors == []
    assert report.failed_sections == []
    assert report.warnings == [warning]
    assert report.budgets == fixture_budgets(reported=True)


def test_collect_projectinfo_timeout_message_uses_the_configured_limit():
    cluster = make_lumi(fixture_results(projectinfo=failed(124)), timeout=42.5)

    report = cluster.collect(sections=("compute",), today=TODAY, now=NOW)

    assert report.warnings == [
        "end of projects unknown: `lumi-ldap-projectinfo` timed out (time limit of 42.5 s reached)"
    ]


def test_collect_without_projectinfo_falls_back_to_configured_end():
    configured = dt.date(2027, 6, 30)
    cluster = make_lumi(fixture_results(projectinfo=failed(1)), project_end={"PROJECT_465000123": configured})

    report = cluster.collect(sections=("compute",), today=TODAY, now=NOW)

    assert report.budgets == fixture_budgets(end=configured, reported=True)
    assert len(report.warnings) == 1


def test_collect_projectinfo_without_days_left_gives_no_end_and_no_warning():
    projectinfo = "Information for project_465000123:\n  - 227 day(s) left until data removal\n"
    cluster = make_lumi(fixture_results(projectinfo=projectinfo))

    report = cluster.collect(sections=("compute",), today=TODAY, now=NOW)

    assert report.warnings == []
    assert report.errors == []
    assert report.budgets == fixture_budgets(reported=True)


def test_collect_several_projects_reported_end_wins_over_configured_end():
    configured = dt.date(2027, 6, 30)
    cluster = make_lumi(
        fixture_results(allocations=ALLOCATIONS_TWO_PROJECTS, projectinfo=PROJECTINFO_TWO_PROJECTS),
        project_end={PROJECT: dt.date(2030, 1, 1), OTHER_PROJECT: configured},
    )

    report = cluster.collect(sections=("compute",), today=TODAY, now=NOW)

    # Without a refresh date in the output, the days are counted from today.
    assert [(b.project, b.resource, b.end, b.note) for b in report.budgets] == [
        (PROJECT, "CPU", TODAY + dt.timedelta(days=137), None),
        (PROJECT, "GPU", TODAY + dt.timedelta(days=137), None),
        (OTHER_PROJECT, "GPU", TODAY - dt.timedelta(days=3), "no compute days left"),
        (OTHER_PROJECT, "Storage", TODAY - dt.timedelta(days=3), "storage billing; no compute days left"),
    ]


def test_collect_project_absent_from_projectinfo_has_no_end():
    projectinfo = "Information for project_465000123:\n    137 day(s) of compute left\n"
    cluster = make_lumi(fixture_results(allocations=ALLOCATIONS_TWO_PROJECTS, projectinfo=projectinfo))

    report = cluster.collect(sections=("compute",), today=TODAY, now=NOW)

    assert [(b.project, b.end) for b in report.budgets] == [
        (PROJECT, TODAY + dt.timedelta(days=137)),
        (PROJECT, TODAY + dt.timedelta(days=137)),
        (OTHER_PROJECT, None),
        (OTHER_PROJECT, None),
    ]


def test_collect_failing_allocations_is_a_compute_error_only():
    allocations = failed(1, stderr="lumi-allocations: database unavailable")
    cluster = make_lumi(fixture_results(allocations=allocations))

    report = cluster.collect(today=TODAY, now=NOW)

    assert report.errors == [
        "compute: `lumi-allocations` failed (exit status 1): lumi-allocations: database unavailable"
    ]
    assert report.failed_sections == ["compute"]
    assert report.warnings == []
    assert report.budgets == []
    assert report.storage == fixture_quotas()
    assert len(report.jobs) == 5


@pytest.mark.parametrize(
    "allocations, error",
    [
        ("Service temporarily unavailable\n", "compute: unexpected output of lumi-allocations (no table header)"),
        (one_cell_table(" maintenance "), "compute: unexpected output of lumi-allocations (no parsable allocation)"),
    ],
)
def test_collect_unexpected_allocations_output_is_a_compute_error(allocations, error):
    cluster = make_lumi(fixture_results(allocations=allocations))

    report = cluster.collect(sections=("compute",), today=TODAY, now=NOW)

    assert report.errors == [error]
    assert report.failed_sections == ["compute"]
    assert report.budgets == []


def test_collect_reports_unparsable_cells_and_quotas_as_warnings():
    allocations = read_fixture("lumi/allocations.txt").replace("0/12000   (0.0%) gpu/hours", "maintenance")
    workspaces = read_fixture("lumi/workspaces.txt").replace("    1    of     3M  /", "  1x    of     3M  /")
    cluster = make_lumi(fixture_results(allocations=allocations, workspaces=workspaces))

    report = cluster.collect(sections=("compute", "storage"), today=TODAY, now=NOW)

    assert report.errors == []
    assert report.warnings == [
        "cannot parse the GPU allocation of project_465000123: 'maintenance'",
        "cannot parse the file quota of /flash/project_465000123: invalid count unit in '1x'",
    ]
    assert [(b.resource, b.end) for b in report.budgets] == [("CPU", END), ("Storage", END)]
    flash = report.storage[3]
    assert (flash.name, flash.used_bytes, flash.files_used, flash.files_limit) == (
        "/flash/project_465000123",
        4 * KiB,
        None,
        None,
    )


@pytest.mark.parametrize(
    "workspaces, error",
    [
        (failed(None), "storage: `lumi-workspaces` did not complete: no exit status"),
        (failed(124), "storage: `lumi-workspaces` timed out (time limit of 180 s reached)"),
    ],
)
def test_collect_failing_workspaces_is_a_storage_error_only(workspaces, error):
    cluster = make_lumi(fixture_results(workspaces=workspaces))

    report = cluster.collect(sections=("compute", "storage"), today=TODAY, now=NOW)

    assert report.errors == [error]
    assert report.failed_sections == ["storage"]
    assert report.storage == []
    assert report.budgets == fixture_budgets(end=END, reported=True)


def test_collect_failing_home_command_names_home_quota_home():
    cluster = make_lumi(fixture_results(home=failed(1)))

    report = cluster.collect(sections=("storage",), today=TODAY, now=NOW)

    assert report.errors == []
    assert report.warnings == []
    assert report.storage == fixture_quotas(home_name="home")


def test_collect_filters_projects_case_insensitively_and_keeps_personal_quotas():
    cluster = make_lumi(
        fixture_results(allocations=ALLOCATIONS_TWO_PROJECTS, workspaces=WORKSPACES_PROJECT_THEN_USER),
        projects=["PROJECT_465000456"],
    )

    report = cluster.collect(today=TODAY, now=NOW)

    assert report.errors == []
    assert [(b.project, b.resource) for b in report.budgets] == [(OTHER_PROJECT, "GPU"), (OTHER_PROJECT, "Storage")]
    assert [(q.name, q.project) for q in report.storage] == [
        ("/project/project_465000456", OTHER_PROJECT),
        ("/scratch/project_465000456", OTHER_PROJECT),
        ("/users/jdoe1234", None),
        (HOME, None),
    ]
    assert report.jobs == []
    assert report.warnings == ["5 job(s) of other accounts hidden by the 'projects' setting"]


def test_collect_warns_about_configured_projects_unknown_to_lumi():
    cluster = make_lumi(fixture_results(), projects=["project_465000123", "project_999"])

    report = cluster.collect(sections=("compute", "storage"), today=TODAY, now=NOW)

    assert report.errors == []
    assert report.warnings == ["project(s) not found on this cluster: project_999"]
    assert report.budgets == fixture_budgets(end=END, reported=True)
