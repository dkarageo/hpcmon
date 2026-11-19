"""LUMI (CSC, Finland).

* Compute budgets: ``lumi-allocations`` (CPU core-hours, GPU-hours and storage
  TB-hours of every project, for the whole allocation period).
* End of the projects: ``lumi-ldap-projectinfo`` ("N day(s) of compute left").
* Storage quotas: ``lumi-workspaces`` (live ``lfs quota`` of the home, project,
  scratch and flash directories).

LUMI does not split the allocations per month.

GPUs: a LUMI-G node has 4 AMD MI250X of 128 GB, each made of 2 GCDs that Slurm
counts as separate GPUs.  LUMI's GPU-hour is one MI250X (2 GCDs) for an hour:
a whole node is billed 4 GPU-hours per hour, and in the shared partitions a
GCD is billed at least half a GPU-hour per hour (more when a job asks for more
than 8 cores or 64 GB per GCD).  CPU core-hours and storage TB-hours cannot be
spent on the GPU nodes.  See
https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Callable, Dict, List, Optional

from hpcmon.clusters.base import HOME_COMMAND, CollectionError, Results
from hpcmon.clusters.slurm import SlurmCluster
from hpcmon.models import ComputeBudget, StorageQuota
from hpcmon.units import parse_count, parse_number, parse_size

#: Receives the description of a non-fatal parsing problem.
Warn = Callable[[str], None]

_ALLOCATION_RE = re.compile(
    r"^\s*(?P<used>[\d.,]+)\s*/\s*(?P<total>[\d.,]+)\s*(?:\([^)]*\))?\s*(?P<unit>[A-Za-z]+/[A-Za-z]+)?\s*$"
)
_UNITS = {"core/hours": "core-h", "gpu/hours": "GPU-h", "tb/hours": "TB-h"}

#: Note attached to the storage billing budget (it is not compute time).
STORAGE_BILLING_NOTE = "storage billing"
#: Note attached to budgets of projects without a full day of compute left.
PERIOD_OVER_NOTE = "no compute days left"

_INFO_HEADER_RE = re.compile(r"^Information for (?:project\s+)?(?!userid\b)(?P<project>[A-Za-z0-9_]+):?(?:\s|$)")
_DAYS_LEFT_RE = re.compile(r"(?P<days>-?\d+)\s+day\(s\)\s+of\s+compute\s+left", re.IGNORECASE)
_REFRESHED_RE = re.compile(r"Data was last refreshed at\s+(?P<when>.+?)\s*$")

_QUOTA = (
    r"quota:\s*[\d.]+%\s*used\s*\(\s*(?P<used>[^\s/]+)\s+of\s+(?P<soft>[^\s/]+)\s*/\s*(?P<hard>[^\s/)]+)\s+soft/hard\)"
)
_BLOCK_RE = re.compile(r"^\s*-\s*(?:(?P<path>/\S+?):\s*)?block " + _QUOTA, re.IGNORECASE)
_FILE_RE = re.compile(r"^\s*(?:-\s*)?file " + _QUOTA, re.IGNORECASE)
_USER_SECTION_RE = re.compile(r"^Information for userid\s+(?P<user>\S+?):?\s*$")
_PROJECT_SECTION_RE = re.compile(r"^Information for project\s+(?P<project>\S+)")

#: ``lfs quota`` marks values that exceed the quota with an asterisk.
OVER_QUOTA_NOTE = "over quota"


def _ignore(message: str) -> None:
    """Default ``warn`` callback: drop the message."""


class LumiCluster(SlurmCluster):
    """LUMI, hosted by CSC in Kajaani, Finland."""

    type_name = "lumi"
    display_name = "LUMI"
    gpus = "4× AMD MI250X 128 GB (2 GCDs each)"
    gpu_rates = {("GPU", "GPU-h"): 1.0}

    def compute_commands(self) -> Dict[str, str]:
        return {"allocations": "lumi-allocations", "projectinfo": "lumi-ldap-projectinfo"}

    def storage_commands(self) -> Dict[str, str]:
        return {"workspaces": "lumi-workspaces", "home": HOME_COMMAND}

    # -- compute ----------------------------------------------------------

    def parse_compute(self, results: Results, today: dt.date) -> List[ComputeBudget]:
        budgets = parse_allocations(self.output(results, "allocations"), warn=self.warn)
        try:
            info = self.output(results, "projectinfo")
        except CollectionError as exc:
            self.warn(f"end of projects unknown: {exc}")
            return budgets
        days_left = parse_days_left(info)
        # The days are counted from LUMI's last data refresh, in its time zone.
        base = parse_refresh_date(info) or today
        for budget in budgets:
            days = days_left.get(budget.project)
            if days is None:
                continue
            budget.end = base + dt.timedelta(days=days)
            if days <= 0:
                budget.note = _join(budget.note, PERIOD_OVER_NOTE)
        return budgets

    # -- storage ----------------------------------------------------------

    def parse_storage(self, results: Results, today: dt.date) -> List[StorageQuota]:
        try:
            home: Optional[str] = self.output(results, "home").strip() or None
        except CollectionError:
            home = None
        return parse_workspaces(self.output(results, "workspaces"), home=home, warn=self.warn)


def _join(note: Optional[str], extra: str) -> str:
    return extra if not note else f"{note}; {extra}"


def parse_allocations(text: str, warn: Warn = _ignore) -> List[ComputeBudget]:
    """Parse the table printed by ``lumi-allocations``.

    :param warn: Receives a message for every cell that cannot be parsed.
    :raises CollectionError: if the table header is missing, or if the table
        has rows but none of them could be parsed.
    """
    lines = text.splitlines()
    header_index = next(
        (i for i, line in enumerate(lines) if "|" in line and line.split("|")[0].strip().lower() == "project"),
        None,
    )
    if header_index is None:
        if not text.strip() or "no project" in text.lower():
            return []
        raise CollectionError("unexpected output of lumi-allocations (no table header)")
    resources = [cell.split("(")[0].strip() for cell in lines[header_index].split("|")[1:]]

    budgets = []
    rows = parsed = 0
    for line in lines[header_index + 1 :]:
        cells = line.split("|")
        project = cells[0].strip()
        if len(cells) < 2 or not project or set(project) <= {"-", "="}:
            continue
        rows += 1
        for resource, cell in zip(resources, cells[1:]):
            match = _ALLOCATION_RE.match(cell)
            if not match:
                _report_cell(warn, resource, project, cell)
                continue
            try:
                total = parse_number(match.group("total"))
                used = parse_number(match.group("used"))
            except ValueError:
                warn(f"cannot parse the {resource} allocation of {project}: {cell.strip()!r}")
                continue
            parsed += 1
            if not total and not used:
                continue  # no allocation of this kind
            raw_unit = (match.group("unit") or "").lower()
            unit = _UNITS.get(raw_unit, raw_unit.replace("/hours", "-h") or "h")
            budget = ComputeBudget(project=project, resource=resource, unit=unit, total=total, used=used)
            if raw_unit == "tb/hours":
                budget.note = STORAGE_BILLING_NOTE
            budgets.append(budget)
    if rows and not parsed:
        raise CollectionError("unexpected output of lumi-allocations (no parsable allocation)")
    return budgets


def _report_cell(warn: Warn, resource: str, project: str, cell: str) -> None:
    """Warn about an allocation cell that cannot be parsed (empty cells are fine)."""
    if cell.strip() not in ("", "-", "--"):
        warn(f"cannot parse the {resource} allocation of {project}: {cell.strip()!r}")


def parse_days_left(text: str) -> Dict[str, int]:
    """Parse ``lumi-ldap-projectinfo``: days of compute left, by project."""
    days: Dict[str, int] = {}
    project: Optional[str] = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Information for"):
            header = _INFO_HEADER_RE.match(stripped)
            # Blocks with an unrecognised header are ignored rather than
            # attributed to the previous project.
            project = header.group("project") if header else None
            continue
        match = _DAYS_LEFT_RE.search(line)
        if match and project is not None:
            days[project] = int(match.group("days"))
    return days


def parse_refresh_date(text: str) -> Optional[dt.date]:
    """The date of "Data was last refreshed at <ctime>" in LUMI's tools."""
    for line in text.splitlines():
        match = _REFRESHED_RE.search(line)
        if match:
            try:
                return dt.datetime.strptime(match.group("when"), "%a %b %d %H:%M:%S %Y").date()
            except ValueError:
                return None
    return None


def _size(text: str) -> "tuple[Optional[int], bool]":
    """Parse a size of ``lfs quota``, which may end with ``*`` when over quota."""
    over = text.endswith("*")
    return parse_size(text.rstrip("*")), over


def parse_workspaces(text: str, home: Optional[str] = None, warn: Warn = _ignore) -> List[StorageQuota]:
    """Parse the quotas printed by ``lumi-workspaces``.

    :param home: Path of the home directory, used to name the home quota.
    :param warn: Receives a message for every quota line that cannot be parsed.
    """
    quotas: List[StorageQuota] = []
    project: Optional[str] = None
    last: Optional[StorageQuota] = None
    for line in text.splitlines():
        if _USER_SECTION_RE.match(line.strip()):
            project, last = None, None
            continue
        section = _PROJECT_SECTION_RE.match(line.strip())
        if section:
            project, last = section.group("project"), None
            continue
        block = _BLOCK_RE.match(line)
        if block:
            name = block.group("path") or home or "home"
            try:
                used, over_used = _size(block.group("used"))
                limit, _ = _size(block.group("soft"))
            except ValueError as exc:
                warn(f"cannot parse the quota of {name}: {exc}")
                last = None
                continue
            last = StorageQuota(name=name, project=project, used_bytes=used, limit_bytes=limit)
            if over_used:
                last.note = OVER_QUOTA_NOTE
            quotas.append(last)
            continue
        files = _FILE_RE.match(line)
        if files and last is not None:
            try:
                files_used = parse_count(files.group("used").rstrip("*"))
                files_limit = parse_count(files.group("soft").rstrip("*"))
            except ValueError as exc:
                warn(f"cannot parse the file quota of {last.name}: {exc}")
                continue
            last.files_used, last.files_limit = files_used, files_limit
            if files.group("used").endswith("*") and last.note is None:
                last.note = OVER_QUOTA_NOTE
    return quotas
