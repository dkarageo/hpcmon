"""MeluXina (LuxProvide, Luxembourg).

Everything comes from LuxProvide's ``myquota`` tool, which is run only once
per query because it refuses to run again within 10 seconds:

* Compute budgets: node-hours of the *current month* for the CPU, GPU, FPGA
  and large-memory partitions.  ``myquota`` reports neither the total budget
  nor the end of the projects (the end date can be set in the configuration).
* Storage quotas: home and project directories (GiB and number of files).

GPUs: a GPU node has 4 NVIDIA A100 of 40 GB.  Nodes are allocated whole, so a
GPU node-hour is 4 GPU-hours.  The CPU, FPGA and large-memory nodes have no
GPU.  See https://docs.lxp.lu/system/overview/
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Callable, Dict, List, Optional, Tuple

from hpcmon.clusters.base import CollectionError, Results
from hpcmon.clusters.slurm import SlurmCluster
from hpcmon.models import ComputeBudget, StorageQuota
from hpcmon.units import parse_count, parse_number, parse_size

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

_RATE_LIMIT_MARKER = "wait at least"
_DATA_HEADER_RE = re.compile(r"DATA ALLOCATIONS|\bDatapath\b")

#: Receives the description of a non-fatal parsing problem.
Warn = Callable[[str], None]


def _ignore(message: str) -> None:
    """Default ``warn`` callback: drop the message."""
_GROUP_RE = re.compile(r"(?P<name>\S+)\s+node-hours", re.IGNORECASE)
_USED_RE = re.compile(r"\bUsed\b")
_PERCENT_RE = re.compile(r"%used\b", re.IGNORECASE)
_DATA_UNIT_RE = re.compile(r"\bData\s+(?P<unit>[KMGTP]i?B)\b")
_DATA_ROW_RE = re.compile(
    r"^\s*(?P<path>/\S+)\s+(?P<used>[\d.,]+|-+)\s*\|\s*(?P<max>[\d.,]+|-+)?\s*\|\s*\S*\s+"
    r"(?P<files>[\d.,]+|-+)\s*\|\s*(?P<files_max>[\d.,]+|-+)?\s*\|"
)

#: Note attached to the MeluXina budgets.
MONTHLY_NOTE = "current month only"


def strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences (colours) from ``text``."""
    return ANSI_RE.sub("", text)


class MeluxinaCluster(SlurmCluster):
    """MeluXina, the EuroHPC petascale system hosted by LuxProvide."""

    type_name = "meluxina"
    display_name = "MeluXina"
    gpus = "4× NVIDIA A100 40 GB"
    gpu_rates = {("GPU", "node-h"): 4.0}

    def compute_commands(self) -> Dict[str, str]:
        return {"myquota": "myquota"}

    def storage_commands(self) -> Dict[str, str]:
        return {"myquota": "myquota"}

    def _myquota(self, results: Results) -> str:
        text = strip_ansi(self.output(results, "myquota", check=False))
        result = results["myquota"]
        if _RATE_LIMIT_MARKER in (text + strip_ansi(result.stderr)).lower():
            raise CollectionError("myquota refused to run (it can only run once every 10 seconds); retry later")
        if not text.strip():
            raise CollectionError(self._failure(result.returncode, result.stderr))
        return text

    def _failure(self, returncode: Optional[int], stderr: str) -> str:
        lines = [line.strip() for line in strip_ansi(stderr).splitlines() if line.strip()]
        detail = f": {lines[-1]}" if lines else ""
        if returncode == 0:
            return f"`myquota` printed nothing{detail}"
        return f"`myquota` failed (exit status {returncode}){detail}"

    def parse_compute(self, results: Results, today: dt.date) -> List[ComputeBudget]:
        return parse_compute_table(self._myquota(results))

    def parse_storage(self, results: Results, today: dt.date) -> List[StorageQuota]:
        return parse_data_table(self._myquota(results), warn=self.warn)


def _unexpected(text: str, what: str) -> CollectionError:
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return CollectionError(f"unexpected output of myquota (no {what} table): {first!r}")


def _group_spans(header: str, subheader: str) -> List[Tuple[str, int, int]]:
    """Return ``(resource, start, end)`` column spans of every partition."""
    names = [match.group("name") for match in _GROUP_RE.finditer(header)]
    used = [match.start() for match in _USED_RE.finditer(subheader)]
    percent = [match.end() for match in _PERCENT_RE.finditer(subheader)]
    if not names or len(names) != len(used):
        raise CollectionError("unexpected output of myquota (compute table header)")
    spans = []
    for i, name in enumerate(names):
        start = 0 if i == 0 else (_at(percent, i - 1, used[i]) + used[i]) // 2
        end = (_at(percent, i, used[i + 1]) + used[i + 1]) // 2 if i + 1 < len(names) else 10**6
        spans.append((name, start, end))
    return spans


def _at(values: List[int], index: int, default: int) -> int:
    return values[index] if index < len(values) else default


def parse_compute_table(text: str) -> List[ComputeBudget]:
    """Parse the "compute allocations" table of ``myquota`` (ANSI-free).

    The table has fixed-width columns: cells of partitions without an
    allocation are blank, so the columns are located from the header.

    Every row is checked against the "%used" column printed by ``myquota``,
    so that a value overflowing into the next column is detected rather than
    misread.

    :raises CollectionError: if the output has neither a compute nor a data
        table, if the header is malformed, or if a row is inconsistent.
    """
    lines = text.splitlines()
    header_index = next((i for i, line in enumerate(lines) if _GROUP_RE.search(line)), None)
    if header_index is None:
        if _DATA_HEADER_RE.search(text):
            return []  # only a data table: no compute allocation
        raise _unexpected(text, "compute")
    if header_index + 1 >= len(lines):
        return []
    spans = _group_spans(lines[header_index], lines[header_index + 1])

    budgets = []
    for line in lines[header_index + 2 :]:
        if not line.strip():
            continue
        if _DATA_HEADER_RE.search(line):
            break
        project = line.split()[0]
        for i, (resource, start, end) in enumerate(spans):
            cells = line[start:end].split()
            if i == 0:
                cells = cells[1:]  # drop the project name
            percents = [cell for cell in cells if cell.endswith("%")]
            cells = [cell for cell in cells if not cell.endswith("%")]
            used = parse_number(cells[0]) if cells else None
            limit = parse_number(cells[1]) if len(cells) > 1 else None
            if percents:
                _check_percent(project, resource, used, limit, parse_number(percents[0]), line)
            if limit is None and not used:
                continue  # no allocation and no usage on this partition
            budgets.append(
                ComputeBudget(
                    project=project,
                    resource=resource,
                    unit="node-h",
                    month_total=limit,
                    month_used=used,
                    note=MONTHLY_NOTE,
                )
            )
    return budgets


def _check_percent(
    project: str,
    resource: str,
    used: Optional[float],
    limit: Optional[float],
    percent: Optional[float],
    line: str,
) -> None:
    """Make sure that the cells of a row agree with its "%used" cell."""
    if percent is None or (not limit and not used):
        return  # nothing to check: no allocation and no usage
    if used is None or not limit:
        raise CollectionError(f"cannot parse the {resource} allocation of {project}: {line.strip()!r}")
    # Used and Max are rounded to whole hours, so allow for that.
    tolerance = 0.5 + 100.0 / limit
    if abs(used / limit * 100.0 - percent) > tolerance:
        raise CollectionError(f"cannot parse the {resource} allocation of {project}: {line.strip()!r}")


def parse_data_table(text: str, warn: Warn = _ignore) -> List[StorageQuota]:
    """Parse the "data allocations" table of ``myquota`` (ANSI-free).

    :param warn: Receives a message for every row that cannot be parsed.
    :raises CollectionError: if there is no data table.
    """
    if not _DATA_HEADER_RE.search(text):
        if _GROUP_RE.search(text):
            return []  # only a compute table: no data allocation
        raise _unexpected(text, "data")
    unit = "GiB"
    quotas = []
    for line in text.splitlines():
        unit_match = _DATA_UNIT_RE.search(line)
        if unit_match:
            unit = unit_match.group("unit")
            continue
        match = _DATA_ROW_RE.match(line)
        if not match:
            _report_line(warn, line)
            continue
        path = match.group("path")
        project = path.rstrip("/").rsplit("/", 1)[-1] if path.startswith("/project/") else None
        quotas.append(
            StorageQuota(
                name=path,
                project=project,
                used_bytes=_size(match.group("used"), unit),
                limit_bytes=_size(match.group("max"), unit),
                files_used=parse_count(match.group("files")),
                files_limit=parse_count(match.group("files_max")),
            )
        )
    return quotas


def _report_line(warn: Warn, line: str) -> None:
    """Warn about a data line (starting with a path) that cannot be parsed."""
    if line.strip().startswith("/"):
        warn(f"cannot parse myquota data line: {line.strip()!r}")


def _size(value: Optional[str], unit: str) -> Optional[int]:
    number = parse_number(value)
    return None if number is None else parse_size(f"{number}{unit}")
