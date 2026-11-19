"""Leonardo (CINECA, Italy).

* Compute budgets: ``saldo -b`` (Booster partition) and ``saldo -b --dcgp``
  (Data Centric General Purpose partition).  CINECA splits the budget of a
  project per month, so both the total and the monthly figures are shown.
  Amounts are "local" core-hours.
* Storage quotas: ``cindata`` (home, public, work and scratch areas).  Its
  figures are refreshed periodically, so they can be a few hours old.

GPUs: a Booster node has 4 NVIDIA A100 of 64 GB and 32 cores.  Nodes are
shared, and a job is charged for the largest share of the node it takes in
cores, GPUs or memory: 8 core-hours per GPU-hour for jobs that take no more
than 8 cores and a quarter of the memory per GPU.  The DCGP nodes have no GPU.
See https://docs.hpc.cineca.it/hpc/hpc_intro.html
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Callable, Dict, List, Optional

from hpcmon.clusters.base import CollectionError, Results
from hpcmon.clusters.slurm import SlurmCluster
from hpcmon.models import ComputeBudget, StorageQuota
from hpcmon.units import parse_date, parse_number, parse_size

_DATE_RE = re.compile(r"^\d{8}$")
_PERSONAL_RE = re.compile(r"/user(?:external|internal)/")
_FRESH_RE = re.compile(r"^-?(?P<amount>\d+)(?P<unit>sec|min|hou|day|wee|mon|yea)", re.IGNORECASE)
_FRESH_UNITS = {
    "sec": "second", "min": "minute", "hou": "hour", "day": "day", "wee": "week", "mon": "month", "yea": "year",
}

#: Receives the description of a non-fatal parsing problem.
Warn = Callable[[str], None]


def _ignore(message: str) -> None:
    """Default ``warn`` callback: drop the message."""
_NO_BUDGET_MARKERS = ("username not existing", "no account", "not found")

#: Columns of ``cindata`` used to build the quotas.
_CINDATA_COLUMNS = ("USER", "AREADESCR", "USED", "QTA", "aUSED", "aQTA")


class LeonardoCluster(SlurmCluster):
    """Leonardo, the EuroHPC pre-exascale system hosted by CINECA."""

    type_name = "leonardo"
    display_name = "Leonardo"
    gpus = "4× NVIDIA A100 64 GB"
    gpu_rates = {("Booster", "core-h"): 1 / 8}

    def compute_commands(self) -> Dict[str, str]:
        return {"saldo": "saldo -b", "saldo_dcgp": "saldo -b --dcgp"}

    def storage_commands(self) -> Dict[str, str]:
        return {"cindata": "cindata"}

    def parse_compute(self, results: Results, today: dt.date) -> List[ComputeBudget]:
        # The partitions are independent: a problem with one of them must not
        # hide the budgets of the other.
        budgets: List[ComputeBudget] = []
        failures = []
        for key, resource in (("saldo", "Booster"), ("saldo_dcgp", "DCGP")):
            try:
                budgets.extend(parse_saldo(self.output(results, key), resource=resource))
            except (CollectionError, ValueError) as exc:
                failures.append((resource, exc))
        if len(failures) == 2:
            raise CollectionError("; ".join(f"{resource}: {exc}" for resource, exc in failures))
        for resource, exc in failures:
            self.warn(f"{resource} budgets unavailable: {exc}")
        return budgets

    def parse_storage(self, results: Results, today: dt.date) -> List[StorageQuota]:
        return parse_cindata(self.output(results, "cindata"), warn=self.warn)


def parse_saldo(text: str, resource: str) -> List[ComputeBudget]:
    """Parse the budget table printed by ``saldo -b``.

    The data lines have the columns: account, start, end, total,
    localCluster consumed, total consumed, consumed %, month total and month
    consumed.  Output without any data line (e.g. "username not existing")
    means that the user has no budget.

    :raises CollectionError: for output that is neither a table nor a known
        "no budget" message.
    """
    budgets = []
    seen_table = False
    for line in text.splitlines():
        if line.lstrip().lower().startswith("account"):
            seen_table = True
        fields = line.split()
        if len(fields) < 9 or not (_DATE_RE.match(fields[1]) and _DATE_RE.match(fields[2])):
            continue
        account, start, end, total, _local, consumed, _percent, month_total, month_used = fields[:9]
        budgets.append(
            ComputeBudget(
                project=account,
                resource=resource,
                unit="core-h",
                total=parse_number(total),
                used=parse_number(consumed),
                month_total=parse_number(month_total),
                month_used=parse_number(month_used),
                start=parse_date(start),
                end=parse_date(end),
            )
        )
    if not budgets and not seen_table and text.strip():
        if not any(marker in text.lower() for marker in _NO_BUDGET_MARKERS):
            raise CollectionError(f"unexpected output of saldo: {text.strip().splitlines()[0]!r}")
    return budgets


def describe_freshness(fresh: str) -> Optional[str]:
    """Turn a ``cindata`` FRESH value such as ``"3hou"`` into a note."""
    match = _FRESH_RE.match(fresh.strip())
    if not match:
        return None
    amount = int(match.group("amount"))
    unit = _FRESH_UNITS[match.group("unit").lower()]
    return f"as of {amount} {unit}{'s' if amount != 1 else ''} ago"


def parse_cindata(text: str, warn: Warn = _ignore) -> List[StorageQuota]:
    """Parse the default ("standard user") output of ``cindata``.

    Personal areas (paths under ``userexternal``/``userinternal`` or ending
    with the user name) show the user's own occupancy and quota.  Other areas
    are project areas (work, fast scratch), named after their last path
    component, which show the occupancy and size of the whole area.  Sizes use
    1024 multipliers for upper case prefixes and 1000 for lower case ones.

    :param warn: Receives a message for every row that cannot be parsed.
    :raises CollectionError: if the header is missing.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    header_index = next((i for i, line in enumerate(lines) if line.split()[:1] == ["USER"]), None)
    if header_index is None:
        if not lines:
            return []
        raise CollectionError("unexpected output of cindata (no header)")
    header = lines[header_index].split()
    missing = [column for column in _CINDATA_COLUMNS if column not in header]
    if missing:
        raise CollectionError(f"unexpected output of cindata (missing columns: {', '.join(missing)})")
    index = {column: header.index(column) for column in _CINDATA_COLUMNS}
    fresh_index = header.index("FRESH") if "FRESH" in header else None

    quotas = []
    for line in lines[header_index + 1 :]:
        fields = line.split()
        if len(fields) != len(header):
            warn(f"cannot parse cindata line: {line.strip()!r}")
            continue
        path = fields[index["AREADESCR"]]
        user = fields[index["USER"]]
        last = path.rstrip("/").rsplit("/", 1)[-1]
        personal = bool(_PERSONAL_RE.search(path)) or last == user
        try:
            area_used = parse_size(fields[index["aUSED"]])
            if personal or area_used is None:
                used = parse_size(fields[index["USED"]])
                limit = parse_size(fields[index["QTA"]])
            else:
                used = area_used
                limit = parse_size(fields[index["aQTA"]])
        except ValueError as exc:
            warn(f"cannot parse the quota of {path}: {exc}")
            continue
        note = describe_freshness(fields[fresh_index]) if fresh_index is not None else None
        quotas.append(
            StorageQuota(
                name=path,
                project=None if personal else (last or None),
                used_bytes=used,
                limit_bytes=limit,
                note=note,
            )
        )
    return quotas
