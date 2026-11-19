"""JUPITER (Jülich Supercomputing Centre, Germany).

All the information comes from JSC's ``jutil`` utility (JSON output):

* Compute budgets: ``jutil user cpuquota --currsys`` (core-hours of every
  budget account on the local system, with its accounting period).
* Storage quotas: ``jutil user dataquota`` for the personal home directory and
  ``jutil project dataquota -p <project>`` for every project of the user.

Budgets in the ``fixed`` accounting mode are not split per month.  For other
modes the monthly figures are shown when ``jutil`` reports them.

GPUs: a JUPITER Booster node has 4 NVIDIA GH200 superchips (H100-class GPU
with 96 GB of HBM3, and a 72-core Grace CPU).  Nodes are allocated whole, and
JSC charges all their 288 cores whatever the number of GPUs used, so a
GPU-hour costs 72 core-hours of the ``jupiter_booster`` budget.  See
https://apps.fz-juelich.de/jsc/hps/jupiter/batchsystem.html and
https://apps.fz-juelich.de/jsc/hps/jupiter/accounting.html
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Dict, List, Optional

from hpcmon.clusters.base import HOME_COMMAND, CollectionError, Results
from hpcmon.clusters.slurm import SlurmCluster
from hpcmon.models import ComputeBudget, StorageQuota
from hpcmon.units import parse_count, parse_date, parse_number, parse_size

#: Shell script printing the data quotas of all the projects of the user.  It
#: fails when the projects cannot be listed, and keeps going (but reports the
#: failure) when the quotas of a single project cannot be read.
PROJECT_DATAQUOTA_COMMAND = (
    "projects=$(jutil user projects --noheader --format parsable) || exit $?; "
    "rc=0; "
    "for p in $(printf '%s\\n' \"$projects\" | grep '|' | cut -d'|' -f1 | sort -u); do "
    'jutil project dataquota --project "$p" --format json || rc=$?; '
    "done; exit $rc"
)

#: JSC accounting mode of budgets that are not split per month.
FIXED_MODE = "fixed"


class JupiterCluster(SlurmCluster):
    """JUPITER, the European exascale system hosted by JSC."""

    type_name = "jupiter"
    display_name = "JUPITER"
    gpus = "4× NVIDIA GH200 96 GB"
    gpu_rates = {("jupiter_booster", "core-h"): 1 / 72}

    def compute_commands(self) -> Dict[str, str]:
        return {"cpuquota": "jutil user cpuquota --currsys --format json"}

    def storage_commands(self) -> Dict[str, str]:
        return {
            "userdata": "jutil user dataquota --format json",
            "projectdata": PROJECT_DATAQUOTA_COMMAND,
            "home": HOME_COMMAND,
        }

    def parse_compute(self, results: Results, today: dt.date) -> List[ComputeBudget]:
        return [_budget(entry) for entry in parse_json_documents(self.output(results, "cpuquota"))]

    def parse_storage(self, results: Results, today: dt.date) -> List[StorageQuota]:
        try:
            home: Optional[str] = self.output(results, "home").strip() or None
        except CollectionError:
            home = None
        quotas = []
        usage_entries = []
        for entry in parse_json_documents(self.output(results, "userdata")):
            if "data-usage" not in entry or not entry.get("group"):
                continue
            # Entries of project groups only hold the user's own usage there;
            # the project totals and limits come from the project data quotas.
            usage_entries.append(entry)
            if _same(entry.get("group"), entry.get("username")):
                filesystem = str(entry.get("filesystem", "home"))
                name = home if home and "home" in filesystem else filesystem
                quotas.append(_quota(entry, name=name, project=None))

        try:
            projectdata = self.output(results, "projectdata", check=False)
        except CollectionError as exc:
            self.warn(f"project data quotas unavailable: {exc}")
            projectdata = ""
        else:
            if results["projectdata"].returncode != 0:
                self.warn(f"project data quotas may be incomplete: {self.output_detail(results, 'projectdata')}")
        covered = set()
        for entry in parse_json_documents(projectdata):
            if "data-usage" not in entry:
                continue
            name = entry.get("project-dir") or entry.get("filesystem") or "?"
            project = _text(entry, "project")
            covered.add((str(project).lower(), str(entry.get("filesystem", "")).lower()))
            quotas.append(_quota(entry, name=str(name), project=project))

        # When the quotas of a project could not be read (e.g. for members who
        # are not allowed to), fall back to the user's own usage there.
        for entry in usage_entries:
            group = str(entry.get("group", ""))
            filesystem = str(entry.get("filesystem", ""))
            if _same(group, entry.get("username")) or (group.lower(), filesystem.lower()) in covered:
                continue
            quota = _quota(entry, name=f"{filesystem} of {group} (your usage)", project=group)
            quota.limit_bytes = quota.files_limit = None
            quotas.append(quota)
        return quotas

    @staticmethod
    def output_detail(results: Results, key: str) -> str:
        """The last line of error output of the command ``key``."""
        result = results[key]
        lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        return lines[-1] if lines else f"exit status {result.returncode}"


def parse_json_documents(text: str) -> List[Dict[str, Any]]:
    """Parse one or more concatenated JSON arrays of objects.

    Empty output means no entries.

    :raises CollectionError: if the output is not valid JSON.
    """
    decoder = json.JSONDecoder()
    entries: List[Dict[str, Any]] = []
    position = 0
    text = text.strip()
    while position < len(text):
        try:
            document, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise CollectionError(f"invalid JSON output from jutil: {exc}") from exc
        items = document if isinstance(document, list) else [document]
        entries.extend(item for item in items if isinstance(item, dict))
        while position < len(text) and text[position].isspace():
            position += 1
    return entries


def _number(entry: Dict[str, Any], key: str) -> Optional[float]:
    return parse_number(_text(entry, key))


def _same(first: Any, second: Any) -> bool:
    """Whether two JSON values denote the same name (numbers or strings)."""
    return first is not None and second is not None and str(first) == str(second)


def _add_note(budget: ComputeBudget, note: str) -> None:
    budget.note = note if budget.note is None else f"{budget.note}; {note}"


def _budget(entry: Dict[str, Any]) -> ComputeBudget:
    mode = (_text(entry, "acct-mode") or "").lower()
    account = _text(entry, "budget-account")
    project = (_text(entry, "project") or account or "?").lower()
    total = _number(entry, "coreh-quota-total")
    used = _number(entry, "coreh-consumed-total")
    available = _number(entry, "coreh-avail")
    budget = ComputeBudget(
        project=project,
        account=account if account and account.lower() != project else None,
        resource=str(entry.get("cont-partition") or entry.get("system") or "compute"),
        unit="core-h",
        total=total,
        used=used,
        # What is left until the end of the period is the quota minus the
        # consumption; "coreh-avail" is what can be used right now, which
        # differs from it in accounting modes other than "fixed".
        available=None if total is not None and used is not None else available,
        start=parse_date(_text(entry, "acct-period-start")),
        end=parse_date(_text(entry, "acct-period-end")),
    )
    if mode and mode != FIXED_MODE:
        budget.month_used = _number(entry, "coreh-consumed-curr-month")
        month_quota = _number(entry, "coreh-quota-curr-month")
        month_avail = _number(entry, "coreh-avail-curr-month")
        if month_quota is not None:
            budget.month_total = month_quota
        elif month_avail is not None and budget.month_used is not None:
            budget.month_total = month_avail + budget.month_used
        _add_note(budget, f"{mode} accounting")
        remaining = budget.remaining
        if available is not None and (remaining is None or abs(available - remaining) >= 0.5):
            _add_note(budget, f"available now: {available:,.0f}")
    state = (_text(entry, "cont-state") or "normal").lower()
    if state not in ("normal", ""):
        _add_note(budget, f"state: {state}")
    project_state = _text(entry, "project-state") or "A"
    if project_state.upper() not in ("A", ""):
        _add_note(budget, f"project state: {project_state}")
    return budget


def _text(entry: Dict[str, Any], key: str) -> Optional[str]:
    value = entry.get(key)
    return None if value is None else str(value)


def _quota(entry: Dict[str, Any], name: str, project: Optional[str]) -> StorageQuota:
    return StorageQuota(
        name=name,
        project=project,
        used_bytes=parse_size(_text(entry, "data-usage"), default_unit="KB"),
        limit_bytes=parse_size(_text(entry, "data-soft-limit"), default_unit="KB"),
        files_used=parse_count(_text(entry, "inode-usage")),
        files_limit=parse_count(_text(entry, "inode-soft-limit")),
    )
