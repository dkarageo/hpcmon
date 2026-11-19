"""Cluster-independent data model of the information displayed by hpcmon."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Slurm job states that count as "running" (the job holds resources).
RUNNING_STATES = frozenset(
    {"RUNNING", "COMPLETING", "CONFIGURING", "STAGE_OUT", "SIGNALING", "R", "CG", "CF", "SO", "SI"}
)

#: Slurm job states that count as "pending" (waiting in the queue).
PENDING_STATES = frozenset(
    {"PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD", "PD", "RQ", "RH", "RF", "RD"}
)


def _date(value: Optional[dt.date]) -> Optional[str]:
    return value.isoformat() if value is not None else None


@dataclass
class ComputeBudget:
    """A compute allocation (budget) of a project on a cluster.

    Amounts are expressed in the native unit of the cluster (``unit``), e.g.
    core-hours, GPU-hours or node-hours.  Any field may be ``None`` when the
    cluster does not report it.
    """

    project: str
    #: What is being accounted, e.g. ``"GPU"``, ``"CPU"`` or ``"Booster"``.
    resource: str
    #: Unit of all amounts, e.g. ``"GPU-h"``, ``"core-h"``, ``"node-h"``.
    unit: str
    #: Total allocation for the whole allocation period of the project.
    total: Optional[float] = None
    #: Amount consumed so far over the whole allocation period.
    used: Optional[float] = None
    #: Amount still available as reported by the cluster (if it reports it).
    available: Optional[float] = None
    #: Allocation granted for the current month (monthly-split budgets only).
    month_total: Optional[float] = None
    #: Amount consumed during the current month.
    month_used: Optional[float] = None
    #: First day of the allocation period.
    start: Optional[dt.date] = None
    #: Last day of the allocation period (end of the project).
    end: Optional[dt.date] = None
    #: Free-form remark shown next to the budget.
    note: Optional[str] = None
    #: The scheduler account charged for this budget, when it differs from
    #: the project name (e.g. JSC budget accounts).
    account: Optional[str] = None
    #: Estimated GPU-hours that one ``unit`` of this budget buys on the GPU
    #: nodes of the cluster, ``None`` when the budget cannot be spent on GPUs
    #: (CPU partitions, storage billing) or the rate is unknown.
    gpu_hours_per_unit: Optional[float] = None

    @property
    def remaining(self) -> Optional[float]:
        """Amount left until the end of the allocation period."""
        if self.available is not None:
            return self.available
        if self.total is not None and self.used is not None:
            return self.total - self.used
        return None

    @property
    def remaining_fraction(self) -> Optional[float]:
        """Fraction (0-1) of the total allocation that is still available."""
        remaining = self.remaining
        if remaining is None or not self.total:
            return None
        return remaining / self.total

    @property
    def month_remaining(self) -> Optional[float]:
        """Amount left until the end of the current month."""
        if self.month_total is not None and self.month_used is not None:
            return self.month_total - self.month_used
        return None

    @property
    def month_remaining_fraction(self) -> Optional[float]:
        """Fraction (0-1) of the monthly allocation that is still available."""
        remaining = self.month_remaining
        if remaining is None or not self.month_total:
            return None
        return remaining / self.month_total

    def gpu_hours(self, amount: Optional[float]) -> Optional[float]:
        """Convert an ``amount`` of this budget to (estimated) GPU-hours.

        It is ``None`` when unknown, including when it is too large to be
        represented (which would otherwise give an invalid JSON document).
        """
        if amount is None or self.gpu_hours_per_unit is None:
            return None
        value = amount * self.gpu_hours_per_unit
        return value if math.isfinite(value) else None

    @property
    def gpu_hours_total(self) -> Optional[float]:
        """The total allocation, in estimated GPU-hours."""
        return self.gpu_hours(self.total)

    @property
    def gpu_hours_remaining(self) -> Optional[float]:
        """The amount left until the end of the period, in estimated GPU-hours."""
        return self.gpu_hours(self.remaining)

    @property
    def month_gpu_hours_total(self) -> Optional[float]:
        """The allocation of the current month, in estimated GPU-hours."""
        return self.gpu_hours(self.month_total)

    @property
    def month_gpu_hours_remaining(self) -> Optional[float]:
        """The amount left until the end of the month, in estimated GPU-hours."""
        return self.gpu_hours(self.month_remaining)

    def days_left(self, today: dt.date) -> Optional[int]:
        """Days from ``today`` to the last day of the period.

        It is 0 on the last day itself and negative once the period is over.
        """
        if self.end is None:
            return None
        return (self.end - today).days

    def has_ended(self, today: dt.date) -> bool:
        """Whether the allocation period is over."""
        days = self.days_left(today)
        return days is not None and days < 0

    def to_dict(self, today: Optional[dt.date] = None) -> Dict[str, Any]:
        """Return a JSON-serialisable representation, including derived values.

        :param today: The date used to compute ``days_left``.
        """
        return {
            "project": self.project,
            "account": self.account,
            "resource": self.resource,
            "unit": self.unit,
            "total": self.total,
            "used": self.used,
            "remaining": self.remaining,
            "remaining_fraction": self.remaining_fraction,
            "month_total": self.month_total,
            "month_used": self.month_used,
            "month_remaining": self.month_remaining,
            "month_remaining_fraction": self.month_remaining_fraction,
            "start": _date(self.start),
            "end": _date(self.end),
            "days_left": self.days_left(today) if today is not None else None,
            "note": self.note,
            "gpu_hours_per_unit": self.gpu_hours_per_unit,
            "gpu_hours_total": self.gpu_hours_total,
            "gpu_hours_remaining": self.gpu_hours_remaining,
            "month_gpu_hours_total": self.month_gpu_hours_total,
            "month_gpu_hours_remaining": self.month_gpu_hours_remaining,
        }


@dataclass
class StorageQuota:
    """Usage and quota of a storage area (file system, directory or fileset)."""

    #: Path (or other name) of the storage area.
    name: str
    #: Project owning the area, or ``None`` for personal areas such as $HOME.
    project: Optional[str] = None
    used_bytes: Optional[int] = None
    #: The (soft) quota of the area in bytes, ``None`` when unlimited/unknown.
    limit_bytes: Optional[int] = None
    files_used: Optional[int] = None
    files_limit: Optional[int] = None
    note: Optional[str] = None

    @property
    def free_bytes(self) -> Optional[int]:
        """Bytes left before reaching the quota."""
        if self.limit_bytes is None or self.used_bytes is None:
            return None
        return self.limit_bytes - self.used_bytes

    @property
    def used_fraction(self) -> Optional[float]:
        """Fraction (0-1) of the quota that is in use."""
        if self.used_bytes is None or not self.limit_bytes:
            return None
        return self.used_bytes / self.limit_bytes

    @property
    def files_fraction(self) -> Optional[float]:
        """Fraction (0-1) of the file-count quota that is in use."""
        if self.files_used is None or not self.files_limit:
            return None
        return self.files_used / self.files_limit

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation, including derived values."""
        return {
            "name": self.name,
            "project": self.project,
            "used_bytes": self.used_bytes,
            "limit_bytes": self.limit_bytes,
            "free_bytes": self.free_bytes,
            "used_fraction": self.used_fraction,
            "files_used": self.files_used,
            "files_limit": self.files_limit,
            "files_fraction": self.files_fraction,
            "note": self.note,
        }


@dataclass
class Job:
    """A batch job of the user, as reported by the scheduler."""

    job_id: str
    name: str = ""
    partition: str = ""
    account: str = ""
    state: str = ""
    #: Time the job has been running, in the scheduler's format.
    elapsed: str = ""
    #: Wall-time limit of the job, in the scheduler's format.
    time_limit: str = ""
    nodes: Optional[int] = None
    #: Allocated nodes (running jobs) or the reason a job is pending.
    reason: str = ""

    @property
    def is_running(self) -> bool:
        """Whether the job is currently holding resources."""
        return self.state.strip().upper() in RUNNING_STATES

    @property
    def is_pending(self) -> bool:
        """Whether the job is waiting in the queue."""
        return self.state.strip().upper() in PENDING_STATES

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "job_id": self.job_id,
            "name": self.name,
            "partition": self.partition,
            "account": self.account,
            "state": self.state,
            "elapsed": self.elapsed,
            "time_limit": self.time_limit,
            "nodes": self.nodes,
            "reason": self.reason,
        }


@dataclass
class ClusterReport:
    """Everything collected from one cluster."""

    #: Name of the cluster in the configuration (e.g. ``"lumi"``).
    name: str
    #: Human readable name of the system (e.g. ``"LUMI"``).
    system: str
    #: The GPUs of a GPU node of the cluster (e.g. ``"4x NVIDIA A100 64 GB"``),
    #: ``None`` when unknown.
    gpus: Optional[str] = None
    budgets: List[ComputeBudget] = field(default_factory=list)
    storage: List[StorageQuota] = field(default_factory=list)
    #: The user's jobs, ``None`` when jobs were not collected.
    jobs: Optional[List[Job]] = None
    #: Problems that prevented collecting (part of) the data.
    errors: List[str] = field(default_factory=list)
    #: Non-fatal problems encountered while collecting the data.
    warnings: List[str] = field(default_factory=list)
    #: Sections whose data could not be collected (see ``errors``).
    failed_sections: List[str] = field(default_factory=list)
    #: Sections (``compute``, ``storage``, ``jobs``) that were requested.
    sections: List[str] = field(default_factory=list)
    collected_at: Optional[dt.datetime] = None

    @property
    def ok(self) -> bool:
        """Whether the data was collected without any error."""
        return not self.errors

    @property
    def running_jobs(self) -> List[Job]:
        """The jobs that are currently running."""
        return [job for job in self.jobs or [] if job.is_running]

    @property
    def pending_jobs(self) -> List[Job]:
        """The jobs that are waiting in the queue."""
        return [job for job in self.jobs or [] if job.is_pending]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        today = self.collected_at.date() if self.collected_at else None
        return {
            "name": self.name,
            "system": self.system,
            "gpus": self.gpus,
            "collected_at": self.collected_at.isoformat(timespec="seconds") if self.collected_at else None,
            "sections": list(self.sections),
            "failed_sections": list(self.failed_sections),
            "budgets": [budget.to_dict(today) for budget in self.budgets],
            "storage": [quota.to_dict() for quota in self.storage],
            "jobs": None if self.jobs is None else [job.to_dict() for job in self.jobs],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }
