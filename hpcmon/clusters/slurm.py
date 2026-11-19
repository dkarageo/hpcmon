"""Support for clusters scheduled by Slurm (all EuroHPC systems)."""

from __future__ import annotations

import datetime as dt
from typing import Dict, List

from hpcmon.clusters.base import Cluster, CollectionError, Results
from hpcmon.models import ComputeBudget, Job, StorageQuota
from hpcmon.units import parse_int

#: Fields requested from ``squeue``: job id, partition, account, state, elapsed
#: time, time limit, number of nodes, node list or pending reason, and name.
#: The name comes last, as it is the only field that may contain the delimiter.
SQUEUE_FORMAT = "%i|%P|%a|%T|%M|%l|%D|%R|%j"

#: Remote command listing the jobs of the user.
SQUEUE_COMMAND = f"squeue --noheader --user=\"$(id -un)\" --format='{SQUEUE_FORMAT}'"

_SQUEUE_FIELDS = SQUEUE_FORMAT.count("|") + 1


def parse_squeue(text: str) -> List[Job]:
    """Parse the output of :data:`SQUEUE_COMMAND`.

    Running jobs are listed first, then pending jobs, then any others; the
    order reported by ``squeue`` is kept otherwise.

    :raises CollectionError: if a line does not have the expected fields.
    """
    jobs = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("|", _SQUEUE_FIELDS - 1)
        if len(fields) != _SQUEUE_FIELDS:
            raise CollectionError(f"unexpected squeue output: {line.strip()!r}")
        job_id, partition, account, state, elapsed, limit, nodes, reason, name = (f.strip() for f in fields)
        try:
            node_count = parse_int(nodes)
        except ValueError:
            node_count = None
        jobs.append(
            Job(
                job_id=job_id,
                name=name,
                partition=partition,
                account=account,
                state=state,
                elapsed=elapsed,
                time_limit=limit,
                nodes=node_count,
                reason=reason,
            )
        )
    return sorted(jobs, key=lambda job: 0 if job.is_running else 1 if job.is_pending else 2)


class SlurmCluster(Cluster):
    """A cluster whose jobs are managed by Slurm.

    Subclasses still have to implement the compute and storage interfaces.
    """

    def jobs_commands(self) -> Dict[str, str]:
        return {"jobs": SQUEUE_COMMAND}

    def parse_jobs(self, results: Results, today: dt.date) -> List[Job]:
        return parse_squeue(self.output(results, "jobs"))


class GenericSlurmCluster(SlurmCluster):
    """Any Slurm cluster without a dedicated implementation (jobs only)."""

    type_name = "slurm"
    display_name = "Slurm cluster"
    supported_sections = ("jobs",)

    def compute_commands(self) -> Dict[str, str]:
        return {}

    def storage_commands(self) -> Dict[str, str]:
        return {}

    def parse_compute(self, results: Results, today: dt.date) -> List[ComputeBudget]:
        return []

    def parse_storage(self, results: Results, today: dt.date) -> List[StorageQuota]:
        return []
