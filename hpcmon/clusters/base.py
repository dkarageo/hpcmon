"""Abstract interface implemented by every supported cluster."""

from __future__ import annotations

import datetime as dt
import logging
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, TypeVar

from hpcmon.config import SECTIONS, ClusterConfig
from hpcmon.models import ClusterReport, ComputeBudget, Job, StorageQuota
from hpcmon.runner import TIMEOUT_EXIT_CODE, CommandResult, RemoteError, SSHRunner, last_line

logger = logging.getLogger(__name__)

#: The results of the remote commands, by command key.
Results = Mapping[str, CommandResult]

#: Remote command printing the home directory of the user.
HOME_COMMAND = 'printf "%s\\n" "$HOME"'

#: Order in which the commands of the sections run: the fast ones first, so
#: that a slow storage tool cannot use up the time limit of the others.
EXECUTION_ORDER = ("jobs", "compute", "storage")

T = TypeVar("T")


class CollectionError(Exception):
    """Raised when the output of a remote command is missing or unusable."""


class Cluster(ABC):
    """Base class of the cluster-specific interfaces.

    A cluster declares the remote commands needed for every section of
    information (compute budgets, storage quotas and jobs) and parses their
    output into the cluster-independent model of :mod:`hpcmon.models`.  All the
    commands of a query are executed through a single SSH session.

    Parsers raise :class:`CollectionError` (or any other exception) when the
    data of their section cannot be obtained; this is reported for that
    section only, without affecting the others.  Non-fatal problems can be
    reported with :meth:`warn`.
    """

    #: Identifier of the cluster type, used for ``type`` in the configuration.
    type_name: ClassVar[str] = ""
    #: Human-readable name of the system.
    display_name: ClassVar[str] = ""
    #: Sections of information that this kind of cluster can provide.
    supported_sections: ClassVar[Tuple[str, ...]] = SECTIONS
    #: The GPUs of a GPU node, e.g. ``"4x NVIDIA A100 64 GB"`` (``None`` if
    #: unknown); the ``gpus`` setting replaces it.
    gpus: ClassVar[Optional[str]] = None
    #: Estimated GPU-hours bought by one unit of a budget, by the
    #: ``(resource, unit)`` of the budget (the resource is compared without
    #: case, the unit exactly).  Budgets missing here (CPU-only partitions,
    #: storage billing, unknown units) get no estimate.  The rates assume that
    #: jobs use all the GPUs of the nodes they are charged for: on shared
    #: nodes, no more than the share of cores and memory of their GPUs.
    gpu_rates: ClassVar[Mapping[Tuple[str, str], float]] = {}

    def __init__(self, config: ClusterConfig, runner: Optional[SSHRunner] = None) -> None:
        self.config = config
        self.runner = runner or SSHRunner(config.ssh, timeout=config.timeout, login_shell=config.login_shell)
        self._commands: Dict[str, str] = {}
        self._warnings: List[str] = []

    @property
    def name(self) -> str:
        """Name of the cluster in the configuration."""
        return self.config.name

    @property
    def gpu_label(self) -> Optional[str]:
        """Description of the GPUs of a node: the setting, else the built-in one."""
        label = self.config.gpus if self.config.gpus is not None else self.gpus
        return (label or "").strip() or None

    def gpu_rate(self, budget: ComputeBudget) -> Optional[float]:
        """GPU-hours bought by one unit of ``budget`` (``None``: not convertible).

        A rate in the ``gpu_hours_per_unit`` setting takes precedence (0
        disables the estimate); otherwise :attr:`gpu_rates` is used.
        """
        configured = self.config.gpu_rate_for(budget.resource)
        if configured is not None:
            return configured or None
        for (resource, unit), rate in self.gpu_rates.items():
            if resource.lower() == budget.resource.lower() and unit == budget.unit:
                return rate
        return None

    # -- interface to implement -------------------------------------------

    @abstractmethod
    def compute_commands(self) -> Dict[str, str]:
        """Remote commands (by unique key) needed by :meth:`parse_compute`."""

    @abstractmethod
    def storage_commands(self) -> Dict[str, str]:
        """Remote commands (by unique key) needed by :meth:`parse_storage`."""

    @abstractmethod
    def jobs_commands(self) -> Dict[str, str]:
        """Remote commands (by unique key) needed by :meth:`parse_jobs`."""

    @abstractmethod
    def parse_compute(self, results: Results, today: dt.date) -> List[ComputeBudget]:
        """Extract the compute budgets of the user's projects."""

    @abstractmethod
    def parse_storage(self, results: Results, today: dt.date) -> List[StorageQuota]:
        """Extract the storage quotas of the user and their projects."""

    @abstractmethod
    def parse_jobs(self, results: Results, today: dt.date) -> List[Job]:
        """Extract the jobs of the user."""

    # -- helpers for implementations --------------------------------------

    def output(self, results: Results, key: str, *, check: bool = True) -> str:
        """Return the standard output of the command ``key``.

        :param check: If true, a non-zero exit status is an error.
        :raises CollectionError: if the command did not run, timed out or
            (with ``check``) failed.
        """
        command = self._commands.get(key, key)
        result = results.get(key)
        if result is None:
            raise CollectionError(f"no output from `{command}`")
        if result.returncode is None:
            raise CollectionError(f"`{command}` did not complete: {result.reason or 'no exit status'}")
        if result.returncode == TIMEOUT_EXIT_CODE:
            raise CollectionError(f"`{command}` timed out (time limit of {self.config.timeout:g} s reached)")
        if check and result.returncode != 0:
            raise CollectionError(f"`{command}` failed (exit status {result.returncode}){_detail(result)}")
        return result.stdout

    def warn(self, message: str) -> None:
        """Report a non-fatal problem in the current collection."""
        self._warnings.append(message)

    # -- collection -------------------------------------------------------

    def commands(self, sections: Iterable[str] = SECTIONS) -> Dict[str, str]:
        """All the remote commands needed for ``sections``, in execution order."""
        providers = {
            "compute": self.compute_commands,
            "storage": self.storage_commands,
            "jobs": self.jobs_commands,
        }
        wanted = set(sections)
        commands: Dict[str, str] = {}
        for section in EXECUTION_ORDER:
            if section in wanted:
                for key, command in providers[section]().items():
                    commands.setdefault(key, command)
        return commands

    def collect(
        self,
        sections: Sequence[str] = SECTIONS,
        today: Optional[dt.date] = None,
        now: Optional[dt.datetime] = None,
    ) -> ClusterReport:
        """Query the cluster and return everything found for ``sections``.

        This never raises for problems with the cluster: they are recorded in
        the ``errors`` of the returned report instead.
        """
        unknown = [section for section in sections if section not in SECTIONS]
        if unknown:
            raise ValueError(f"unknown section(s): {', '.join(unknown)}")
        sections = [section for section in SECTIONS if section in sections and section in self.supported_sections]
        now = now or dt.datetime.now().astimezone()
        today = today or now.date()
        report = ClusterReport(
            name=self.name, system=self.display_name, gpus=self.gpu_label, sections=sections, collected_at=now
        )
        self._commands = self.commands(sections)
        self._warnings = []
        if self.config.gpu_hours_per_unit and "compute" not in self.supported_sections:
            self.warn("the gpu_hours_per_unit setting is ignored: this type of cluster reports no compute budgets")
        try:
            results = self.runner.run(self._commands)
        except RemoteError as exc:
            report.errors.append(str(exc))
            report.failed_sections = list(sections)
            return report
        for key, result in results.items():
            logger.debug(
                "%s: `%s` exited with %s\n--- stdout ---\n%s\n--- stderr ---\n%s",
                self.name, self._commands[key], result.returncode, result.stdout, result.stderr,
            )

        # Names of the projects known to the cluster, to detect typos in the
        # "projects" setting (jobs are not used: the queue may be empty).
        seen: Set[str] = set()
        if "compute" in sections:
            budgets = self._guard(report, "compute", self.parse_compute, results, today) or []
            seen.update(name for budget in budgets for name in (budget.project, budget.account) if name)
            report.budgets = [self._complete(budget) for budget in budgets if self._wants_budget(budget)]
            if "compute" not in report.failed_sections:
                self._check_gpu_rates(budgets)
        if "storage" in sections:
            quotas = self._guard(report, "storage", self.parse_storage, results, today) or []
            seen.update(quota.project for quota in quotas if quota.project)
            report.storage = [quota for quota in quotas if self.config.wants_project(quota.project)]
        if "jobs" in sections:
            jobs = self._guard(report, "jobs", self.parse_jobs, results, today)
            if jobs is not None:
                report.jobs = [job for job in jobs if self.config.wants_project(job.account or None)]
                hidden = len(jobs) - len(report.jobs)
                if hidden:
                    self.warn(f"{hidden} job(s) of other accounts hidden by the 'projects' setting")
        checked = {"compute", "storage"} & set(sections)
        if self.config.projects is not None and checked and not report.failed_sections:
            known = {name.lower() for name in seen}
            missing = [name for name in self.config.projects if name.lower() not in known]
            if missing:
                self.warn(f"project(s) not found on this cluster: {', '.join(missing)}")
        report.warnings.extend(self._warnings)
        return report

    def _check_gpu_rates(self, budgets: List[ComputeBudget]) -> None:
        """Warn about rates of the ``gpu_hours_per_unit`` setting that apply to
        no budget of the cluster (e.g. typos in the resource names)."""
        resources = sorted({budget.resource for budget in budgets})
        unused = [name for name in self.config.gpu_hours_per_unit if name.lower() not in {r.lower() for r in resources}]
        if unused:
            known = f" (budget resources: {', '.join(resources)})" if resources else ""
            self.warn(f"gpu_hours_per_unit: no budget of resource {', '.join(unused)} on this cluster{known}")

    def _wants_budget(self, budget: ComputeBudget) -> bool:
        if self.config.wants_project(budget.project):
            return True
        return budget.account is not None and self.config.wants_project(budget.account)

    def _guard(
        self,
        report: ClusterReport,
        section: str,
        parser: Callable[[Results, dt.date], List[T]],
        results: Results,
        today: dt.date,
    ) -> Optional[List[T]]:
        try:
            return parser(results, today)
        except CollectionError as exc:
            report.errors.append(f"{section}: {exc}")
        except Exception as exc:  # noqa: BLE001 - one broken parser must not hide the rest
            logger.debug("%s: cannot parse %s", self.name, section, exc_info=True)
            report.errors.append(f"{section}: unexpected output ({type(exc).__name__}: {exc})")
        report.failed_sections.append(section)
        return None

    def _complete(self, budget: ComputeBudget) -> ComputeBudget:
        if budget.end is None:
            budget.end = self.config.end_date_for(budget.project)
        # A rate set by the parser is kept, unless the configuration has one.
        if budget.gpu_hours_per_unit is None or self.config.gpu_rate_for(budget.resource) is not None:
            budget.gpu_hours_per_unit = self.gpu_rate(budget)
        return budget


def _detail(result: CommandResult) -> str:
    line = last_line(result.stderr, result.stdout)
    return f": {line}" if line else ""
