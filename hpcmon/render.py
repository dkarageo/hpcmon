"""Presentation of the collected reports: rich terminal tables and JSON."""

from __future__ import annotations

import datetime as dt
import json
import re
from typing import Iterable, List, Optional, Sequence

from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from hpcmon.models import ClusterReport, ComputeBudget, Job, StorageQuota
from hpcmon.units import format_amount, format_bytes, format_count

#: Terminal escape sequences (CSI, OSC, other ESC sequences) and control
#: characters, which must not reach the terminal from remote strings.
_CONTROL_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI sequences, e.g. colours
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC sequences, e.g. window title
    r"|\x1b(?:[ -/]*[0-~])?"  # other escape sequences, e.g. ESC ( B, ESC 7, ESC c
    r"|[\x00-\x08\x0b-\x1f\x7f-\x9f]"  # remaining control characters
)


def clean(text: Optional[str]) -> str:
    """Remove terminal control sequences from a string coming from a cluster."""
    if not text:
        return ""
    return _CONTROL_RE.sub("", text.replace("\n", " ").replace("\t", " "))


def _text(value: Optional[str], style: str = "") -> Text:
    return Text(clean(value), style=style)


def _left_style(fraction: Optional[float]) -> str:
    """Colour of a "remaining" percentage: the less is left, the redder."""
    if fraction is None:
        return ""
    if fraction >= 0.5:
        return "green"
    if fraction >= 0.2:
        return "yellow"
    return "red"


def _usage_style(fraction: Optional[float]) -> str:
    """Colour of a "used" percentage: the more is used, the redder."""
    if fraction is None:
        return ""
    if fraction < 0.8:
        return "green"
    if fraction < 0.95:
        return "yellow"
    return "red"


def _table(title: str) -> Table:
    return Table(title=title, title_justify="left", title_style="bold", box=box.SIMPLE_HEAD, caption_justify="left")


def _column(table: Table, header: str, number: bool = False) -> None:
    """Add a column (numbers are right-aligned).

    Values wrap at spaces, and "fold" breaks longer words over several lines
    rather than truncating them: no identifier or number is ever cut with an
    ellipsis on narrow terminals.
    """
    table.add_column(header, justify="right" if number else "left", overflow="fold")


def _shared_note(notes: Sequence[Optional[str]]) -> Optional[str]:
    """The note common to all rows, if they all have the same one."""
    unique = set(notes)
    if len(unique) == 1 and None not in unique:
        return unique.pop()
    return None


def format_end(budget: ComputeBudget, today: dt.date) -> Text:
    """Format the end of the allocation period and the days left until it."""
    days = budget.days_left(today)
    if budget.end is None or days is None:
        return Text("-", style="dim")
    end = budget.end.isoformat()
    if days < 0:
        return Text(f"{end} (ended)", style="red")
    if days == 0:
        return Text(f"{end} (last day)", style="bold red")
    style = "red" if days <= 14 else "yellow" if days <= 30 else ""
    return Text(f"{end} ({days}d)", style=style)


def short_percent(fraction: Optional[float]) -> str:
    """Format a fraction as a compact percentage (decimals only below 10%)."""
    if fraction is None:
        return "-"
    percent = fraction * 100
    return f"{percent:.1f}%" if abs(percent) < 9.95 else f"{percent:.0f}%"


def _amount(value: Optional[float]) -> Text:
    return Text(format_amount(value)) if value is not None else Text("-", style="dim")


def _remaining(
    value: Optional[float], fraction: Optional[float], ended: bool = False, gpu_hours: Optional[float] = None
) -> Text:
    """A remaining amount, with its percentage and, below, its GPU-hours estimate."""
    if value is None:
        return Text("-", style="dim")
    text = format_amount(value)
    if fraction is not None:
        text += f" ({short_percent(fraction)})"
    result = Text(text, style="dim" if ended else _left_style(fraction) or "bold")
    if gpu_hours is not None:
        result.append(f"\n{format_gpu_hours(gpu_hours)}", style="dim" if ended else GPU_STYLE)
    return result


#: Style of the GPU information (node GPUs and GPU-hour estimates).
GPU_STYLE = "magenta"


def format_gpu_hours(value: float) -> str:
    """Format an estimate of GPU-hours, e.g. ``"~17,602 GPU-h"``."""
    return f"~{format_amount(value)} GPU-h"


def _gpu_estimate(budget: ComputeBudget, amount: Optional[float]) -> Optional[float]:
    """The GPU-hours of ``amount``, unless it is the very same number."""
    if budget.unit == "GPU-h" and budget.gpu_hours_per_unit == 1:
        return None
    return budget.gpu_hours(amount)


def _project(budget: ComputeBudget) -> Text:
    if budget.account:
        return Text(f"{clean(budget.project)} ({clean(budget.account)})")
    return _text(budget.project)


def compute_table(budgets: List[ComputeBudget], today: dt.date) -> Table:
    """Table of the compute budgets of a cluster."""
    monthly = any(b.month_total is not None or b.month_used is not None for b in budgets)
    shared_note = _shared_note([b.note for b in budgets])
    notes = shared_note is None and any(b.note for b in budgets)
    table = _table("Compute allocations")
    _column(table, "Project")
    _column(table, "Resource")
    _column(table, "Total", number=True)
    _column(table, "Remaining", number=True)
    if monthly:
        _column(table, "Month\ntotal", number=True)
        _column(table, "Month\nremaining", number=True)
    _column(table, "Ends (days left)", number=True)
    if notes:
        _column(table, "Notes")
    if shared_note:
        table.caption = Text(f"Note: {clean(shared_note)}", style="dim")
    for budget in budgets:
        ended = budget.has_ended(today)
        row = [
            _project(budget),
            _text(f"{budget.resource} [{budget.unit}]"),
            _amount(budget.total),
            _remaining(budget.remaining, budget.remaining_fraction, ended, _gpu_estimate(budget, budget.remaining)),
        ]
        if monthly:
            row += [
                _amount(budget.month_total),
                _remaining(
                    budget.month_remaining,
                    budget.month_remaining_fraction,
                    ended,
                    _gpu_estimate(budget, budget.month_remaining),
                ),
            ]
        row.append(format_end(budget, today))
        if notes:
            row.append(_text(budget.note, style="dim"))
        table.add_row(*row)
    return table


def _limit(quota: StorageQuota) -> Text:
    if quota.limit_bytes is not None:
        return Text(format_bytes(quota.limit_bytes))
    if quota.used_bytes is not None:
        return Text("no quota", style="dim")
    return Text("-", style="dim")


def _files(used: Optional[int], limit: Optional[int], fraction: Optional[float]) -> Text:
    if used is None and limit is None:
        return Text("-", style="dim")
    return Text(f"{format_count(used)} / {format_count(limit)}", style=_usage_style(fraction))


def storage_table(quotas: List[StorageQuota]) -> Table:
    """Table of the storage quotas of a cluster."""
    shared_note = _shared_note([q.note for q in quotas])
    notes = shared_note is None and any(q.note for q in quotas)
    table = _table("Storage")
    _column(table, "Area")
    for header in ("Used", "Quota", "Free", "Use", "Files / quota"):
        _column(table, header, number=True)
    if notes:
        _column(table, "Notes")
    if shared_note:
        table.caption = Text(f"Note: {clean(shared_note)}", style="dim")
    for quota in quotas:
        fraction = quota.used_fraction
        files_fraction = quota.files_fraction
        row = [
            _text(quota.name),
            Text(format_bytes(quota.used_bytes)),
            _limit(quota),
            Text(format_bytes(quota.free_bytes), style="bold"),
            Text(short_percent(fraction), style=_usage_style(fraction)),
            _files(quota.files_used, quota.files_limit, files_fraction),
        ]
        if notes:
            row.append(_text(quota.note, style="dim"))
        table.add_row(*row)
    return table


def jobs_title(jobs: List[Job]) -> str:
    """Summary line of the jobs of a cluster."""
    running = sum(job.is_running for job in jobs)
    pending = sum(job.is_pending for job in jobs)
    other = len(jobs) - running - pending
    title = f"Jobs: {running} running, {pending} pending"
    return f"{title}, {other} other" if other else title


def jobs_table(jobs: List[Job]) -> Table:
    """Table of the jobs of the user on a cluster.

    The account column is only shown when the jobs use several accounts.
    """
    accounts = len({job.account for job in jobs}) > 1
    table = _table(jobs_title(jobs))
    for column in ("Job ID", "Name") + (("Account",) if accounts else ()) + ("Partition", "State"):
        _column(table, column)
    _column(table, "Time / limit", number=True)
    _column(table, "Nodes", number=True)
    _column(table, "Node list / reason")
    for job in jobs:
        style = "green" if job.is_running else "yellow" if job.is_pending else ""
        row = [_text(job.job_id), _text(job.name)]
        if accounts:
            row.append(_text(job.account))
        row += [
            _text(job.partition),
            _text(job.state, style=style),
            _text(f"{job.elapsed} / {job.time_limit}"),
            Text("-" if job.nodes is None else str(job.nodes)),
            _text(job.reason),
        ]
        table.add_row(*row)
    return table


def cluster_name(report: ClusterReport) -> str:
    """The name of the system, with the configured name when it differs."""
    if report.system.lower() == report.name.lower():
        return report.system
    return f"{report.system} ({report.name})"


def _gpu_label(report: ClusterReport) -> str:
    return clean(report.gpus).strip()


#: ASCII replacements of the characters of the titles, for terminals that
#: cannot print anything else (other characters become "?").
_ASCII = str.maketrans({"·": "-", "×": "x"})


def _printable(text: Text, ascii_only: bool) -> Text:
    """``text``, made ASCII-only if the terminal requires it."""
    if ascii_only:
        # Same length: the styles of the text keep applying to the same characters.
        text.plain = text.plain.translate(_ASCII).encode("ascii", "replace").decode("ascii")
    return text


def report_title(report: ClusterReport, ascii_only: bool = False) -> Text:
    """The name of the cluster, followed by the GPUs of its nodes."""
    title = _text(cluster_name(report), style="bold cyan")
    label = _gpu_label(report)
    if label:
        title.append(f" · GPU nodes: {label}", style=GPU_STYLE)
    return _printable(title, ascii_only)


def render_title(report: ClusterReport, console: Console) -> None:
    """Print the rule introducing a cluster.

    The GPUs of the nodes go on a line of their own when the rule is too narrow
    for them, rather than being cut.
    """
    ascii_only = console.options.ascii_only
    title = report_title(report, ascii_only)
    # A left-aligned rule cuts titles longer than its width minus 2.
    if title.cell_len <= console.width - 2:
        console.print(Rule(title, align="left", style="cyan"))
        return
    name = _printable(_text(cluster_name(report), style="bold cyan"), ascii_only)
    console.print(Rule(name, align="left", style="cyan"))
    if _gpu_label(report):
        console.print(_printable(Text(f"GPU nodes: {_gpu_label(report)}", style=GPU_STYLE), ascii_only))


def render_report(report: ClusterReport, console: Console, today: dt.date) -> None:
    """Print everything collected from one cluster."""
    render_title(report, console)
    if "compute" in report.sections and "compute" not in report.failed_sections:
        if report.budgets:
            console.print(compute_table(report.budgets, today))
        else:
            console.print(Text("No compute allocations found.", style="dim"))
    if "storage" in report.sections and "storage" not in report.failed_sections:
        if report.storage:
            console.print(storage_table(report.storage))
        else:
            console.print(Text("No storage quotas found.", style="dim"))
    if "jobs" in report.sections and report.jobs is not None:
        if report.jobs:
            console.print(jobs_table(report.jobs))
        else:
            console.print(Text("No jobs in the queue.", style="dim"))
    for warning in report.warnings:
        console.print(Text(f"warning: {clean(warning)}", style="yellow"))
    for error in report.errors:
        console.print(Text(f"error: {clean(error)}", style="bold red"))
    console.print()


#: Assumption behind the GPU-hour estimates, shown below the summary.
GPU_HOURS_ASSUMPTION = (
    "GPU-hours (of the GPUs named next to each cluster) are estimates: they assume that jobs use all the "
    "GPUs of the nodes they are charged for (on shared nodes, no more than the cores and memory that come "
    "with their GPUs)."
)


def _gpu_budgets(report: ClusterReport) -> List[ComputeBudget]:
    if "compute" not in report.sections or "compute" in report.failed_sections:
        return []
    return [budget for budget in report.budgets if budget.gpu_hours_per_unit is not None]


def gpu_summary_table(reports: Sequence[ClusterReport], today: dt.date) -> Optional[Table]:
    """The budgets of all ``reports`` in estimated GPU-hours.

    :returns: ``None`` when no budget of the reports can be spent on GPUs.
    """
    rows = [(report, budget) for report in reports for budget in _gpu_budgets(report)]
    if not rows:
        return None
    monthly = any(budget.month_gpu_hours_remaining is not None for _, budget in rows)
    table = _table("GPU-hours left (estimated)")
    table.pad_edge = False  # fits 80 columns
    table.collapse_padding = True
    _column(table, "Cluster")
    _column(table, "Project")
    _column(table, "Remaining", number=True)
    if monthly:
        _column(table, "Month\nremaining", number=True)
    _column(table, "Ends (days left)", number=True)
    table.caption = Text(f"Note: {GPU_HOURS_ASSUMPTION}", style="dim")
    for report, budget in rows:
        ended = budget.has_ended(today)
        row = [
            _text(cluster_name(report)),
            _project(budget),
            _remaining(budget.gpu_hours_remaining, budget.remaining_fraction, ended),
        ]
        if monthly:
            row.append(_remaining(budget.month_gpu_hours_remaining, budget.month_remaining_fraction, ended))
        row.append(format_end(budget, today))
        table.add_row(*row)
    return table


def render_gpu_summary(reports: Sequence[ClusterReport], console: Console, today: dt.date) -> None:
    """Print the GPU-hours summary when it covers more than one cluster."""
    if sum(1 for report in reports if _gpu_budgets(report)) < 2:
        return
    table = gpu_summary_table(reports, today)
    console.print(Rule(Text("All clusters", style="bold cyan"), align="left", style="cyan"))
    console.print(table)
    console.print()


def render_reports(reports: Iterable[ClusterReport], console: Console, today: dt.date) -> List[ClusterReport]:
    """Print all ``reports`` as they become available, then the GPU-hours
    summary of all of them, and return them."""
    rendered = []
    for report in reports:
        render_report(report, console, today)
        rendered.append(report)
    render_gpu_summary(rendered, console, today)
    return rendered


def reports_to_json(reports: Iterable[ClusterReport], generated_at: dt.datetime) -> str:
    """Serialise ``reports`` to a JSON document."""
    document = {
        "generated_at": generated_at.isoformat(timespec="seconds"),
        "clusters": [report.to_dict() for report in reports],
    }
    return json.dumps(document, indent=2)
