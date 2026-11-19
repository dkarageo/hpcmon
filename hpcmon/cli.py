"""Command line interface of hpcmon."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence

from rich.console import Console
from rich.markup import escape
from rich.text import Text
from rich.table import Table

from hpcmon import __version__
from hpcmon.clusters import CLUSTER_TYPES, Cluster, create_cluster
from hpcmon.config import (
    CONFIG_ENV_VAR,
    SECTIONS,
    Config,
    ConfigError,
    default_config_path,
    load_config,
    write_default_config,
)
from hpcmon.models import ClusterReport
from hpcmon.render import render_reports, reports_to_json
from hpcmon.runner import terminate_all

#: Exit status when every cluster was queried successfully.
EXIT_OK = 0
#: Exit status when the data of at least one cluster is incomplete.
EXIT_CLUSTER_ERROR = 1
#: Exit status for invalid usage or configuration.
EXIT_USAGE = 2
#: Exit status when interrupted with Ctrl-C (128 + SIGINT).
EXIT_INTERRUPTED = 130


def build_parser() -> argparse.ArgumentParser:
    """Create the parser of the command line arguments."""
    parser = argparse.ArgumentParser(
        prog="hpcmon",
        description=(
            "Display the remaining compute budgets, the storage quotas and the running jobs "
            "of your projects on EuroHPC clusters (LUMI, JUPITER, Leonardo, MeluXina, ...)."
        ),
        epilog=(
            "By default all the enabled clusters of the configuration file are queried and all "
            "sections are shown. Configuration file: $HPCMON_CONFIG or ~/.config/hpcmon/config.toml."
        ),
    )
    parser.add_argument("clusters", nargs="*", metavar="CLUSTER", help="names of the clusters to query (default: all enabled)")
    parser.add_argument("-c", "--config", type=Path, metavar="PATH", help="configuration file to use")
    sections = parser.add_argument_group("sections (default: all)")
    sections.add_argument("--compute", action="store_true", help="show the compute allocations")
    sections.add_argument("--storage", action="store_true", help="show the storage quotas")
    sections.add_argument("--jobs", action="store_true", help="show the jobs in the queue")
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    parser.add_argument("-t", "--timeout", type=float, metavar="SECONDS", help="maximum time to wait for each cluster")
    parser.add_argument("--sequential", action="store_true", help="query the clusters one after the other")
    parser.add_argument("--no-color", action="store_true", help="disable colours")
    parser.add_argument("--debug", action="store_true", help="log the raw output of the remote commands")
    parser.add_argument("--init-config", action="store_true", help="write a default configuration file and exit")
    parser.add_argument("--force", action="store_true", help="with --init-config, overwrite an existing file")
    parser.add_argument("--show-config", action="store_true", help="show the configured clusters and exit")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def select_clusters(config: Config, names: Sequence[str]) -> List[str]:
    """Return the names of the clusters to query.

    :param names: Names given on the command line; when empty, all the
        enabled clusters are selected.
    :raises ConfigError: for unknown names.
    """
    if not names:
        return [name for name, cluster in config.clusters.items() if cluster.enabled]
    unknown = [name for name in names if name not in config.clusters]
    if unknown:
        available = ", ".join(config.clusters) or "none"
        raise ConfigError(f"unknown cluster(s): {', '.join(unknown)} (configured: {available})")
    return list(dict.fromkeys(names))


def collect_reports(
    clusters: Sequence[Cluster], sections: Sequence[str], parallel: bool, now: dt.datetime
) -> Iterator[ClusterReport]:
    """Query ``clusters`` and yield their reports in the same order.

    With ``parallel``, all the clusters are queried at the same time and each
    report is yielded as soon as it and all the preceding ones are complete.
    """
    if not parallel or len(clusters) < 2:
        for cluster in clusters:
            yield _collect(cluster, sections, now)
        return
    pool = ThreadPoolExecutor(max_workers=len(clusters))
    try:
        futures = [pool.submit(_collect, cluster, sections, now) for cluster in clusters]
        for future in futures:
            yield future.result()
    finally:
        # Don't block (e.g. on Ctrl-C) until the remaining sessions end.
        pool.shutdown(wait=False, cancel_futures=True)


def _collect(cluster: Cluster, sections: Sequence[str], now: dt.datetime) -> ClusterReport:
    try:
        return cluster.collect(sections, now=now)
    except Exception as exc:  # noqa: BLE001 - report bugs per cluster rather than crash
        logging.getLogger(__name__).debug("%s: collection failed", cluster.name, exc_info=True)
        report = ClusterReport(
            name=cluster.name, system=cluster.display_name, gpus=cluster.gpu_label, sections=list(sections),
            collected_at=now,
        )
        report.errors.append(f"internal error ({type(exc).__name__}: {exc})")
        report.failed_sections = list(sections)
        return report


@contextlib.contextmanager
def preserve_terminal() -> Iterator[None]:
    """Restore the terminal settings on exit.

    An SSH command killed while prompting for a password (e.g. on timeout)
    can leave the terminal with echo disabled; this undoes that.
    """
    try:
        import termios
    except ImportError:  # pragma: no cover - not available on Windows
        yield
        return
    saved: Any = None
    if sys.stdin is not None and sys.stdin.isatty():
        try:
            saved = termios.tcgetattr(sys.stdin.fileno())
        except (termios.error, OSError, ValueError):
            saved = None
    try:
        yield
    finally:
        if saved is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, saved)
            except (termios.error, OSError, ValueError):
                pass


def show_config(config: Config, console: Console) -> None:
    """Print the configured clusters."""
    source = str(config.path) if config.path else f"built-in defaults (no file at {default_config_path()})"
    console.print(Text(f"Configuration: {source}"))
    table = Table("Name", "Type", "SSH command", "Enabled", "Timeout", "Projects")
    for cluster in config.clusters.values():
        table.add_row(
            Text(cluster.name),
            Text(cluster.type),
            Text(" ".join(cluster.ssh)),
            Text("yes" if cluster.enabled else "no"),
            Text(f"{cluster.timeout:g} s"),
            Text(", ".join(cluster.projects) if cluster.projects is not None else "all"),
        )
    console.print(table)
    console.print(Text(f"Supported types: {', '.join(CLUSTER_TYPES)}"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point of the ``hpcmon`` command."""
    parser = build_parser()
    # Intermixed parsing accepts cluster names after options (hpcmon --jobs lumi).
    args = parser.parse_intermixed_args(argv)
    sections = [section for section in SECTIONS if getattr(args, section)]
    if args.force and not args.init_config:
        parser.error("--force can only be used with --init-config")
    if args.init_config and (args.clusters or sections or args.show_config or args.json):
        parser.error("--init-config cannot be combined with clusters, sections, --show-config or --json")
    console = Console(no_color=args.no_color, highlight=False)
    err_console = Console(stderr=True, no_color=args.no_color, highlight=False)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    if args.init_config:
        try:
            path = write_default_config(args.config or default_config_path(), force=args.force)
        except ConfigError as exc:
            err_console.print(f"[bold red]error:[/] {escape(str(exc))}")
            return EXIT_USAGE
        console.print(Text(f"Configuration written to {path}"))
        return EXIT_OK

    try:
        explicit = args.config is not None or bool(os.environ.get(CONFIG_ENV_VAR))
        config = load_config(args.config, required=explicit)
        names = select_clusters(config, args.clusters)
    except ConfigError as exc:
        err_console.print(f"[bold red]error:[/] {escape(str(exc))}")
        return EXIT_USAGE

    if args.show_config:
        show_config(config, console)
        return EXIT_OK
    if not names:
        err_console.print("[bold red]error:[/] no cluster is enabled in the configuration")
        return EXIT_USAGE
    if args.timeout is not None and not (math.isfinite(args.timeout) and args.timeout > 0):
        err_console.print("[bold red]error:[/] the timeout must be positive")
        return EXIT_USAGE

    sections = sections or list(SECTIONS)
    clusters = []
    for name in names:
        cluster_config = config.clusters[name]
        if args.timeout is not None:
            cluster_config = dataclasses.replace(cluster_config, timeout=args.timeout)
        clusters.append(create_cluster(cluster_config))

    now = dt.datetime.now().astimezone()
    parallel = config.parallel and not args.sequential and len(clusters) > 1
    reports = collect_reports(clusters, sections, parallel=parallel, now=now)
    status_console = err_console if args.json else console
    # No spinner when querying sequentially (the mode meant for clusters that
    # prompt for a password or OTP): its redraws would hide the SSH prompts.
    spinner = (
        status_console.status(Text(f"Querying {', '.join(names)}..."))
        if parallel
        else contextlib.nullcontext()
    )
    try:
        with preserve_terminal(), spinner:
            if args.json:
                collected = list(reports)
            else:
                collected = render_reports(reports, console, today=now.date())
    except KeyboardInterrupt:
        terminate_all()
        err_console.print("[bold red]interrupted[/]")
        return EXIT_INTERRUPTED
    if args.json:
        sys.stdout.write(reports_to_json(collected, generated_at=now) + "\n")
    return EXIT_OK if all(report.ok for report in collected) else EXIT_CLUSTER_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
