"""Loading and validation of the hpcmon configuration file."""

from __future__ import annotations

import datetime as dt
import math
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from hpcmon.units import parse_date

#: Sections of information that can be collected from a cluster.
SECTIONS = ("compute", "storage", "jobs")

#: Default limit, in seconds, for querying a cluster.  Some tools are slow
#: (e.g. ``cindata`` on Leonardo takes more than half a minute).
DEFAULT_TIMEOUT = 180.0

#: Clusters configured when no configuration file exists: name -> type.
DEFAULT_CLUSTERS = ("lumi", "jupiter", "leonardo", "meluxina")

#: Name of the environment variable that points to the configuration file.
CONFIG_ENV_VAR = "HPCMON_CONFIG"

DEFAULT_CONFIG_TEMPLATE = """\
# hpcmon configuration file.
#
# Every [clusters.<name>] table defines a cluster to query.  <name> is the name
# shown in the output and used on the command line (e.g. `hpcmon lumi`).

[settings]
# Maximum time, in seconds, to wait for each cluster.
timeout = 180
# Query all clusters at the same time.  Set to false if your SSH connections
# need interactive authentication (password, OTP), so that prompts don't mix.
parallel = true
# Run the remote commands in a login shell (loads the site's profile/PATH).
login_shell = true

[clusters.lumi]
# The type of system: lumi, jupiter, leonardo, meluxina or slurm (generic
# Slurm cluster, jobs only).  It can be omitted when <name> is the type.
type = "lumi"
# The command used to connect to the cluster (a string or a list of arguments).
ssh = "ssh lumi"
# Set to false to skip the cluster unless it is named on the command line.
enabled = true
# Only show these projects (default: all your projects).
# projects = ["project_465000000"]
# The GPUs of a GPU node, shown next to the cluster name.  Built in for the
# supported systems; set it to describe other clusters ("" hides it).
# gpus = "4× AMD MI250X 128 GB (2 GCDs each)"
# GPU-hours bought by one unit of each budget, by the name of its resource
# (the Resource column without the [unit], e.g. GPU, Booster or
# jupiter_booster), replacing the built-in estimates; 0 disables one.
# [clusters.lumi.gpu_hours_per_unit]
# GPU = 1.0

[clusters.jupiter]
type = "jupiter"
ssh = "ssh jupiter"

[clusters.leonardo]
type = "leonardo"
ssh = "ssh leonardo"

[clusters.meluxina]
type = "meluxina"
ssh = "ssh meluxina"
# MeluXina does not report when projects end.  Set the end dates yourself to
# have them displayed (these dates are only used when the cluster reports none).
# [clusters.meluxina.project_end]
# p200000 = 2027-06-30
"""

_SETTINGS_KEYS = {"timeout", "parallel", "login_shell"}
_CLUSTER_KEYS = {
    "type", "ssh", "enabled", "timeout", "login_shell", "projects", "project_end", "gpus", "gpu_hours_per_unit",
}


class ConfigError(Exception):
    """Raised for invalid or unreadable configuration files."""


@dataclass
class ClusterConfig:
    """Configuration of a single cluster."""

    #: Name of the cluster, as used on the command line and in the output.
    name: str
    #: Type of the system, selecting the implementation used to query it.
    type: str
    #: Command (argv) used to connect to the cluster.
    ssh: List[str]
    enabled: bool = True
    timeout: float = DEFAULT_TIMEOUT
    login_shell: bool = True
    #: Projects to display, ``None`` for all projects.
    projects: Optional[List[str]] = None
    #: End dates of projects, used when the cluster does not report them.
    project_end: Dict[str, dt.date] = field(default_factory=dict)
    #: Description of the GPUs of a node, replacing the built-in one (an
    #: empty string hides it), ``None`` for the built-in description.
    gpus: Optional[str] = None
    #: GPU-hours bought by one unit of a budget, by budget resource (e.g.
    #: ``"Booster"``), replacing the built-in rates; 0 disables the estimate.
    gpu_hours_per_unit: Dict[str, float] = field(default_factory=dict)

    def wants_project(self, project: Optional[str]) -> bool:
        """Whether information about ``project`` should be displayed."""
        if self.projects is None or project is None:
            return True
        return project.lower() in {name.lower() for name in self.projects}

    def end_date_for(self, project: str) -> Optional[dt.date]:
        """The configured end date of ``project`` (case-insensitive), if any."""
        for name, date in self.project_end.items():
            if name.lower() == project.lower():
                return date
        return None

    def gpu_rate_for(self, resource: str) -> Optional[float]:
        """The configured GPU-hours per unit of ``resource`` (case-insensitive), if any."""
        for name, rate in self.gpu_hours_per_unit.items():
            if name.lower() == resource.lower():
                return rate
        return None


@dataclass
class Config:
    """The complete hpcmon configuration."""

    clusters: Dict[str, ClusterConfig]
    parallel: bool = True
    #: The file the configuration was read from, ``None`` for defaults.
    path: Optional[Path] = None


def default_config_path(environ: Optional[Mapping[str, str]] = None) -> Path:
    """Return the configuration file location.

    It is ``$HPCMON_CONFIG`` if set, otherwise ``hpcmon/config.toml`` in
    ``$XDG_CONFIG_HOME`` (default ``~/.config``).
    """
    env = os.environ if environ is None else environ
    if env.get(CONFIG_ENV_VAR):
        return Path(env[CONFIG_ENV_VAR]).expanduser()
    base = env.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "hpcmon" / "config.toml"


def _known_types() -> Iterable[str]:
    from hpcmon.clusters import CLUSTER_TYPES

    return CLUSTER_TYPES.keys()


def default_config() -> Config:
    """The configuration used when there is no configuration file."""
    clusters = {name: ClusterConfig(name=name, type=name, ssh=["ssh", name]) for name in DEFAULT_CLUSTERS}
    return Config(clusters=clusters)


def load_config(path: Optional[Path] = None, *, required: bool = False) -> Config:
    """Load the configuration from ``path`` (default: :func:`default_config_path`).

    :param required: If true, a missing file is an error; otherwise the
        default configuration is returned.
    :raises ConfigError: if the file is missing (and required) or invalid.
    """
    path = default_config_path() if path is None else Path(path).expanduser()
    if not path.exists():
        if required:
            raise ConfigError(f"configuration file not found: {path}")
        return default_config()
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise ConfigError(f"{path}: invalid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc}") from exc
    config = parse_config(data, source=str(path))
    config.path = path
    return config


def write_default_config(path: Path, *, force: bool = False) -> Path:
    """Write the commented default configuration to ``path``.

    :raises ConfigError: if the file exists and ``force`` is false.
    """
    path = Path(path).expanduser()
    if path.exists() and not force:
        raise ConfigError(f"{path} already exists (use --force to overwrite it)")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"cannot write {path}: {exc}") from exc
    return path


def parse_config(data: Mapping[str, Any], source: str = "<config>") -> Config:
    """Build a :class:`Config` from the parsed contents of a TOML file.

    :raises ConfigError: if the contents are invalid.
    """
    unknown = set(data) - {"settings", "clusters"}
    if unknown:
        raise ConfigError(f"{source}: unknown top-level key(s): {', '.join(sorted(unknown))}")

    settings = _table(data.get("settings", {}), f"{source}: [settings]")
    _check_keys(settings, _SETTINGS_KEYS, f"{source}: [settings]")
    timeout = _positive_number(settings.get("timeout", DEFAULT_TIMEOUT), f"{source}: settings.timeout")
    parallel = _boolean(settings.get("parallel", True), f"{source}: settings.parallel")
    login_shell = _boolean(settings.get("login_shell", True), f"{source}: settings.login_shell")

    if "clusters" not in data:
        clusters = default_config().clusters
        for cluster in clusters.values():
            cluster.timeout = timeout
            cluster.login_shell = login_shell
        return Config(clusters=clusters, parallel=parallel)

    tables = _table(data["clusters"], f"{source}: [clusters]")
    clusters = {}
    for name, table in tables.items():
        where = f"{source}: [clusters.{name}]"
        clusters[name] = _parse_cluster(name, _table(table, where), where, timeout, login_shell)
    return Config(clusters=clusters, parallel=parallel)


def _parse_cluster(
    name: str, table: Mapping[str, Any], where: str, timeout: float, login_shell: bool
) -> ClusterConfig:
    _check_keys(table, _CLUSTER_KEYS, where)
    known = list(_known_types())

    cluster_type = table.get("type", name if name in known else None)
    if cluster_type is None:
        raise ConfigError(f"{where}: missing 'type' (one of: {', '.join(known)})")
    if cluster_type not in known:
        raise ConfigError(f"{where}: unknown type {cluster_type!r} (one of: {', '.join(known)})")

    ssh = table.get("ssh", ["ssh", name])
    if isinstance(ssh, str):
        try:
            ssh = shlex.split(ssh)
        except ValueError as exc:
            raise ConfigError(f"{where}: invalid 'ssh' command: {exc}") from exc
    if not isinstance(ssh, list) or not ssh or not all(isinstance(arg, str) and arg for arg in ssh):
        raise ConfigError(f"{where}: 'ssh' must be a non-empty string or list of strings")

    projects = table.get("projects")
    if projects is not None and (not isinstance(projects, list) or not all(isinstance(p, str) for p in projects)):
        raise ConfigError(f"{where}: 'projects' must be a list of strings")

    project_end = {}
    for project, value in _table(table.get("project_end", {}), f"{where} project_end").items():
        project_end[project] = _date(value, f"{where}: project_end.{project}")

    gpus = table.get("gpus")
    if gpus is not None and not isinstance(gpus, str):
        raise ConfigError(f"{where}: 'gpus' must be a string")

    gpu_hours_per_unit = {}
    for resource, value in _table(table.get("gpu_hours_per_unit", {}), f"{where} gpu_hours_per_unit").items():
        gpu_hours_per_unit[resource] = _non_negative_number(value, f"{where}: gpu_hours_per_unit.{resource}")

    return ClusterConfig(
        name=name,
        type=cluster_type,
        ssh=list(ssh),
        enabled=_boolean(table.get("enabled", True), f"{where}: enabled"),
        timeout=_positive_number(table.get("timeout", timeout), f"{where}: timeout"),
        login_shell=_boolean(table.get("login_shell", login_shell), f"{where}: login_shell"),
        projects=list(projects) if projects is not None else None,
        project_end=project_end,
        gpus=gpus,
        gpu_hours_per_unit=gpu_hours_per_unit,
    )


def _table(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a table")
    return value


def _check_keys(table: Mapping[str, Any], allowed: Iterable[str], where: str) -> None:
    unknown = set(table) - set(allowed)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s): {', '.join(sorted(unknown))}")


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{where} must be true or false")
    return value


def _positive_number(value: Any, where: str) -> float:
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        if math.isfinite(number) and number > 0:
            return number
    raise ConfigError(f"{where} must be a positive number")


def _non_negative_number(value: Any, where: str) -> float:
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        try:
            number = float(value)
        except OverflowError:
            number = math.inf
        if math.isfinite(number) and number >= 0:
            return number
    raise ConfigError(f"{where} must be a number (0 or more)")


def _date(value: Any, where: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            date = parse_date(value)
        except ValueError:
            date = None
        if date is not None:
            return date
    raise ConfigError(f"{where} must be a date (YYYY-MM-DD)")
