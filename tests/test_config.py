"""Tests for :mod:`hpcmon.config`: locating, loading, writing and validating the configuration.

Error messages are compared exactly (with ``==``, not regular expressions)
because they are what the user sees on the command line.  Messages that embed
text produced by the TOML parser or the operating system are only checked up
to the part hpcmon controls.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from hpcmon.clusters import CLUSTER_TYPES
from hpcmon.config import (
    CONFIG_ENV_VAR,
    DEFAULT_CLUSTERS,
    DEFAULT_CONFIG_TEMPLATE,
    DEFAULT_TIMEOUT,
    SECTIONS,
    ClusterConfig,
    Config,
    ConfigError,
    default_config,
    default_config_path,
    load_config,
    parse_config,
    write_default_config,
)
from tests.conftest import make_config

KNOWN_TYPES = "lumi, jupiter, leonardo, meluxina, slurm"


def parse(text: str, source: str = "<config>") -> Config:
    """Parse TOML ``text`` and build the configuration from it."""
    return parse_config(tomllib.loads(text), source=source)


def parse_error(text: str, source: str = "<config>") -> str:
    """The message of the :class:`ConfigError` raised for TOML ``text``."""
    with pytest.raises(ConfigError) as excinfo:
        parse(text, source=source)
    return str(excinfo.value)


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake home directory, so that ``~`` expands deterministically."""
    path = tmp_path / "home"
    path.mkdir()
    monkeypatch.setenv("HOME", str(path))
    return path


# -- constants -----------------------------------------------------------------


def test_module_constants():
    assert SECTIONS == ("compute", "storage", "jobs")
    assert DEFAULT_TIMEOUT == 180.0
    assert DEFAULT_CLUSTERS == ("lumi", "jupiter", "leonardo", "meluxina")
    assert CONFIG_ENV_VAR == "HPCMON_CONFIG"
    assert ", ".join(CLUSTER_TYPES) == KNOWN_TYPES


# -- default_config_path ---------------------------------------------------------


@pytest.mark.parametrize(
    "environ, expected",
    [
        pytest.param(
            {"HPCMON_CONFIG": "/etc/hpcmon.toml", "XDG_CONFIG_HOME": "/xdg"},
            "/etc/hpcmon.toml",
            id="hpcmon-config-wins-over-xdg",
        ),
        pytest.param({"HPCMON_CONFIG": "relative/conf.toml"}, "relative/conf.toml", id="hpcmon-config-relative"),
        pytest.param({"HPCMON_CONFIG": "~/my/hpcmon.toml"}, "{home}/my/hpcmon.toml", id="hpcmon-config-tilde"),
        pytest.param(
            {"HPCMON_CONFIG": "", "XDG_CONFIG_HOME": "/xdg"},
            "/xdg/hpcmon/config.toml",
            id="empty-hpcmon-config-falls-back-to-xdg",
        ),
        pytest.param({"XDG_CONFIG_HOME": "/xdg"}, "/xdg/hpcmon/config.toml", id="xdg"),
        pytest.param({"XDG_CONFIG_HOME": "~/xdg"}, "{home}/xdg/hpcmon/config.toml", id="xdg-tilde"),
        pytest.param({"XDG_CONFIG_HOME": ""}, "{home}/.config/hpcmon/config.toml", id="empty-xdg"),
        pytest.param(
            {"HPCMON_CONFIG": "", "XDG_CONFIG_HOME": ""},
            "{home}/.config/hpcmon/config.toml",
            id="both-empty",
        ),
        pytest.param({"OTHER": "x"}, "{home}/.config/hpcmon/config.toml", id="neither-set"),
    ],
)
def test_default_config_path_precedence_with_explicit_environ(home, environ, expected):
    assert default_config_path(environ) == Path(expected.format(home=home))


def test_default_config_path_with_empty_mapping_ignores_os_environ(home, monkeypatch):
    monkeypatch.setenv("HPCMON_CONFIG", "/from/os/environ.toml")
    monkeypatch.setenv("XDG_CONFIG_HOME", "/from/os/xdg")
    assert default_config_path({}) == home / ".config" / "hpcmon" / "config.toml"


def test_default_config_path_reads_hpcmon_config_from_os_environ(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCMON_CONFIG", str(tmp_path / "explicit.toml"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert default_config_path() == tmp_path / "explicit.toml"


def test_default_config_path_reads_xdg_config_home_from_os_environ(monkeypatch, tmp_path):
    monkeypatch.delenv("HPCMON_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert default_config_path() == tmp_path / "xdg" / "hpcmon" / "config.toml"


def test_default_config_path_falls_back_to_home_config_from_os_environ(home, monkeypatch):
    monkeypatch.delenv("HPCMON_CONFIG", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert default_config_path() == home / ".config" / "hpcmon" / "config.toml"


# -- default_config --------------------------------------------------------------


def test_default_config_has_the_four_eurohpc_clusters():
    config = default_config()
    assert isinstance(config, Config)
    assert config.parallel is True
    assert config.path is None
    assert list(config.clusters) == ["lumi", "jupiter", "leonardo", "meluxina"]
    for name, cluster in config.clusters.items():
        assert cluster == ClusterConfig(
            name=name,
            type=name,
            ssh=["ssh", name],
            enabled=True,
            timeout=180.0,
            login_shell=True,
            projects=None,
            project_end={},
        )


def test_default_config_returns_independent_objects():
    first = default_config()
    second = default_config()
    assert first == second
    for name in DEFAULT_CLUSTERS:
        assert first.clusters[name] is not second.clusters[name]
        assert first.clusters[name].ssh is not second.clusters[name].ssh
        assert first.clusters[name].project_end is not second.clusters[name].project_end
    first.clusters["lumi"].ssh.append("-v")
    first.clusters["lumi"].timeout = 1.0
    assert default_config().clusters["lumi"].ssh == ["ssh", "lumi"]
    assert default_config().clusters["lumi"].timeout == 180.0


# -- load_config -----------------------------------------------------------------


def test_load_config_missing_file_returns_defaults(tmp_path):
    config = load_config(tmp_path / "missing.toml")
    assert config == default_config()
    assert config.path is None


def test_load_config_missing_required_file_is_an_error(tmp_path):
    path = tmp_path / "missing.toml"
    with pytest.raises(ConfigError) as excinfo:
        load_config(path, required=True)
    assert str(excinfo.value) == f"configuration file not found: {path}"


def test_load_config_reads_file_and_records_its_path(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[settings]\nparallel = false\n\n[clusters.lumi]\nssh = "ssh -J jump lumi"\nprojects = ["project_465000001"]\n',
        encoding="utf-8",
    )
    config = load_config(path, required=True)
    assert config.path == path
    assert config.parallel is False
    assert list(config.clusters) == ["lumi"]
    assert config.clusters["lumi"] == ClusterConfig(
        name="lumi", type="lumi", ssh=["ssh", "-J", "jump", "lumi"], projects=["project_465000001"]
    )


def test_load_config_accepts_a_string_path(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[clusters.jupiter]\n", encoding="utf-8")
    config = load_config(str(path))  # type: ignore[arg-type]
    assert config.path == path
    assert isinstance(config.path, Path)
    assert list(config.clusters) == ["jupiter"]


def test_load_config_expands_the_home_directory(home):
    (home / "hpcmon.toml").write_text("[clusters.leonardo]\n", encoding="utf-8")
    config = load_config(Path("~/hpcmon.toml"))
    assert config.path == home / "hpcmon.toml"
    assert list(config.clusters) == ["leonardo"]


def test_load_config_empty_file_gives_defaults_with_path(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("", encoding="utf-8")
    config = load_config(path)
    assert config.clusters == default_config().clusters
    assert config.parallel is True
    assert config.path == path


def test_load_config_without_path_uses_default_config_path(monkeypatch, tmp_path):
    path = tmp_path / "from-env.toml"
    path.write_text("[clusters.meluxina]\nenabled = false\n", encoding="utf-8")
    monkeypatch.setenv("HPCMON_CONFIG", str(path))
    config = load_config()
    assert config.path == path
    assert list(config.clusters) == ["meluxina"]
    assert config.clusters["meluxina"].enabled is False


def test_load_config_without_path_and_without_file_returns_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCMON_CONFIG", str(tmp_path / "absent.toml"))
    config = load_config()
    assert config == default_config()


def test_load_config_without_path_required_reports_the_default_path(monkeypatch, tmp_path):
    monkeypatch.delenv("HPCMON_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    with pytest.raises(ConfigError) as excinfo:
        load_config(required=True)
    expected = tmp_path / "xdg" / "hpcmon" / "config.toml"
    assert str(excinfo.value) == f"configuration file not found: {expected}"


def test_load_config_invalid_toml_is_an_error(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[clusters.lumi\nssh = 'ssh lumi'\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert str(excinfo.value).startswith(f"{path}: invalid TOML: ")
    assert len(str(excinfo.value)) > len(f"{path}: invalid TOML: ")
    assert isinstance(excinfo.value.__cause__, tomllib.TOMLDecodeError)


def test_load_config_duplicate_key_is_invalid_toml(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[settings]\ntimeout = 1\ntimeout = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert str(excinfo.value).startswith(f"{path}: invalid TOML: ")


def test_load_config_unreadable_path_is_an_error(tmp_path):
    path = tmp_path / "config.toml"
    path.mkdir()  # a directory exists but cannot be read as a file
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert str(excinfo.value).startswith(f"cannot read {path}: ")
    assert isinstance(excinfo.value.__cause__, OSError)


def test_load_config_os_error_while_reading_is_an_error(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text("[settings]\n", encoding="utf-8")

    def denied(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert str(excinfo.value) == f"cannot read {path}: [Errno 13] Permission denied"
    assert isinstance(excinfo.value.__cause__, PermissionError)


def test_load_config_reports_validation_errors_with_the_file_name(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[clusters.lumi]\nhost = 'lumi.csc.fi'\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert str(excinfo.value) == f"{path}: [clusters.lumi]: unknown key(s): host"


def test_load_config_non_utf8_file_is_a_config_error(tmp_path):
    path = tmp_path / "config.toml"
    path.write_bytes(b"[settings]\ntimeout = 60\n# caf\xe9\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert str(excinfo.value).startswith(f"{path}: ")


# -- write_default_config --------------------------------------------------------


def test_write_default_config_creates_parent_directories(tmp_path):
    path = tmp_path / "a" / "b" / "config.toml"
    result = write_default_config(path)
    assert result == path
    assert path.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEMPLATE


def test_write_default_config_accepts_string_and_expands_home(home):
    result = write_default_config("~/conf/hpcmon.toml")  # type: ignore[arg-type]
    assert result == home / "conf" / "hpcmon.toml"
    assert result.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEMPLATE


def test_write_default_config_refuses_to_overwrite_without_force(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("# mine\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        write_default_config(path)
    assert str(excinfo.value) == f"{path} already exists (use --force to overwrite it)"
    assert path.read_text(encoding="utf-8") == "# mine\n"


def test_write_default_config_overwrites_with_force(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("# mine\n", encoding="utf-8")
    assert write_default_config(path, force=True) == path
    assert path.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEMPLATE


def test_write_default_config_with_force_on_a_new_file(tmp_path):
    path = tmp_path / "new" / "config.toml"
    assert write_default_config(path, force=True) == path
    assert path.read_text(encoding="utf-8") == DEFAULT_CONFIG_TEMPLATE


def test_write_default_config_parent_is_a_file(tmp_path):
    parent = tmp_path / "not-a-dir"
    parent.write_text("", encoding="utf-8")
    path = parent / "config.toml"
    with pytest.raises(ConfigError) as excinfo:
        write_default_config(path)
    assert str(excinfo.value).startswith(f"cannot write {path}: ")
    assert isinstance(excinfo.value.__cause__, OSError)


def test_write_default_config_forced_onto_a_directory(tmp_path):
    path = tmp_path / "config.toml"
    path.mkdir()
    with pytest.raises(ConfigError) as excinfo:
        write_default_config(path, force=True)
    assert str(excinfo.value).startswith(f"cannot write {path}: ")
    assert isinstance(excinfo.value.__cause__, IsADirectoryError)
    assert path.is_dir()


def test_written_default_config_loads_as_the_defaults(tmp_path):
    path = write_default_config(tmp_path / "hpcmon" / "config.toml")
    config = load_config(path, required=True)
    assert config.path == path
    assert config.clusters == default_config().clusters
    assert config.parallel is True


# -- DEFAULT_CONFIG_TEMPLATE -------------------------------------------------------


def test_default_template_parses_to_the_default_config():
    data = tomllib.loads(DEFAULT_CONFIG_TEMPLATE)
    assert data["settings"] == {"timeout": 180, "parallel": True, "login_shell": True}
    assert parse_config(data) == default_config()


def test_default_template_commented_examples_are_valid():
    text = DEFAULT_CONFIG_TEMPLATE.replace("# projects = [", "projects = [")
    text = text.replace("# [clusters.meluxina.project_end]", "[clusters.meluxina.project_end]")
    text = text.replace("# p200000 = 2027-06-30", "p200000 = 2027-06-30")
    assert text != DEFAULT_CONFIG_TEMPLATE
    config = parse(text)
    assert config.clusters["lumi"].projects == ["project_465000000"]
    assert config.clusters["meluxina"].project_end == {"p200000": dt.date(2027, 6, 30)}
    assert config.clusters["meluxina"].end_date_for("P200000") == dt.date(2027, 6, 30)


# -- parse_config: top level and [settings] ---------------------------------------


def test_parse_config_empty_mapping_gives_defaults():
    assert parse_config({}) == default_config()


def test_parse_config_unknown_top_level_keys():
    data = {"clusters": {}, "zeta": 1, "alpha": {}, "settings": {}}
    with pytest.raises(ConfigError) as excinfo:
        parse_config(data)
    assert str(excinfo.value) == "<config>: unknown top-level key(s): alpha, zeta"


def test_parse_config_messages_use_the_given_source():
    assert parse_error("cluster = 1\n", source="/etc/hpcmon.toml") == (
        "/etc/hpcmon.toml: unknown top-level key(s): cluster"
    )


def test_settings_must_be_a_table():
    assert parse_error("settings = 5\n") == "<config>: [settings] must be a table"


def test_settings_unknown_keys():
    text = "[settings]\ntimeout = 5\nverbose = true\ncolour = 'no'\n"
    assert parse_error(text) == "<config>: [settings]: unknown key(s): colour, verbose"


@pytest.mark.parametrize(
    "value",
    ["true", "false", "0", "0.0", "-5", "-0.5", '"60"', "[60]", "{ seconds = 60 }"],
    ids=["bool-true", "bool-false", "zero", "zero-float", "negative", "negative-float", "string", "array", "table"],
)
def test_settings_timeout_must_be_a_positive_number(value):
    text = f"[settings]\ntimeout = {value}\n"
    assert parse_error(text) == "<config>: settings.timeout must be a positive number"


@pytest.mark.parametrize("value, expected", [("30", 30.0), ("2.5", 2.5), ("0.001", 0.001)])
def test_settings_timeout_applies_to_every_cluster_as_float(value, expected):
    config = parse(f"[settings]\ntimeout = {value}\n\n[clusters.lumi]\n[clusters.slurm]\n")
    for cluster in config.clusters.values():
        assert cluster.timeout == expected
        assert type(cluster.timeout) is float


@pytest.mark.parametrize("key", ["parallel", "login_shell"])
@pytest.mark.parametrize("value", ["1", "0", '"yes"', '"true"', "[true]"])
def test_settings_booleans_must_be_booleans(key, value):
    assert parse_error(f"[settings]\n{key} = {value}\n") == f"<config>: settings.{key} must be true or false"


@pytest.mark.parametrize("value", [True, False])
def test_settings_parallel(value):
    assert parse_config({"settings": {"parallel": value}}).parallel is value


def test_settings_are_validated_before_clusters():
    text = "[settings]\nparallel = 'no'\n\n[clusters.nonsense]\n"
    assert parse_error(text) == "<config>: settings.parallel must be true or false"


def test_no_clusters_table_gives_defaults_inheriting_settings():
    config = parse("[settings]\ntimeout = 42\nparallel = false\nlogin_shell = false\n")
    assert config.parallel is False
    assert config.path is None
    assert list(config.clusters) == list(DEFAULT_CLUSTERS)
    for name, cluster in config.clusters.items():
        assert cluster == ClusterConfig(
            name=name, type=name, ssh=["ssh", name], timeout=42.0, login_shell=False
        )


def test_no_clusters_table_with_default_settings():
    config = parse("[settings]\n")
    assert config == default_config()


def test_explicit_clusters_inherit_settings():
    config = parse("[settings]\ntimeout = 10\nlogin_shell = false\n\n[clusters.lumi]\n")
    assert config.clusters["lumi"].timeout == 10.0
    assert config.clusters["lumi"].login_shell is False


# -- parse_config: [clusters] -------------------------------------------------------


@pytest.mark.parametrize("text", ["clusters = 'lumi'\n", "clusters = ['lumi', 'jupiter']\n", "clusters = 3\n"])
def test_clusters_must_be_a_table(text):
    assert parse_error(text) == "<config>: [clusters] must be a table"


@pytest.mark.parametrize("value", ["'ssh lumi'", "true", "['ssh', 'lumi']"])
def test_each_cluster_must_be_a_table(value):
    assert parse_error(f"[clusters]\nlumi = {value}\n") == "<config>: [clusters.lumi] must be a table"


def test_empty_clusters_table_gives_no_clusters():
    config = parse("[clusters]\n")
    assert config.clusters == {}
    assert config.parallel is True


def test_clusters_keep_the_file_order():
    config = parse("[clusters.meluxina]\n[clusters.lumi]\n[clusters.leonardo]\n")
    assert list(config.clusters) == ["meluxina", "lumi", "leonardo"]


def test_minimal_cluster_gets_defaults():
    config = parse("[clusters.lumi]\n")
    assert config.clusters["lumi"] == ClusterConfig(
        name="lumi",
        type="lumi",
        ssh=["ssh", "lumi"],
        enabled=True,
        timeout=DEFAULT_TIMEOUT,
        login_shell=True,
        projects=None,
        project_end={},
    )


def test_fully_specified_cluster():
    text = """
[settings]
timeout = 60
login_shell = true

[clusters.lumi-g]
type = "lumi"
ssh = ["ssh", "-i", "~/.ssh/lumi", "user@lumi.csc.fi"]
enabled = false
timeout = 300
login_shell = false
projects = ["project_465000001", "project_465000002"]

[clusters.lumi-g.project_end]
project_465000001 = 2027-03-31
"""
    assert parse(text).clusters == {
        "lumi-g": ClusterConfig(
            name="lumi-g",
            type="lumi",
            ssh=["ssh", "-i", "~/.ssh/lumi", "user@lumi.csc.fi"],
            enabled=False,
            timeout=300.0,
            login_shell=False,
            projects=["project_465000001", "project_465000002"],
            project_end={"project_465000001": dt.date(2027, 3, 31)},
        )
    }


def test_cluster_unknown_keys():
    text = "[clusters.lumi]\nuser = 'me'\nhost = 'lumi'\nssh = 'ssh lumi'\n"
    assert parse_error(text) == "<config>: [clusters.lumi]: unknown key(s): host, user"


@pytest.mark.parametrize("name", ["lumi", "jupiter", "leonardo", "meluxina", "slurm"])
def test_cluster_type_is_inferred_from_the_name(name):
    cluster = parse(f"[clusters.{name}]\n").clusters[name]
    assert cluster.type == name
    assert cluster.ssh == ["ssh", name]


def test_explicit_type_for_a_custom_name():
    cluster = parse("[clusters.mylab]\ntype = 'slurm'\n").clusters["mylab"]
    assert cluster.name == "mylab"
    assert cluster.type == "slurm"
    assert cluster.ssh == ["ssh", "mylab"]


def test_explicit_type_overrides_the_name():
    assert parse("[clusters.lumi]\ntype = 'slurm'\n").clusters["lumi"].type == "slurm"


@pytest.mark.parametrize("name", ["mylab", "LUMI", "lumi-g"])
def test_missing_type_for_a_name_that_is_not_a_type(name):
    assert parse_error(f'[clusters."{name}"]\n') == (
        f"<config>: [clusters.{name}]: missing 'type' (one of: {KNOWN_TYPES})"
    )


@pytest.mark.parametrize(
    "value, shown",
    [("'pbs'", "'pbs'"), ("'LUMI'", "'LUMI'"), ("''", "''"), ("3", "3"), ("true", "True")],
)
def test_unknown_cluster_type(value, shown):
    assert parse_error(f"[clusters.lumi]\ntype = {value}\n") == (
        f"<config>: [clusters.lumi]: unknown type {shown} (one of: {KNOWN_TYPES})"
    )


# -- ssh ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("'ssh lumi'", ["ssh", "lumi"]),
        ("'ssh -J jump.example.org user@lumi.csc.fi'", ["ssh", "-J", "jump.example.org", "user@lumi.csc.fi"]),
        (
            """'ssh -o "ProxyCommand ssh -W %h:%p bastion" lumi'""",
            ["ssh", "-o", "ProxyCommand ssh -W %h:%p bastion", "lumi"],
        ),
        ("'  /usr/bin/ssh   lumi  '", ["/usr/bin/ssh", "lumi"]),
        ("'mosh'", ["mosh"]),
    ],
    ids=["simple", "jump-host", "quoted-argument", "extra-whitespace", "single-word"],
)
def test_ssh_string_is_split_like_a_shell(value, expected):
    assert parse(f"[clusters.lumi]\nssh = {value}\n").clusters["lumi"].ssh == expected


def test_ssh_list_is_used_verbatim_and_copied():
    argv = ["ssh", "-o", "BatchMode=yes", "lumi 2"]
    cluster = parse_config({"clusters": {"lumi": {"ssh": argv}}}).clusters["lumi"]
    assert cluster.ssh == ["ssh", "-o", "BatchMode=yes", "lumi 2"]
    assert cluster.ssh is not argv


def test_ssh_defaults_to_ssh_and_the_cluster_name():
    cluster = parse("[clusters.mylab]\ntype = 'slurm'\n").clusters["mylab"]
    assert cluster.ssh == ["ssh", "mylab"]


@pytest.mark.parametrize(
    "value, detail",
    [("'ssh \"lumi'", "No closing quotation"), ("'ssh lumi\\'", "No escaped character")],
    ids=["unclosed-quote", "trailing-backslash"],
)
def test_ssh_string_with_invalid_quoting(value, detail):
    assert parse_error(f"[clusters.lumi]\nssh = {value}\n") == (
        f"<config>: [clusters.lumi]: invalid 'ssh' command: {detail}"
    )


@pytest.mark.parametrize(
    "value",
    ["''", "'   '", "[]", "['']", "['ssh', '']", "['ssh', 1]", "[['ssh']]", "42", "true", "{ cmd = 'ssh' }"],
    ids=[
        "empty-string",
        "blank-string",
        "empty-list",
        "list-of-empty-string",
        "list-with-empty-string",
        "list-with-integer",
        "nested-list",
        "integer",
        "boolean",
        "table",
    ],
)
def test_invalid_ssh_command(value):
    assert parse_error(f"[clusters.lumi]\nssh = {value}\n") == (
        "<config>: [clusters.lumi]: 'ssh' must be a non-empty string or list of strings"
    )


# -- projects -------------------------------------------------------------------


def test_projects_default_to_all():
    assert parse("[clusters.lumi]\n").clusters["lumi"].projects is None


def test_projects_list_is_kept_and_copied():
    projects = ["project_465000001", "Project_465000002"]
    cluster = parse_config({"clusters": {"lumi": {"projects": projects}}}).clusters["lumi"]
    assert cluster.projects == ["project_465000001", "Project_465000002"]
    assert cluster.projects is not projects


def test_empty_projects_list_is_allowed():
    assert parse("[clusters.lumi]\nprojects = []\n").clusters["lumi"].projects == []


@pytest.mark.parametrize("value", ["'project_465000001'", "[1, 2]", "['p1', 2]", "[['p1']]", "{ p1 = true }", "true"])
def test_projects_must_be_a_list_of_strings(value):
    assert parse_error(f"[clusters.lumi]\nprojects = {value}\n") == (
        "<config>: [clusters.lumi]: 'projects' must be a list of strings"
    )


# -- project_end ----------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "2027-06-30",
        "2027-06-30T12:34:56",
        "2027-06-30T23:30:00+02:00",
        "2027-06-30 00:00:00Z",
        "'2027-06-30'",
        "' 2027-06-30 '",
        "'20270630'",
        "'30.06.2027'",
        "'2027/06/30'",
    ],
    ids=[
        "toml-date",
        "toml-local-datetime",
        "toml-offset-datetime",
        "toml-utc-datetime",
        "iso-string",
        "padded-string",
        "compact-string",
        "european-string",
        "slash-string",
    ],
)
def test_project_end_accepts_dates(value):
    text = f"[clusters.meluxina.project_end]\np200000 = {value}\n"
    end = parse(text).clusters["meluxina"].project_end["p200000"]
    assert end == dt.date(2027, 6, 30)
    assert type(end) is dt.date


def test_project_end_from_python_values():
    data = {
        "clusters": {
            "meluxina": {
                "project_end": {
                    "a": dt.date(2027, 1, 2),
                    "b": dt.datetime(2027, 3, 4, 23, 59, tzinfo=dt.timezone.utc),
                    "c": "2027-05-06",
                }
            }
        }
    }
    assert parse_config(data).clusters["meluxina"].project_end == {
        "a": dt.date(2027, 1, 2),
        "b": dt.date(2027, 3, 4),
        "c": dt.date(2027, 5, 6),
    }


def test_project_end_keeps_names_and_order():
    text = "[clusters.meluxina.project_end]\nP200001 = 2027-01-31\np200000 = 2026-12-31\n"
    project_end = parse(text).clusters["meluxina"].project_end
    assert list(project_end.items()) == [("P200001", dt.date(2027, 1, 31)), ("p200000", dt.date(2026, 12, 31))]


def test_project_end_defaults_to_empty():
    assert parse("[clusters.meluxina]\n").clusters["meluxina"].project_end == {}


@pytest.mark.parametrize(
    "value",
    [
        "'next year'",
        "'2027-02-30'",
        "'06/30/2027'",
        "''",
        "'n/a'",
        "'-'",
        "'unlimited'",
        "20270630",
        "2027.5",
        "true",
        "12:00:00",
        "[2027, 6, 30]",
        "{ year = 2027 }",
    ],
    ids=[
        "garbage-string",
        "impossible-date",
        "us-format",
        "empty-string",
        "missing-na",
        "missing-dash",
        "missing-unlimited",
        "integer",
        "float",
        "boolean",
        "toml-time",
        "array",
        "table",
    ],
)
def test_project_end_rejects_non_dates(value):
    text = f"[clusters.meluxina.project_end]\np200000 = {value}\n"
    assert parse_error(text) == "<config>: [clusters.meluxina]: project_end.p200000 must be a date (YYYY-MM-DD)"


@pytest.mark.parametrize("value", ["'2027-06-30'", "2027-06-30", "['p1']"])
def test_project_end_must_be_a_table(value):
    text = f"[clusters.meluxina]\nproject_end = {value}\n"
    assert parse_error(text) == "<config>: [clusters.meluxina] project_end must be a table"


# -- per-cluster overrides ---------------------------------------------------------


def test_cluster_overrides_settings():
    text = """
[settings]
timeout = 10
login_shell = false

[clusters.lumi]
timeout = 99.5
login_shell = true
enabled = false

[clusters.jupiter]
"""
    clusters = parse(text).clusters
    assert (clusters["lumi"].timeout, clusters["lumi"].login_shell, clusters["lumi"].enabled) == (99.5, True, False)
    assert (clusters["jupiter"].timeout, clusters["jupiter"].login_shell, clusters["jupiter"].enabled) == (
        10.0,
        False,
        True,
    )


def test_cluster_timeout_integer_is_converted_to_float():
    timeout = parse("[clusters.lumi]\ntimeout = 30\n").clusters["lumi"].timeout
    assert timeout == 30.0
    assert type(timeout) is float


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("enabled", "'no'", "enabled must be true or false"),
        ("enabled", "0", "enabled must be true or false"),
        ("login_shell", "1", "login_shell must be true or false"),
        ("login_shell", "'false'", "login_shell must be true or false"),
        ("timeout", "0", "timeout must be a positive number"),
        ("timeout", "-1.5", "timeout must be a positive number"),
        ("timeout", "true", "timeout must be a positive number"),
        ("timeout", "'30'", "timeout must be a positive number"),
    ],
)
def test_cluster_value_type_errors(key, value, message):
    assert parse_error(f"[clusters.lumi]\n{key} = {value}\n") == f"<config>: [clusters.lumi]: {message}"


def test_first_invalid_cluster_is_reported():
    text = "[clusters.lumi]\n[clusters.bad]\ntype = 'pbs'\n[clusters.worse]\n"
    assert parse_error(text) == f"<config>: [clusters.bad]: unknown type 'pbs' (one of: {KNOWN_TYPES})"


# -- non-finite and huge timeouts (suspected bugs) -----------------------------------


@pytest.mark.parametrize("where", ["settings", "clusters.lumi"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param("0", id="zero-control"),
        pytest.param(
            "nan",
            id="nan",
        ),
        pytest.param(
            "inf",
            id="inf",
        ),
        pytest.param(
            "1" + "0" * 400,
            id="huge-integer",
        ),
    ],
)
def test_unusable_timeouts_are_rejected(where, value):
    with pytest.raises(ConfigError) as excinfo:
        parse(f"[{where}]\ntimeout = {value}\n")
    prefix = "<config>: settings.timeout" if where == "settings" else "<config>: [clusters.lumi]: timeout"
    assert str(excinfo.value) == f"{prefix} must be a positive number"


# -- ClusterConfig ----------------------------------------------------------------


def test_cluster_config_defaults():
    cluster = ClusterConfig(name="x", type="slurm", ssh=["ssh", "x"])
    assert cluster.enabled is True
    assert cluster.timeout == DEFAULT_TIMEOUT
    assert cluster.login_shell is True
    assert cluster.projects is None
    assert cluster.project_end == {}
    assert cluster.project_end is not ClusterConfig(name="y", type="slurm", ssh=["ssh"]).project_end


@pytest.mark.parametrize("project", ["project_465000001", "anything", "", None])
def test_wants_project_without_filter_wants_everything(project):
    assert make_config("lumi").wants_project(project) is True


@pytest.mark.parametrize(
    "project, expected",
    [
        ("project_465000001", True),
        ("PROJECT_465000001", True),
        ("p200000", True),
        ("P200000", True),
        ("project_465000003", False),
        ("project_46500000", False),
        ("", False),
        (None, True),
    ],
)
def test_wants_project_with_filter_is_case_insensitive(project, expected):
    cluster = make_config("lumi", projects=["Project_465000001", "p200000"])
    assert cluster.wants_project(project) is expected


def test_wants_project_with_empty_filter_hides_named_projects():
    cluster = make_config("lumi", projects=[])
    assert cluster.wants_project("project_465000001") is False
    assert cluster.wants_project(None) is True


@pytest.mark.parametrize(
    "project, expected",
    [
        ("p200000", dt.date(2027, 6, 30)),
        ("P200000", dt.date(2027, 6, 30)),
        ("other", dt.date(2026, 12, 31)),
        ("OTHER", dt.date(2026, 12, 31)),
        ("p200001", None),
        ("", None),
    ],
)
def test_end_date_for_is_case_insensitive(project, expected):
    cluster = make_config(
        "meluxina", project_end={"P200000": dt.date(2027, 6, 30), "Other": dt.date(2026, 12, 31)}
    )
    assert cluster.end_date_for(project) == expected


def test_end_date_for_without_configured_dates():
    assert make_config("meluxina").end_date_for("p200000") is None
