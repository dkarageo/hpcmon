"""Tests of the GPU information: the GPUs of the nodes of every cluster and the
estimates, in GPU-hours, of the compute budgets.

The rates of the supported systems were checked against their accounting: the
core-hours charged by JSC (JUPITER) and CINECA (Leonardo) and the node-hours
charged by LuxProvide (MeluXina) for real jobs match the node-hours of these
jobs times 288, 32 and 1 respectively, i.e. 72, 8 and 1/4 per GPU-hour.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
from typing import Any, List, Optional

import pytest
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from hpcmon import cli
from hpcmon.clusters import CLUSTER_TYPES, create_cluster
from hpcmon.config import DEFAULT_CONFIG_TEMPLATE, ConfigError, parse_config
from hpcmon.models import ClusterReport, ComputeBudget
from hpcmon.render import (
    GPU_HOURS_ASSUMPTION,
    GPU_STYLE,
    compute_table,
    format_gpu_hours,
    gpu_summary_table,
    render_gpu_summary,
    render_report,
    render_reports,
    report_title,
)
from hpcmon.runner import RemoteError
from tests.conftest import NOW, TODAY, FakeRunner, make_config, ok
from tests.test_clusters_base import DummyCluster, dummy_runner

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


#: The GPUs of the nodes of the supported systems.
LABELS = {
    "lumi": "4× AMD MI250X 128 GB (2 GCDs each)",
    "jupiter": "4× NVIDIA GH200 96 GB",
    "leonardo": "4× NVIDIA A100 64 GB",
    "meluxina": "4× NVIDIA A100 40 GB",
    "slurm": None,
}


class GpuDummyCluster(DummyCluster):
    """The dummy cluster of the base tests, with GPU nodes."""

    gpus = "2× Dummy GPU 16 GB"
    gpu_rates = {("GPU", "core-h"): 0.5}


def gpu_dummy(runner: Optional[FakeRunner] = None, **config: Any) -> GpuDummyCluster:
    return GpuDummyCluster(make_config("mycluster", type="dummy", **config), runner=runner or dummy_runner())


def budget(**kwargs: Any) -> ComputeBudget:
    values = dict(project="EUHPC_D00_001", resource="Booster", unit="core-h")
    values.update(kwargs)
    return ComputeBudget(**values)


def report(**kwargs: Any) -> ClusterReport:
    kwargs.setdefault("name", "leonardo")
    kwargs.setdefault("system", "Leonardo")
    kwargs.setdefault("sections", ["compute"])
    return ClusterReport(**kwargs)


def rows(table: Table) -> List[tuple]:
    columns = [list(column.cells) for column in table.columns]
    return [tuple(cell.plain for cell in row) for row in zip(*columns)]


def render_to_text(renderable_or_reports: Any, width: int = 250) -> str:
    console = Console(file=io.StringIO(), width=width, color_system=None, legacy_windows=False)
    console.print(renderable_or_reports)
    return console.file.getvalue()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def _neutral_terminal_environment(monkeypatch):
    for name in ("NO_COLOR", "FORCE_COLOR", "TERM", "COLORTERM", "COLUMNS", "LINES"):
        monkeypatch.delenv(name, raising=False)


# -- models -----------------------------------------------------------------------


def test_budget_without_a_rate_has_no_gpu_hours():
    b = budget(total=96_000, used=3_187, month_total=8_250, month_used=0)

    assert b.gpu_hours_per_unit is None
    assert (b.gpu_hours_total, b.gpu_hours_remaining) == (None, None)
    assert (b.month_gpu_hours_total, b.month_gpu_hours_remaining) == (None, None)
    assert b.gpu_hours(100.0) is None


def test_budget_gpu_hours_convert_every_amount():
    b = budget(total=96_000, used=3_187, month_total=8_250, month_used=0, gpu_hours_per_unit=1 / 8)

    assert b.gpu_hours_total == 12_000
    assert b.gpu_hours_remaining == 92_813 / 8
    assert b.month_gpu_hours_total == b.month_gpu_hours_remaining == 8_250 / 8
    assert b.gpu_hours(None) is None
    assert b.gpu_hours(0) == 0


def test_budget_gpu_hours_follow_the_available_amount_reported_by_the_cluster():
    b = budget(total=1_000, used=100, available=800, gpu_hours_per_unit=4.0)

    assert b.remaining == 800
    assert b.gpu_hours_remaining == 3_200


def test_budget_gpu_hours_of_unknown_amounts_are_unknown():
    b = budget(month_total=210, month_used=27, gpu_hours_per_unit=4.0)

    assert (b.gpu_hours_total, b.gpu_hours_remaining) == (None, None)
    assert (b.month_gpu_hours_total, b.month_gpu_hours_remaining) == (840, 732)


def test_budget_gpu_hours_of_an_overdrawn_budget_are_negative():
    assert budget(total=10, used=12, gpu_hours_per_unit=2.0).gpu_hours_remaining == -4


def test_budget_gpu_hours_are_in_the_json_representation():
    data = budget(total=100, used=20, gpu_hours_per_unit=0.25).to_dict(TODAY)

    assert {key: data[key] for key in data if "gpu" in key} == {
        "gpu_hours_per_unit": 0.25,
        "gpu_hours_total": 25.0,
        "gpu_hours_remaining": 20.0,
        "month_gpu_hours_total": None,
        "month_gpu_hours_remaining": None,
    }
    assert json.loads(json.dumps(data)) == data


def test_report_gpus_default_to_unknown():
    assert ClusterReport(name="x", system="X").gpus is None
    assert ClusterReport(name="x", system="X", gpus="1× GPU").to_dict()["gpus"] == "1× GPU"


# -- configuration ------------------------------------------------------------------


def parse(text: str):
    return parse_config(tomllib.loads(text))


def test_config_gpus_and_rates_are_optional():
    config = parse('[clusters.leonardo]\n')

    assert config.clusters["leonardo"].gpus is None
    assert config.clusters["leonardo"].gpu_hours_per_unit == {}


def test_config_gpus_and_rates():
    config = parse(
        '[clusters.leonardo]\n'
        'gpus = "4x A100"\n'
        '[clusters.leonardo.gpu_hours_per_unit]\n'
        'Booster = 0.125\n'
        'DCGP = 0\n'
        'GPU = 2\n'
    )
    cluster = config.clusters["leonardo"]

    assert cluster.gpus == "4x A100"
    assert cluster.gpu_hours_per_unit == {"Booster": 0.125, "DCGP": 0.0, "GPU": 2.0}
    assert all(isinstance(rate, float) for rate in cluster.gpu_hours_per_unit.values())


def test_config_gpus_can_be_empty_to_hide_them():
    assert parse('[clusters.lumi]\ngpus = ""\n').clusters["lumi"].gpus == ""


@pytest.mark.parametrize("value", [4, ["4x A100"], True, {"model": "A100"}])
def test_config_gpus_must_be_a_string(value):
    with pytest.raises(ConfigError, match=r"\[clusters.lumi\]: 'gpus' must be a string"):
        parse_config({"clusters": {"lumi": {"gpus": value}}}, source="cfg.toml")


@pytest.mark.parametrize("value", [-1, -0.5, True, False, "1", None, [1], float("inf"), float("nan"), 10**400])
def test_config_gpu_rates_must_be_non_negative_numbers(value):
    data = {"clusters": {"lumi": {"gpu_hours_per_unit": {"GPU": value}}}}
    with pytest.raises(ConfigError, match=r"gpu_hours_per_unit.GPU must be a number \(0 or more\)"):
        parse_config(data, source="cfg.toml")


@pytest.mark.parametrize("value", [1, "GPU=1", [1]])
def test_config_gpu_rates_must_be_a_table(value):
    with pytest.raises(ConfigError, match=r"gpu_hours_per_unit must be a table"):
        parse_config({"clusters": {"lumi": {"gpu_hours_per_unit": value}}})


def test_config_gpu_rate_lookup_is_case_insensitive():
    cluster = make_config("leonardo", gpu_hours_per_unit={"Booster": 0.125, "dcgp": 0.0})

    assert cluster.gpu_rate_for("Booster") == 0.125
    assert cluster.gpu_rate_for("BOOSTER") == 0.125
    assert cluster.gpu_rate_for("DCGP") == 0.0
    assert cluster.gpu_rate_for("GPU") is None
    assert make_config("x").gpu_rate_for("GPU") is None


def test_default_template_documents_the_gpu_settings():
    text = DEFAULT_CONFIG_TEMPLATE
    for commented in ("# gpus = ", "# [clusters.lumi.gpu_hours_per_unit]", "# GPU = 1.0"):
        assert commented in text
        text = text.replace(commented, commented[2:])

    lumi = parse(text).clusters["lumi"]

    assert lumi.gpus == LABELS["lumi"]
    assert lumi.gpu_hours_per_unit == {"GPU": 1.0}


# -- built-in GPUs and rates of the supported systems ---------------------------------


@pytest.mark.parametrize("cluster_type", sorted(CLUSTER_TYPES))
def test_built_in_gpus_of_every_system(cluster_type):
    cluster = create_cluster(make_config(cluster_type), runner=FakeRunner())

    assert cluster.gpu_label == LABELS[cluster_type]
    assert CLUSTER_TYPES[cluster_type].gpus == LABELS[cluster_type]


@pytest.mark.parametrize(
    "cluster_type, resource, unit, rate",
    [
        # LUMI bills GPU-hours of MI250X modules (2 GCDs); CPU core-hours and
        # storage TB-hours cannot be spent on the GPU partitions.
        ("lumi", "GPU", "GPU-h", 1.0),
        ("lumi", "CPU", "core-h", None),
        ("lumi", "Storage", "TB-h", None),
        ("lumi", "GPU", "core-h", None),  # an unexpected unit is not converted
        # JUPITER: 288 Grace cores per node of 4 GH200, accounted per core-hour.
        ("jupiter", "jupiter_booster", "core-h", 1 / 72),
        ("jupiter", "JUPITER_BOOSTER", "core-h", 1 / 72),  # as spelled by `jutil project cpuquota`
        ("jupiter", "jupiter_cluster", "core-h", None),
        ("jupiter", "jupiter_booster", "node-h", None),
        # Leonardo: 32 cores per node of 4 A100, i.e. 8 core-hours per GPU-hour;
        # the DCGP partition has no GPU.
        ("leonardo", "Booster", "core-h", 1 / 8),
        ("leonardo", "DCGP", "core-h", None),
        ("leonardo", "booster", "core-h", 1 / 8),  # resources are compared without case
        ("leonardo", "Booster", "local-h", None),  # but units exactly
        # MeluXina: whole nodes of 4 A100.
        ("meluxina", "GPU", "node-h", 4.0),
        ("meluxina", "CPU", "node-h", None),
        ("meluxina", "FPGA", "node-h", None),
        ("meluxina", "LargeMem", "node-h", None),
        ("slurm", "GPU", "GPU-h", None),
    ],
)
def test_built_in_gpu_rates(cluster_type, resource, unit, rate):
    cluster = create_cluster(make_config(cluster_type), runner=FakeRunner())

    assert cluster.gpu_rate(budget(resource=resource, unit=unit)) == rate


def test_rates_of_the_supported_systems_buy_four_gpu_hours_per_node_hour():
    # A whole node for one hour, in the unit charged by each system.
    node_hour = {
        "lumi": ("GPU", "GPU-h", 4),  # 8 GCDs = 4 MI250X: 4 GPU-hours per node-hour
        "jupiter": ("jupiter_booster", "core-h", 288),
        "leonardo": ("Booster", "core-h", 32),
        "meluxina": ("GPU", "node-h", 1),
    }
    for cluster_type, (resource, unit, amount) in node_hour.items():
        cluster = create_cluster(make_config(cluster_type), runner=FakeRunner())
        b = budget(resource=resource, unit=unit, gpu_hours_per_unit=cluster.gpu_rate(budget(resource=resource,
                                                                                              unit=unit)))
        assert b.gpu_hours(amount) == pytest.approx(4), cluster_type


# -- Cluster: settings and collection ---------------------------------------------------


def test_gpu_label_setting_replaces_the_built_in_one():
    cluster = create_cluster(make_config("lumi", gpus="8× MI250X GCD"), runner=FakeRunner())
    assert cluster.gpu_label == "8× MI250X GCD"


def test_empty_gpu_label_setting_hides_the_built_in_one():
    cluster = create_cluster(make_config("lumi", gpus=""), runner=FakeRunner())
    assert cluster.gpu_label is None


def test_gpu_label_setting_describes_generic_clusters():
    cluster = create_cluster(make_config("mine", type="slurm", gpus="8× NVIDIA H100"), runner=FakeRunner())
    assert cluster.gpu_label == "8× NVIDIA H100"


def test_configured_rate_takes_precedence_over_the_built_in_one():
    cluster = create_cluster(make_config("leonardo", gpu_hours_per_unit={"booster": 0.25}), runner=FakeRunner())

    assert cluster.gpu_rate(budget(resource="Booster")) == 0.25
    # Whatever the unit, and for resources without a built-in rate.
    assert cluster.gpu_rate(budget(resource="Booster", unit="node-h")) == 0.25
    assert cluster.gpu_rate(budget(resource="DCGP")) is None


def test_configured_rate_of_zero_disables_the_estimate():
    cluster = create_cluster(make_config("leonardo", gpu_hours_per_unit={"Booster": 0}), runner=FakeRunner())
    assert cluster.gpu_rate(budget(resource="Booster")) is None


def test_configured_rate_converts_resources_unknown_to_hpcmon():
    cluster = create_cluster(make_config("leonardo", gpu_hours_per_unit={"DCGP": 0.01}), runner=FakeRunner())

    assert cluster.gpu_rate(budget(resource="DCGP")) == 0.01
    assert cluster.gpu_rate(budget(resource="Booster")) == 1 / 8


def test_collect_reports_the_gpus_and_the_rate_of_every_budget():
    cluster = gpu_dummy()

    report = cluster.collect(now=NOW)

    assert report.gpus == "2× Dummy GPU 16 GB"
    assert [(b.project, b.resource, b.gpu_hours_per_unit) for b in report.budgets] == [
        ("proj_a", "GPU", 0.5),
        ("proj_b", "CPU", None),
        ("proj_c", "GPU", 0.5),
    ]
    assert report.budgets[0].gpu_hours_remaining == 440
    assert report.to_dict()["budgets"][0]["gpu_hours_remaining"] == 440


def test_collect_applies_the_configured_rates_and_gpus():
    cluster = gpu_dummy(gpus="", gpu_hours_per_unit={"cpu": 0.01, "GPU": 0})

    report = cluster.collect(["compute"], now=NOW)

    assert report.gpus is None
    assert [b.gpu_hours_per_unit for b in report.budgets] == [None, 0.01, None]


def test_collect_keeps_a_rate_set_by_the_parser():
    class ParserRateCluster(GpuDummyCluster):
        def parse_compute(self, results, today):
            budgets = super().parse_compute(results, today)
            budgets[1].gpu_hours_per_unit = 0.02
            return budgets

    cluster = ParserRateCluster(make_config("mycluster", type="dummy"), runner=dummy_runner())

    report = cluster.collect(["compute"], now=NOW)

    assert [b.gpu_hours_per_unit for b in report.budgets] == [0.5, 0.02, 0.5]

    # The configuration takes precedence over the parser too.
    for configured, expected in ((0.25, 0.25), (0, None)):
        cluster = ParserRateCluster(
            make_config("mycluster", type="dummy", gpu_hours_per_unit={"CPU": configured}), runner=dummy_runner()
        )
        assert [b.gpu_hours_per_unit for b in cluster.collect(["compute"], now=NOW).budgets] == [0.5, expected, 0.5]


def test_collect_reports_the_gpus_even_when_the_cluster_cannot_be_reached():
    cluster = gpu_dummy(FakeRunner(error=RemoteError("SSH connection failed: no route")))

    report = cluster.collect(now=NOW)

    assert report.errors == ["SSH connection failed: no route"]
    assert report.gpus == "2× Dummy GPU 16 GB"


def test_cli_report_of_a_crashing_cluster_keeps_its_gpus():
    class Crashing(GpuDummyCluster):
        def collect(self, sections=(), today=None, now=None):
            raise RuntimeError("bug")

    report = cli._collect(Crashing(make_config("mycluster", type="dummy"), runner=FakeRunner()), ["compute"], NOW)

    assert report.errors == ["internal error (RuntimeError: bug)"]
    assert report.gpus == "2× Dummy GPU 16 GB"


# -- rendering: titles ------------------------------------------------------------------


def test_report_title_names_the_gpus_of_the_nodes():
    title = report_title(report(gpus="4× NVIDIA A100 64 GB"))

    assert title.plain == "Leonardo · GPU nodes: 4× NVIDIA A100 64 GB"
    assert title.style == "bold cyan"
    assert [(title.plain[span.start:span.end], span.style) for span in title.spans] == [
        (" · GPU nodes: 4× NVIDIA A100 64 GB", GPU_STYLE),
    ]


@pytest.mark.parametrize("gpus", [None, ""])
def test_report_title_without_gpus(gpus):
    title = report_title(report(name="leo", gpus=gpus))

    assert title.plain == "Leonardo (leo)"
    assert title.spans == []


def test_report_title_cleans_and_escapes_the_gpus():
    title = report_title(report(gpus="\x1b[31m4× [bold]X[/bold]\x07"))
    assert title.plain == "Leonardo · GPU nodes: 4× [bold]X[/bold]"


def test_render_report_shows_the_gpus_in_the_rule():
    console = Console(file=io.StringIO(), width=120, color_system=None)

    render_report(report(gpus="4× NVIDIA A100 64 GB", sections=[]), console, TODAY)

    first = console.file.getvalue().splitlines()[0]  # type: ignore[attr-defined]
    assert first.startswith("Leonardo · GPU nodes: 4× NVIDIA A100 64 GB ─")


# -- rendering: estimates in the compute table ----------------------------------------------


def test_compute_table_shows_gpu_hours_below_the_remaining_amounts():
    budgets = [
        budget(total=96_000, used=3_187, month_total=8_250, month_used=0, end=dt.date(2027, 6, 30),
               gpu_hours_per_unit=1 / 8),
        budget(project="EUHPC_D00_002", resource="DCGP", total=1_000, used=0, month_total=100, month_used=0),
    ]

    table = compute_table(budgets, TODAY)

    assert [row[3:6] for row in rows(table)] == [
        ("92,813 (97%)\n~11,602 GPU-h", "8,250", "8,250 (100%)\n~1,031 GPU-h"),
        ("1,000 (100%)", "100", "100 (100%)"),
    ]
    remaining = table.columns[3].cells
    first = list(remaining)[0]
    assert first.style == "green"
    assert [(first.plain[s.start:s.end], s.style) for s in first.spans] == [("\n~11,602 GPU-h", GPU_STYLE)]


def test_compute_table_shows_gpu_hours_of_monthly_only_budgets():
    b = budget(project="p200001", resource="GPU", unit="node-h", month_total=210, month_used=27,
               gpu_hours_per_unit=4.0)

    assert rows(compute_table([b], TODAY)) == [
        ("p200001", "GPU [node-h]", "-", "-", "210", "183 (87%)\n~732 GPU-h", "-"),
    ]


def test_compute_table_does_not_repeat_budgets_already_in_gpu_hours():
    same = budget(project="project_465000123", resource="GPU", unit="GPU-h", total=100, used=0,
                  gpu_hours_per_unit=1.0)
    gcd = dataclasses.replace(same, gpu_hours_per_unit=0.5)

    assert [row[3] for row in rows(compute_table([same, gcd], TODAY))] == [
        "100 (100%)",
        "100 (100%)\n~50 GPU-h",
    ]


def test_compute_table_dims_the_gpu_hours_of_ended_budgets():
    b = budget(total=800, used=0, end=dt.date(2026, 11, 1), gpu_hours_per_unit=1 / 8)

    cell = list(compute_table([b], TODAY).columns[3].cells)[0]

    assert cell.plain == "800 (100%)\n~100 GPU-h"
    assert cell.style == "dim"
    assert [span.style for span in cell.spans] == ["dim"]


def test_compute_table_without_a_remaining_amount_shows_no_gpu_hours():
    b = budget(total=None, used=5, gpu_hours_per_unit=1 / 8)
    assert rows(compute_table([b], TODAY))[0][3] == "-"


@pytest.mark.parametrize(
    "value, expected",
    [(11_601.625, "~11,602 GPU-h"), (0.5, "~0.5 GPU-h"), (0, "~0 GPU-h"), (-4, "~-4 GPU-h"), (1e7, "~10,000,000 GPU-h")],
)
def test_format_gpu_hours(value, expected):
    assert format_gpu_hours(value) == expected


# -- rendering: summary of all the clusters ----------------------------------------------------


def lumi_report(**kwargs: Any) -> ClusterReport:
    values = dict(
        name="lumi",
        system="LUMI",
        gpus=LABELS["lumi"],
        budgets=[
            budget(project="project_465000123", resource="CPU", total=50_000, used=0, end=dt.date(2027, 4, 4)),
            budget(project="project_465000123", resource="GPU", unit="GPU-h", total=12_000, used=0,
                   end=dt.date(2027, 4, 4), gpu_hours_per_unit=1.0),
        ],
    )
    values.update(kwargs)
    return report(**values)


def leonardo_report(**kwargs: Any) -> ClusterReport:
    values = dict(
        gpus=LABELS["leonardo"],
        budgets=[
            budget(total=96_000, used=3_187, month_total=8_250, month_used=0, end=dt.date(2027, 6, 30),
                   gpu_hours_per_unit=1 / 8),
            budget(project="EUHPC_D00_001", resource="DCGP", total=1_000, used=0, month_total=10, month_used=0),
        ],
    )
    values.update(kwargs)
    return report(**values)


def jupiter_report(**kwargs: Any) -> ClusterReport:
    values = dict(
        name="jupiter",
        system="JUPITER",
        gpus=LABELS["jupiter"],
        budgets=[budget(project="e-dev-2026d00-001", resource="jupiter_booster", total=864_000, used=62_738.88,
                        end=dt.date(2027, 5, 31), account="e-dev-acct", gpu_hours_per_unit=1 / 72)],
    )
    values.update(kwargs)
    return report(**values)


def meluxina_report(**kwargs: Any) -> ClusterReport:
    values = dict(
        name="meluxina",
        system="MeluXina",
        gpus=LABELS["meluxina"],
        budgets=[budget(project="p200001", resource="GPU", unit="node-h", month_total=210, month_used=27,
                        gpu_hours_per_unit=4.0)],
    )
    values.update(kwargs)
    return report(**values)


def test_gpu_summary_lists_every_budget_that_can_buy_gpu_hours():
    reports = [lumi_report(), jupiter_report(), leonardo_report(), meluxina_report()]

    table = gpu_summary_table(reports, TODAY)

    assert table is not None
    assert table.title == "GPU-hours left (estimated)"
    assert [str(column.header) for column in table.columns] == [
        "Cluster", "Project", "Remaining", "Month\nremaining", "Ends (days left)",
    ]
    assert [column.justify for column in table.columns] == ["left", "left", "right", "right", "right"]
    assert rows(table) == [
        ("LUMI", "project_465000123", "12,000 (100%)", "-", "2027-04-04 (137d)"),
        ("JUPITER", "e-dev-2026d00-001 (e-dev-acct)", "11,129 (93%)", "-", "2027-05-31 (194d)"),
        ("Leonardo", "EUHPC_D00_001", "11,602 (97%)", "1,031 (100%)", "2027-06-30 (224d)"),
        ("MeluXina", "p200001", "-", "732 (87%)", "-"),
    ]
    assert isinstance(table.caption, Text)
    assert table.caption.plain == f"Note: {GPU_HOURS_ASSUMPTION}"
    assert table.caption.style == "dim"


def test_gpu_summary_has_no_monthly_column_without_monthly_budgets():
    table = gpu_summary_table([lumi_report(), jupiter_report()], TODAY)

    assert table is not None
    assert [str(column.header) for column in table.columns] == ["Cluster", "Project", "Remaining", "Ends (days left)"]


def test_gpu_summary_dims_ended_budgets():
    ended = jupiter_report(budgets=[budget(total=720, used=0, end=dt.date(2026, 1, 1), gpu_hours_per_unit=1 / 72)])

    table = gpu_summary_table([ended], TODAY)

    assert table is not None
    assert rows(table) == [("JUPITER", "EUHPC_D00_001", "10 (100%)", "2026-01-01 (ended)")]
    assert str(list(table.columns[2].cells)[0].style) == "dim"


@pytest.mark.parametrize(
    "reports",
    [
        [],
        [leonardo_report(budgets=[budget(total=1, used=0)])],  # no rate
        [leonardo_report(sections=["storage", "jobs"])],  # compute not collected
        [leonardo_report(failed_sections=["compute"])],
    ],
    ids=["nothing", "no-rate", "no-compute", "failed-compute"],
)
def test_gpu_summary_without_gpu_budgets_is_none(reports):
    assert gpu_summary_table(reports, TODAY) is None


def test_gpu_summary_ignores_clusters_whose_compute_section_failed():
    table = gpu_summary_table([lumi_report(failed_sections=["compute"]), leonardo_report()], TODAY)

    assert table is not None
    assert [row[0] for row in rows(table)] == ["Leonardo"]


def test_gpu_summary_fits_80_columns_without_folding_project_names():
    reports = [lumi_report(), jupiter_report(budgets=[dataclasses.replace(jupiter_report().budgets[0], account=None)]),
               leonardo_report(), meluxina_report()]

    # On a wide terminal, the table (caption apart) takes no more than 80 columns...
    wide = render_to_text(gpu_summary_table(reports, TODAY), width=200).splitlines()
    assert max(len(line.rstrip()) for line in wide if not line.startswith("Note:")) <= 80
    # ... so nothing is folded on an 80-column one.
    text = render_to_text(gpu_summary_table(reports, TODAY), width=80)
    for project in ("project_465000123", "e-dev-2026d00-001", "EUHPC_D00_001", "p200001"):
        assert project in text
    assert "2027-06-30 (224d)" in text


class Recorder:
    def __init__(self) -> None:
        self.printed: List[Any] = []

    def print(self, *objects: Any) -> None:
        self.printed.append(objects[0] if objects else None)


@pytest.mark.parametrize(
    "reports",
    [
        [leonardo_report()],
        [leonardo_report(), lumi_report(budgets=[])],
        [leonardo_report(), meluxina_report(failed_sections=["compute"])],
    ],
    ids=["one-cluster", "other-without-budgets", "other-failed"],
)
def test_gpu_summary_is_only_printed_for_several_clusters(reports):
    console = Recorder()

    render_gpu_summary(reports, console, TODAY)  # type: ignore[arg-type]

    assert console.printed == []


def test_gpu_summary_of_several_clusters_is_printed_under_its_own_rule():
    console = Recorder()

    render_gpu_summary([leonardo_report(), meluxina_report()], console, TODAY)  # type: ignore[arg-type]

    rule, table, blank = console.printed
    assert isinstance(rule, Rule)
    assert isinstance(rule.title, Text) and rule.title.plain == "All clusters"
    assert isinstance(table, Table) and table.title == "GPU-hours left (estimated)"
    assert blank is None


def test_render_reports_prints_the_gpu_summary_after_all_the_reports():
    console = Console(file=io.StringIO(), width=120, color_system=None)
    reports = [lumi_report(), leonardo_report(), meluxina_report()]

    rendered = render_reports(iter(reports), console, TODAY)

    text = console.file.getvalue()  # type: ignore[attr-defined]
    assert rendered == reports
    assert text.index("All clusters ─") > text.index("MeluXina · GPU nodes: 4× NVIDIA A100 40 GB ─")
    assert text.rstrip().endswith("that come with their GPUs).")


# -- review fixes -----------------------------------------------------------------------


def test_gpu_hours_too_large_to_represent_are_unknown():
    b = budget(total=1e300, used=0, gpu_hours_per_unit=1e10)

    assert b.gpu_hours_total is None
    assert b.gpu_hours_remaining is None
    assert "Infinity" not in json.dumps(b.to_dict())


def test_huge_configured_rate_does_not_produce_invalid_json():
    report = ClusterReport(name="x", system="X", budgets=[budget(total=1e200, used=0, gpu_hours_per_unit=1e200)])

    text = json.dumps(report.to_dict())

    assert "Infinity" not in text
    assert json.loads(text)["budgets"][0]["gpu_hours_total"] is None


@pytest.mark.parametrize("label", ["   ", "\t", " \n "])
def test_blank_gpu_label_setting_hides_the_label(label):
    cluster = create_cluster(make_config("lumi", gpus=label), runner=FakeRunner())
    assert cluster.gpu_label is None


@pytest.mark.parametrize("gpus", ["   ", "\x1b[0m", "\x07 "])
def test_report_title_ignores_labels_without_printable_text(gpus):
    assert report_title(report(gpus=gpus)).plain == "Leonardo"


def test_unused_configured_rates_are_reported():
    cluster = gpu_dummy(gpu_hours_per_unit={"GPU": 1.0, "Boster": 0.25, "cpu": 0.5, "FPGA": 2})

    report = cluster.collect(["compute"], now=NOW)

    assert report.warnings == [
        "gpu_hours_per_unit: no budget of resource Boster, FPGA on this cluster (budget resources: CPU, GPU)"
    ]


def test_unused_configured_rates_consider_the_budgets_hidden_by_the_projects_setting():
    cluster = gpu_dummy(gpu_hours_per_unit={"CPU": 0.5}, projects=["proj_a"])

    report = cluster.collect(["compute"], now=NOW)

    assert [b.project for b in report.budgets] == ["proj_a"]
    assert report.warnings == []


def test_configured_rates_on_a_cluster_without_budgets_are_reported():
    cluster = gpu_dummy(FakeRunner({"alloc": ok(""), "home": ok("/h\n")}), gpu_hours_per_unit={"GPU": 1.0})

    report = cluster.collect(["compute"], now=NOW)

    assert report.budgets == []
    assert report.warnings == ["gpu_hours_per_unit: no budget of resource GPU on this cluster"]


def test_configured_rates_are_not_checked_when_the_budgets_are_unknown():
    cluster = gpu_dummy(dummy_runner(alloc=None), gpu_hours_per_unit={"Boster": 0.25})

    report = cluster.collect(["compute", "jobs"], now=NOW)

    assert report.failed_sections == ["compute"]
    assert report.warnings == []
    # Nor when the budgets are not collected.
    assert gpu_dummy(gpu_hours_per_unit={"Boster": 0.25}).collect(["jobs"], now=NOW).warnings == []


def test_configured_rates_of_a_jobs_only_cluster_are_reported():
    cluster = create_cluster(
        make_config("mine", type="slurm", gpu_hours_per_unit={"GPU": 4.0}),
        runner=FakeRunner({"jobs": ok("")}),
    )

    report = cluster.collect(now=NOW)

    assert report.warnings == [
        "the gpu_hours_per_unit setting is ignored: this type of cluster reports no compute budgets"
    ]
    assert create_cluster(make_config("mine", type="slurm"), runner=FakeRunner()).collect(now=NOW).warnings == []


def test_gpu_summary_names_clusters_like_their_titles():
    other = lumi_report(name="lumi-other")

    table = gpu_summary_table([lumi_report(), other], TODAY)

    assert table is not None
    assert [row[0] for row in rows(table)] == ["LUMI", "LUMI (lumi-other)"]


def ascii_console(width: int = 120) -> Console:
    return Console(file=io.TextIOWrapper(io.BytesIO(), encoding="ascii"), width=width, color_system=None)


def ascii_output(console: Console) -> str:
    console.file.flush()
    return console.file.buffer.getvalue().decode("ascii")  # type: ignore[attr-defined]


def test_titles_are_printable_on_ascii_only_terminals():
    console = ascii_console()
    reports = [lumi_report(gpus="4× AMD MI250X 128 GB (2 GCDs each) – é"), leonardo_report()]

    render_reports(reports, console, TODAY)

    text = ascii_output(console)
    assert "LUMI - GPU nodes: 4x AMD MI250X 128 GB (2 GCDs each) ? ?" in text
    assert "Leonardo - GPU nodes: 4x NVIDIA A100 64 GB" in text
    assert "~11,602 GPU-h" in text


def test_report_title_for_ascii_only_terminals_keeps_its_styles():
    title = report_title(report(gpus="4× A100"), ascii_only=True)

    assert title.plain == "Leonardo - GPU nodes: 4x A100"
    assert title.style == "bold cyan"
    assert [(title.plain[s.start:s.end], s.style) for s in title.spans] == [(" - GPU nodes: 4x A100", GPU_STYLE)]


def first_lines(console: Console, count: int) -> List[str]:
    return [line.rstrip() for line in console.file.getvalue().splitlines()[:count]]  # type: ignore[attr-defined]


def test_long_gpu_labels_go_on_their_own_line_rather_than_being_cut():
    console = Console(file=io.StringIO(), width=50, color_system=None)

    render_report(lumi_report(name="lumi-gpu", sections=[]), console, TODAY)

    rule, label = first_lines(console, 2)
    assert rule.startswith("LUMI (lumi-gpu) ─")
    assert "…" not in rule
    assert label == "GPU nodes: 4× AMD MI250X 128 GB (2 GCDs each)"


def test_gpu_labels_stay_in_the_rule_when_they_fit():
    title = "LUMI (lumi-gpu) · GPU nodes: 4× AMD MI250X 128 GB (2 GCDs each)"
    console = Console(file=io.StringIO(), width=len(title) + 2, color_system=None)

    render_report(lumi_report(name="lumi-gpu", sections=[]), console, TODAY)

    assert first_lines(console, 2) == [title + " ─", ""]


def test_long_titles_without_gpus_are_left_to_the_rule():
    console = Console(file=io.StringIO(), width=20, color_system=None)

    render_report(report(name="a-very-long-cluster-name", sections=[]), console, TODAY)

    assert first_lines(console, 2)[1] == ""


def test_long_titles_on_ascii_only_terminals():
    console = ascii_console(width=48)

    render_report(lumi_report(sections=[]), console, TODAY)

    rule, label = ascii_output(console).splitlines()[:2]
    assert rule.startswith("LUMI ")
    assert label == "GPU nodes: 4x AMD MI250X 128 GB (2 GCDs each)"
