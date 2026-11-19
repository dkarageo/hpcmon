"""Tests of :mod:`hpcmon.render`: rich tables, report rendering and JSON output.

Values are checked on the returned rich objects (the plain text and style of
every cell), while escaping, headers and the overall layout are checked on the
text rendered by a real :class:`rich.console.Console` writing to a string.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import re
import types
from typing import Any, Iterator, List, Optional, Tuple

import pytest
from rich import box
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from hpcmon.models import ClusterReport, ComputeBudget, Job, StorageQuota
from hpcmon.render import (
    _left_style,
    _remaining,
    _usage_style,
    clean,
    compute_table,
    format_end,
    jobs_table,
    jobs_title,
    render_report,
    render_reports,
    reports_to_json,
    short_percent,
    storage_table,
)
from tests.conftest import NOW, TODAY

GiB = 1024**3
TiB = 1024**4

#: Strings that rich would interpret (or choke on) if they were parsed as markup.
MARKUP_LIKE = ["[bold]x[/bold]", "res [core-h]", "[/red]", "x[/]", "[link=http://evil]y[/link]"]


@pytest.fixture(autouse=True)
def _neutral_terminal_environment(monkeypatch):
    """Keep rich's environment detection from changing what the tests see."""
    for name in (
        "NO_COLOR",
        "FORCE_COLOR",
        "TERM",
        "COLORTERM",
        "COLUMNS",
        "LINES",
        "TTY_COMPATIBLE",
        "TTY_INTERACTIVE",
        "JUPYTER_COLUMNS",
        "JUPYTER_LINES",
    ):
        monkeypatch.delenv(name, raising=False)


# -- helpers ------------------------------------------------------------------


def make_console(color: bool = False) -> Console:
    """A console writing to a string, fully independent of the environment."""
    return Console(
        file=io.StringIO(),
        width=250,
        height=100,
        color_system="truecolor" if color else None,
        force_terminal=color,
        no_color=False,
        legacy_windows=False,
    )


def output_of(console: Console) -> str:
    return console.file.getvalue()  # type: ignore[attr-defined]


def render(renderable: Any, color: bool = False) -> str:
    """Render ``renderable`` and return what would appear in the terminal."""
    console = make_console(color)
    console.print(renderable)
    return output_of(console)


def headers(table: Table) -> List[str]:
    return [str(column.header) for column in table.columns]


def rows(table: Table) -> List[Tuple[str, ...]]:
    """The plain text of every cell, row by row."""
    columns = [list(column.cells) for column in table.columns]
    return [tuple(cell.plain for cell in row) for row in zip(*columns)]


def styles(table: Table) -> List[Tuple[str, ...]]:
    """The style of every cell, row by row."""
    columns = [list(column.cells) for column in table.columns]
    return [tuple(str(cell.style) for cell in row) for row in zip(*columns)]


def rendered_rows(text: str) -> List[List[str]]:
    """Rendered table lines split into cells (columns are 2+ spaces apart)."""
    return [re.split(r"\s{2,}", line.strip()) for line in text.splitlines() if line.strip()]


def title_line(text: str) -> str:
    """The title of the rule on the first rendered line, without the rule."""
    return text.splitlines()[0].replace("─", "").strip()


class RecordingConsole:
    """Duck-typed console recording what is printed instead of rendering it."""

    width = 250

    def __init__(self) -> None:
        self.printed: List[Tuple[Any, ...]] = []
        self.options = types.SimpleNamespace(ascii_only=False)

    def print(self, *objects: Any, **kwargs: Any) -> None:
        assert not kwargs
        self.printed.append(objects)

    @property
    def renderables(self) -> List[Any]:
        """Everything printed, with ``None`` for the bare ``print()`` calls."""
        return [objects[0] if objects else None for objects in self.printed]


def report(**kwargs: Any) -> ClusterReport:
    kwargs.setdefault("name", "lumi")
    kwargs.setdefault("system", "LUMI")
    kwargs.setdefault("sections", ["compute", "storage", "jobs"])
    return ClusterReport(**kwargs)


def lumi_budget(**kwargs: Any) -> ComputeBudget:
    values = dict(
        project="project_465000123",
        resource="GPU",
        unit="GPU-h",
        total=96_000,
        used=3_187,
        end=dt.date(2027, 3, 31),
    )
    values.update(kwargs)
    return ComputeBudget(**values)


def fixture_jobs() -> List[Job]:
    """Jobs modelled on ``tests/fixtures/slurm/squeue.txt``, built directly.

    They are listed in the order of :func:`hpcmon.clusters.slurm.parse_squeue`,
    with one deviation: the node count of the array job is unknown (``None``).
    """
    return [
        Job("1234567", "train|model [v2]", "standard-g", "project_465000123", "RUNNING", "1:02:03", "2-00:00:00", 4,
            "nid[005000-005003]"),
        Job("1234570", "cleanup", "dev-g", "project_465000123", "COMPLETING", "10:00", "30:00", 1, "nid005100"),
        Job("1234568", "eval", "small-g", "project_465000123", "PENDING", "0:00", "1:00:00", 1, "(Priority)"),
        Job("1234569_[1-10]", "sweep", "small-g", "project_465000999", "PENDING", "0:00", "30:00", None,
            "(AssocGrpGRESMinutes)"),
        Job("1234571", "suspended-job", "small", "project_465000123", "SUSPENDED", "5:00", "30:00", 1, "nid001001"),
    ]


#: A string coming from a cluster that tries to take over the terminal: colour
#: (CSI), window title and hyperlink (OSC), cursor movement, a reverse index, a
#: carriage return overwriting the line, a bell and a C1 control character.
HOSTILE = "\x1b[31mred\x1b[0m\x1b]0;pwned\x07\x1b]8;;http://evil\x1b\\link\x1b]8;;\x1b\\\x1b[2Aup\x1bM\rcr\x07\x9b"

#: Any terminal control character that must never be printed.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def assert_terminal_safe(text: str) -> None:
    """``text`` contains no escape sequence or control character (but newlines)."""
    assert "\x1b" not in text
    assert not _CONTROL_CHARS.search(text), repr(text)


# -- clean ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        (None, ""),
        ("", ""),
        ("plain text", "plain text"),
        ("unicode: é ü 中 – ok", "unicode: é ü 中 – ok"),
        ("[bold]x[/bold]", "[bold]x[/bold]"),
        # CSI sequences: colours, cursor movement, private modes.
        ("\x1b[1;31mred\x1b[0m", "red"),
        ("\x1b[mreset", "reset"),
        ("a\x1b[2Ab", "ab"),
        ("a\x1b[?25lb\x1b[?25h", "ab"),
        ("a\x1b[1 qb", "ab"),
        # OSC sequences terminated by BEL or ST, and unterminated.
        ("\x1b]0;window title\x07x", "x"),
        ("\x1b]2;title\x1b\\x", "x"),
        ("\x1b]8;;http://evil\x1b\\link\x1b]8;;\x1b\\", "link"),
        ("x\x1b]0;unterminated", "x"),
        ("x\x1b]0;stops at\x1b[1mnext", "xnext"),
        # Other escape sequences (ESC followed by one byte in @-_).
        ("\x1bMreverse index", "reverse index"),
        ("a\x1bPb", "ab"),
        ("a\x1b_b", "ab"),
        ("lone escape\x1b", "lone escape"),
        ("\x1b", ""),
        ("\x1b[", ""),
        # Newlines and tabs become spaces, other control characters vanish.
        ("a\nb", "a b"),
        ("a\tb", "a b"),
        ("a\r\nb", "a b"),
        ("line\roverwritten", "lineoverwritten"),
        ("a\x00b\x07c\x08d", "abcd"),
        ("a\x0bb\x0cc\x1fd", "abcd"),
        ("a\x7fb", "ab"),
        ("a\x80b\x9bc\x9fd", "abcd"),
        # Characters just outside the control ranges are kept.
        ("\xa0nbsp \x20 ~", "\xa0nbsp   ~"),
    ],
)
def test_clean_removes_escape_sequences_and_control_characters(text: Optional[str], expected: str) -> None:
    assert clean(text) == expected


def test_clean_hostile_string_keeps_only_the_visible_text() -> None:
    cleaned = clean(HOSTILE)

    assert cleaned == "redlinkupcr"
    assert_terminal_safe(cleaned)


@pytest.mark.parametrize("sequence", ["\x1b(B", "\x1b)0", "\x1b7", "\x1b8", "\x1bc", "\x1b=", "\x1b#8", "\x1b%G"])
def test_clean_never_lets_an_escape_character_through(sequence: str) -> None:
    cleaned = clean(f"a{sequence}b")

    assert_terminal_safe(cleaned)
    assert cleaned.startswith("a") and cleaned.endswith("b")


@pytest.mark.parametrize("sequence", ["\x1b(B", "\x1b7", "\x1b8", "\x1bc"])
def test_clean_strips_other_escape_sequences_completely(sequence: str) -> None:
    assert clean(f"a{sequence}b") == "ab"


# -- colour thresholds ----------------------------------------------------------


@pytest.mark.parametrize(
    "fraction, expected",
    [
        (None, ""),
        (1.5, "green"),
        (1.0, "green"),
        (0.5, "green"),
        (0.4999, "yellow"),
        (0.2, "yellow"),
        (0.1999, "red"),
        (0.0, "red"),
        (-0.25, "red"),
    ],
)
def test_left_style_gets_redder_as_less_is_left(fraction: Optional[float], expected: str) -> None:
    assert _left_style(fraction) == expected


@pytest.mark.parametrize(
    "fraction, expected",
    [
        (None, ""),
        (-0.1, "green"),
        (0.0, "green"),
        (0.7999, "green"),
        (0.8, "yellow"),
        (0.9499, "yellow"),
        (0.95, "red"),
        (1.0, "red"),
        (1.2, "red"),
    ],
)
def test_usage_style_gets_redder_as_more_is_used(fraction: Optional[float], expected: str) -> None:
    assert _usage_style(fraction) == expected


# -- format_end -----------------------------------------------------------------


@pytest.mark.parametrize(
    "end, expected_text, expected_style",
    [
        (dt.date(2025, 10, 18), "2025-10-18 (ended)", "red"),
        (dt.date(2026, 11, 17), "2026-11-17 (ended)", "red"),
        (dt.date(2026, 11, 18), "2026-11-18 (last day)", "bold red"),
        (dt.date(2026, 11, 19), "2026-11-19 (1d)", "red"),
        (dt.date(2026, 11, 20), "2026-11-20 (2d)", "red"),
        (dt.date(2026, 12, 2), "2026-12-02 (14d)", "red"),
        (dt.date(2026, 12, 3), "2026-12-03 (15d)", "yellow"),
        (dt.date(2026, 12, 18), "2026-12-18 (30d)", "yellow"),
        (dt.date(2026, 12, 19), "2026-12-19 (31d)", ""),
        (dt.date(2027, 3, 31), "2027-03-31 (133d)", ""),
        (dt.date(2027, 11, 18), "2027-11-18 (365d)", ""),
    ],
)
def test_format_end_shows_date_days_left_and_urgency(end: dt.date, expected_text: str, expected_style: str) -> None:
    text = format_end(lumi_budget(end=end), TODAY)

    assert isinstance(text, Text)
    assert text.plain == expected_text
    assert text.style == expected_style


def test_format_end_without_end_date_is_a_dim_dash() -> None:
    text = format_end(lumi_budget(end=None, start=dt.date(2026, 1, 1)), TODAY)

    assert text.plain == "-"
    assert text.style == "dim"


def test_format_end_counts_days_from_the_given_day_not_the_clock() -> None:
    budget = lumi_budget(end=dt.date(2026, 12, 31))

    assert format_end(budget, dt.date(2026, 12, 1)).plain == "2026-12-31 (30d)"
    assert format_end(budget, dt.date(2026, 12, 31)).plain == "2026-12-31 (last day)"
    assert format_end(budget, dt.date(2027, 1, 1)).plain == "2026-12-31 (ended)"


# -- short_percent --------------------------------------------------------------


@pytest.mark.parametrize(
    "fraction, expected",
    [
        (None, "-"),
        (0, "0.0%"),
        (0.0, "0.0%"),
        (0.0004, "0.0%"),
        (0.047, "4.7%"),
        (0.0994, "9.9%"),
        (0.09949, "9.9%"),
        # One decimal only below 9.95 %: from there on it would round to "10.0%".
        (0.0995, "10%"),
        (0.1, "10%"),
        (0.25, "25%"),
        (0.98, "98%"),
        (92_813 / 96_000, "97%"),
        (0.9996, "100%"),
        (1.0, "100%"),
        (1.5, "150%"),
        (-0.05, "-5.0%"),
        (-0.1, "-10%"),
    ],
)
def test_short_percent_shows_a_decimal_only_for_small_percentages(fraction: Optional[float], expected: str) -> None:
    assert short_percent(fraction) == expected


# -- _remaining -----------------------------------------------------------------


@pytest.mark.parametrize(
    "value, fraction, expected_text, expected_style",
    [
        (None, None, "-", "dim"),
        (None, 0.5, "-", "dim"),
        (92_813, 92_813 / 96_000, "92,813 (97%)", "green"),
        (750, 0.75, "750 (75%)", "green"),
        (60, 0.3, "60 (30%)", "yellow"),
        (50, 0.05, "50 (5.0%)", "red"),
        (0, 0.0, "0 (0.0%)", "red"),
        (-50.5, -0.05, "-50.5 (-5.0%)", "red"),
        (5, None, "5", "bold"),
        (12.25, None, "12.2", "bold"),
    ],
)
def test_remaining_formats_amount_percentage_and_colour(
    value: Optional[float], fraction: Optional[float], expected_text: str, expected_style: str
) -> None:
    text = _remaining(value, fraction)

    assert text.plain == expected_text
    assert text.style == expected_style


@pytest.mark.parametrize(
    "value, fraction, expected_text",
    [
        (92_813, 92_813 / 96_000, "92,813 (97%)"),
        (50, 0.05, "50 (5.0%)"),
        (5, None, "5"),
    ],
)
def test_remaining_of_an_ended_budget_is_dim(value: float, fraction: Optional[float], expected_text: str) -> None:
    text = _remaining(value, fraction, ended=True)

    assert text.plain == expected_text
    assert text.style == "dim"


def test_remaining_unknown_value_is_a_dim_dash_even_when_ended() -> None:
    text = _remaining(None, None, ended=True)

    assert (text.plain, text.style) == ("-", "dim")


# -- compute_table ----------------------------------------------------------------

COMPUTE_HEADERS = ["Project", "Resource", "Total", "Remaining", "Ends (days left)"]
MONTHLY_HEADERS = ["Project", "Resource", "Total", "Remaining", "Month\ntotal", "Month\nremaining", "Ends (days left)"]


def test_compute_table_basic_columns_and_values() -> None:
    budgets = [
        lumi_budget(),
        ComputeBudget(project="project_465000999", resource="CPU", unit="core-h"),
        ComputeBudget(project="hpc_proj", resource="Booster", unit="core-h", total=1_000, used=900, available=50),
    ]

    table = compute_table(budgets, TODAY)

    assert table.title == "Compute allocations"
    assert table.title_justify == "left"
    assert table.title_style == "bold"
    assert table.box is box.SIMPLE_HEAD
    assert table.caption is None
    assert headers(table) == COMPUTE_HEADERS
    assert [column.justify for column in table.columns] == ["left", "left", "right", "right", "right"]
    assert all(column.overflow == "fold" for column in table.columns)
    assert rows(table) == [
        ("project_465000123", "GPU [GPU-h]", "96,000", "92,813 (97%)", "2027-03-31 (133d)"),
        ("project_465000999", "CPU [core-h]", "-", "-", "-"),
        ("hpc_proj", "Booster [core-h]", "1,000", "50 (5.0%)", "-"),
    ]
    assert styles(table) == [
        ("", "", "", "green", ""),
        ("", "", "dim", "dim", "dim"),
        ("", "", "", "red", "dim"),
    ]


def test_compute_table_without_budgets_has_only_the_header() -> None:
    table = compute_table([], TODAY)

    assert headers(table) == COMPUTE_HEADERS
    assert table.row_count == 0
    assert table.caption is None


def test_compute_table_adds_monthly_columns_when_any_budget_is_monthly() -> None:
    budgets = [
        lumi_budget(),
        ComputeBudget(
            project="p200000",
            resource="GPU",
            unit="node-h",
            month_total=1_000,
            month_used=250,
            end=dt.date(2026, 12, 2),
        ),
        ComputeBudget(project="p200001", resource="CPU", unit="node-h", month_used=20),
        ComputeBudget(project="p200002", resource="FPGA", unit="node-h", month_total=8, month_used=7.5),
    ]

    table = compute_table(budgets, TODAY)

    assert headers(table) == MONTHLY_HEADERS
    assert [column.justify for column in table.columns] == ["left", "left"] + ["right"] * 5
    assert all(column.overflow == "fold" for column in table.columns)
    assert rows(table) == [
        ("project_465000123", "GPU [GPU-h]", "96,000", "92,813 (97%)", "-", "-", "2027-03-31 (133d)"),
        ("p200000", "GPU [node-h]", "-", "-", "1,000", "750 (75%)", "2026-12-02 (14d)"),
        ("p200001", "CPU [node-h]", "-", "-", "-", "-", "-"),
        ("p200002", "FPGA [node-h]", "-", "-", "8", "0.5 (6.2%)", "-"),
    ]
    assert [row[4] for row in styles(table)] == ["dim", "", "dim", ""]
    assert [row[5] for row in styles(table)] == ["dim", "green", "dim", "red"]
    assert [row[6] for row in styles(table)] == ["", "red", "dim", "dim"]


@pytest.mark.parametrize(
    "budget",
    [
        ComputeBudget(project="p", resource="CPU", unit="node-h", month_total=10),
        ComputeBudget(project="p", resource="CPU", unit="node-h", month_used=0),
    ],
    ids=["month_total-only", "month_used-zero-only"],
)
def test_compute_table_monthly_columns_are_triggered_by_either_monthly_field(budget: ComputeBudget) -> None:
    assert headers(compute_table([budget], TODAY)) == MONTHLY_HEADERS


def test_compute_table_shows_project_and_account_when_they_differ() -> None:
    budgets = [
        ComputeBudget(project="myproject", resource="CPU", unit="core-h", account="budget_acc"),
        ComputeBudget(project="other", resource="CPU", unit="core-h", account=None),
        ComputeBudget(project="third", resource="CPU", unit="core-h", account=""),
    ]

    table = compute_table(budgets, TODAY)

    assert [row[0] for row in rows(table)] == ["myproject (budget_acc)", "other", "third"]
    assert [row[0] for row in styles(table)] == ["", "", ""]


def test_compute_table_dims_the_remaining_amounts_of_ended_budgets() -> None:
    ended = dict(total=1_000, used=100, month_total=100, month_used=95)
    budgets = [
        ComputeBudget(project="ended", resource="CPU", unit="core-h", end=dt.date(2026, 11, 17), **ended),
        ComputeBudget(project="last-day", resource="CPU", unit="core-h", end=dt.date(2026, 11, 18), **ended),
        ComputeBudget(project="no-end", resource="CPU", unit="core-h", **ended),
        ComputeBudget(project="ended-no-total", resource="CPU", unit="core-h", available=40, month_total=0,
                      month_used=0, end=dt.date(2025, 1, 1)),
        ComputeBudget(project="open-no-total", resource="CPU", unit="core-h", available=40, month_total=0,
                      month_used=0, end=dt.date(2027, 1, 1)),
    ]

    table = compute_table(budgets, TODAY)

    assert headers(table) == MONTHLY_HEADERS
    assert [row[3:] for row in rows(table)] == [
        ("900 (90%)", "100", "5 (5.0%)", "2026-11-17 (ended)"),
        ("900 (90%)", "100", "5 (5.0%)", "2026-11-18 (last day)"),
        ("900 (90%)", "100", "5 (5.0%)", "-"),
        ("40", "0", "0", "2025-01-01 (ended)"),
        ("40", "0", "0", "2027-01-01 (44d)"),
    ]
    assert [(row[3], row[5], row[6]) for row in styles(table)] == [
        ("dim", "dim", "red"),
        ("green", "red", "bold red"),
        ("green", "red", "dim"),
        ("dim", "dim", "red"),
        ("bold", "bold", ""),
    ]


def test_compute_table_ended_state_follows_the_given_day() -> None:
    budget = lumi_budget(end=dt.date(2026, 12, 31))

    assert styles(compute_table([budget], dt.date(2026, 12, 31)))[0][3] == "green"
    assert styles(compute_table([budget], dt.date(2027, 1, 1)))[0][3] == "dim"


def test_compute_table_adds_notes_column_last_when_the_notes_differ() -> None:
    budgets = [
        ComputeBudget(project="p200000", resource="GPU", unit="node-h", month_total=100, month_used=0,
                      note="monthly allocation, unused hours expire"),
        ComputeBudget(project="p200001", resource="CPU", unit="node-h", total=10, used=10),
    ]

    table = compute_table(budgets, TODAY)

    assert headers(table)[-2:] == ["Ends (days left)", "Notes"]
    assert table.columns[-1].justify == "left"
    assert table.columns[-1].overflow == "fold"
    assert table.caption is None
    assert [row[-1] for row in rows(table)] == ["monthly allocation, unused hours expire", ""]
    assert [row[-1] for row in styles(table)] == ["dim", "dim"]
    assert rows(table)[1][2:4] == ("10", "0 (0.0%)")


def test_compute_table_two_different_notes_get_a_notes_column() -> None:
    table = compute_table([lumi_budget(note="LUMI-G"), lumi_budget(resource="CPU", note="LUMI-C")], TODAY)

    assert headers(table)[-1] == "Notes"
    assert table.caption is None
    assert [row[-1] for row in rows(table)] == ["LUMI-G", "LUMI-C"]


@pytest.mark.parametrize("count", [1, 3])
def test_compute_table_note_shared_by_all_rows_becomes_the_caption(count: int) -> None:
    budgets = [lumi_budget(resource=f"R{i}", note="current month only") for i in range(count)]

    table = compute_table(budgets, TODAY)

    assert headers(table) == COMPUTE_HEADERS
    assert isinstance(table.caption, Text)
    assert table.caption.plain == "Note: current month only"
    assert table.caption.style == "dim"
    assert table.caption_justify == "left"
    assert len(rows(table)) == count


def test_compute_table_caption_is_rendered_below_the_table() -> None:
    budgets = [lumi_budget(note="storage billing"), lumi_budget(resource="CPU", note="storage billing")]

    lines = render(compute_table(budgets, TODAY)).splitlines()

    assert lines[-1].strip() == "Note: storage billing"
    assert not lines[-1].startswith(" " * 10)
    assert "Notes" not in "\n".join(lines)


def test_compute_table_note_on_some_rows_only_is_not_a_caption() -> None:
    table = compute_table([lumi_budget(note="LUMI-G"), lumi_budget(note=None)], TODAY)

    assert table.caption is None
    assert headers(table)[-1] == "Notes"
    assert [row[-1] for row in rows(table)] == ["LUMI-G", ""]


@pytest.mark.parametrize(
    "notes", [["", None], ["", ""], [None, None], [""]], ids=["mixed", "empty", "none", "one-empty"]
)
def test_compute_table_ignores_empty_notes(notes: List[Optional[str]]) -> None:
    table = compute_table([lumi_budget(note=note) for note in notes], TODAY)

    assert headers(table) == COMPUTE_HEADERS
    assert table.caption is None


def test_compute_table_renders_headers_and_values_in_the_terminal() -> None:
    budgets = [lumi_budget(note="LUMI-G"), ComputeBudget(project="p2", resource="CPU", unit="core-h", month_used=3)]

    out = render(compute_table(budgets, TODAY))
    lines = rendered_rows(out)

    assert lines[0] == ["Compute allocations"]
    # The monthly headers span two lines.
    assert lines[1] == ["Month", "Month"]
    assert lines[2] == ["Project", "Resource", "Total", "Remaining", "total", "remaining", "Ends (days left)", "Notes"]
    assert set(lines[3]) == {"─" * len(lines[3][0])}
    assert lines[4] == ["project_465000123", "GPU [GPU-h]", "96,000", "92,813 (97%)", "-", "-",
                        "2027-03-31 (133d)", "LUMI-G"]
    assert lines[5] == ["p2", "CPU [core-h]", "-", "-", "-", "-", "-"]
    assert len(lines) == 6


def test_compute_table_right_aligns_numbers_in_the_terminal() -> None:
    budgets = [lumi_budget(), lumi_budget(project="p", total=5, used=1)]

    lines = render(compute_table(budgets, TODAY)).splitlines()
    header = next(line for line in lines if "Project" in line)
    first, second = [line for line in lines if "GPU [GPU-h]" in line]

    # "Total" and the amounts end in the same terminal column.
    end = header.index("Total") + len("Total")
    assert first[:end].endswith("96,000")
    assert second[:end].endswith("5")


def test_compute_table_folds_long_values_instead_of_truncating_them() -> None:
    project = "a_very_long_project_identifier_" + "x" * 40
    console = Console(file=io.StringIO(), width=60, color_system=None)

    console.print(compute_table([lumi_budget(project=project)], TODAY))
    lines = output_of(console).splitlines()

    assert not any("…" in line for line in lines)
    rule = next(i for i, line in enumerate(lines) if line.strip().startswith("─"))
    data = [line for line in lines[rule + 1:] if line.strip()]
    assert len(data) > 2
    assert "".join(line.split()[0] for line in data) == project
    assert "".join(data).count("2027-03-31") == 1


@pytest.mark.parametrize("markup", MARKUP_LIKE)
def test_compute_table_shows_markup_like_text_literally(markup: str) -> None:
    budgets = [
        ComputeBudget(project=markup, resource=markup, unit=markup, total=10, used=1, note=markup),
        ComputeBudget(project="p", resource="CPU", unit="core-h", note="other"),
    ]

    lines = rendered_rows(render(compute_table(budgets, TODAY)))

    assert lines[3] == [markup, f"{markup} [{markup}]", "10", "9 (90%)", "-", markup]


@pytest.mark.parametrize("markup", MARKUP_LIKE)
def test_compute_table_shows_markup_like_accounts_and_captions_literally(markup: str) -> None:
    budget = ComputeBudget(project=markup, resource="CPU", unit="core-h", account=markup, note=markup)

    out = render(compute_table([budget], TODAY))

    assert f"{markup} ({markup})" in out
    assert out.splitlines()[-1].strip() == f"Note: {markup}"


def test_compute_table_resource_with_unit_in_brackets_is_literal() -> None:
    out = render(compute_table([ComputeBudget(project="[bold]x[/bold]", resource="res", unit="core-h")], TODAY))

    assert "[bold]x[/bold]" in out
    assert "res [core-h]" in out


def test_compute_table_cleans_every_remote_string() -> None:
    budgets = [
        ComputeBudget(project=f"p\x1b[31m1{HOSTILE}", resource=f"G\x1b]0;t\x07PU\n", unit="GPU-h\x07",
                      account=f"acc\t{HOSTILE}", note=f"note\x1b[2J{HOSTILE}"),
        ComputeBudget(project="p2", resource="CPU", unit="core-h", note=f"other\r{HOSTILE}"),
    ]

    table = compute_table(budgets, TODAY)
    out = render(table)

    assert rows(table)[0][:2] == ("p1redlinkupcr (acc redlinkupcr)", "GPU  [GPU-h]")
    assert [row[-1] for row in rows(table)] == ["noteredlinkupcr", "otherredlinkupcr"]
    assert_terminal_safe(out)


def test_compute_table_cleans_the_caption() -> None:
    budgets = [lumi_budget(note=f"shared\n{HOSTILE}"), lumi_budget(resource="CPU", note=f"shared\n{HOSTILE}")]

    table = compute_table(budgets, TODAY)

    assert table.caption.plain == "Note: shared redlinkupcr"
    assert_terminal_safe(render(table))


# -- storage_table ----------------------------------------------------------------

STORAGE_HEADERS = ["Area", "Used", "Quota", "Free", "Use", "Files / quota"]


def test_storage_table_columns_and_values() -> None:
    quotas = [
        StorageQuota("/users/jdoe", None, used_bytes=5 * GiB, limit_bytes=20 * GiB, files_used=12_345,
                     files_limit=100_000),
        StorageQuota("/scratch/project_465000123", "project_465000123", used_bytes=int(9.5 * TiB),
                     limit_bytes=10 * TiB, files_used=1_900_000, files_limit=2_000_000),
        StorageQuota("/project/project_465000123", "project_465000123", used_bytes=85 * GiB, limit_bytes=100 * GiB,
                     files_used=850, files_limit=1_000),
        StorageQuota("/flash/project_465000123", "project_465000123", used_bytes=1_024),
        StorageQuota("/work/empty", "", used_bytes=0, limit_bytes=0, files_used=3, files_limit=0),
    ]

    table = storage_table(quotas)

    assert table.title == "Storage"
    assert table.title_justify == "left"
    assert table.title_style == "bold"
    assert table.box is box.SIMPLE_HEAD
    assert table.caption is None
    assert headers(table) == STORAGE_HEADERS
    assert [column.justify for column in table.columns] == ["left"] + ["right"] * 5
    assert all(column.overflow == "fold" for column in table.columns)
    assert rows(table) == [
        ("/users/jdoe", "5.0 GiB", "20.0 GiB", "15.0 GiB", "25%", "12.3k / 100k"),
        ("/scratch/project_465000123", "9.5 TiB", "10.0 TiB", "512.0 GiB", "95%", "1.9M / 2M"),
        ("/project/project_465000123", "85.0 GiB", "100.0 GiB", "15.0 GiB", "85%", "850 / 1k"),
        ("/flash/project_465000123", "1.0 KiB", "no quota", "-", "-", "-"),
        ("/work/empty", "0 B", "0 B", "0 B", "-", "3 / 0"),
    ]
    assert styles(table) == [
        ("", "", "", "bold", "green", "green"),
        ("", "", "", "bold", "red", "red"),
        ("", "", "", "bold", "yellow", "yellow"),
        ("", "", "dim", "bold", "", "dim"),
        ("", "", "", "bold", "", ""),
    ]


@pytest.mark.parametrize(
    "used, limit, expected_text, expected_style",
    [
        (5 * GiB, 20 * GiB, "20.0 GiB", ""),
        (None, 20 * GiB, "20.0 GiB", ""),
        (0, 0, "0 B", ""),
        (1_024, None, "no quota", "dim"),
        (0, None, "no quota", "dim"),
        (None, None, "-", "dim"),
    ],
)
def test_storage_table_quota_cell(
    used: Optional[int], limit: Optional[int], expected_text: str, expected_style: str
) -> None:
    table = storage_table([StorageQuota("/scratch/p", "p", used_bytes=used, limit_bytes=limit)])

    assert rows(table)[0][2] == expected_text
    assert styles(table)[0][2] == expected_style


@pytest.mark.parametrize(
    "files_used, files_limit, expected_text, expected_style",
    [
        (None, None, "-", "dim"),
        (12, None, "12 / -", ""),
        (None, 100, "- / 100", ""),
        (0, 100, "0 / 100", "green"),
        (79, 100, "79 / 100", "green"),
        (80, 100, "80 / 100", "yellow"),
        (95, 100, "95 / 100", "red"),
        (120_000, 100_000, "120k / 100k", "red"),
        (5, 0, "5 / 0", ""),
    ],
)
def test_storage_table_files_cell_shows_used_and_limit(
    files_used: Optional[int], files_limit: Optional[int], expected_text: str, expected_style: str
) -> None:
    table = storage_table([StorageQuota("/scratch/p", "p", files_used=files_used, files_limit=files_limit)])

    assert rows(table)[0][5] == expected_text
    assert styles(table)[0][5] == expected_style


def test_storage_table_colours_files_by_the_file_quota_not_the_byte_quota() -> None:
    quotas = [
        StorageQuota("/users/jdoe", None, used_bytes=5 * GiB, limit_bytes=20 * GiB, files_used=99_000,
                     files_limit=100_000),
        StorageQuota("/scratch/p", "p", used_bytes=99, limit_bytes=100, files_used=10, files_limit=100),
        StorageQuota("/flash/p", "p", used_bytes=85, limit_bytes=100, files_used=5),
    ]

    table = storage_table(quotas)

    assert [row[4:6] for row in rows(table)] == [("25%", "99k / 100k"), ("99%", "10 / 100"), ("85%", "5 / -")]
    assert [row[4:6] for row in styles(table)] == [("green", "red"), ("red", "green"), ("yellow", "")]


def test_storage_table_use_column_has_a_decimal_only_below_ten_percent() -> None:
    quotas = [
        StorageQuota("/a", used_bytes=47, limit_bytes=1_000),
        StorageQuota("/b", used_bytes=995, limit_bytes=10_000),
        StorageQuota("/c", used_bytes=1_000, limit_bytes=1_000),
    ]

    assert [row[4] for row in rows(storage_table(quotas))] == ["4.7%", "10%", "100%"]


def test_storage_table_without_quotas_has_only_the_header() -> None:
    table = storage_table([])

    assert headers(table) == STORAGE_HEADERS
    assert table.row_count == 0
    assert table.caption is None


def test_storage_table_has_no_project_column() -> None:
    table = storage_table([StorageQuota("/scratch/project_465000123", "project_465000123", used_bytes=1)])

    assert "Project" not in headers(table)
    assert all("project_465000123" != cell for cell in rows(table)[0])


def test_storage_table_renders_personal_and_unlimited_areas() -> None:
    quotas = [
        StorageQuota("/users/jdoe", used_bytes=5 * GiB, limit_bytes=20 * GiB),
        StorageQuota("/flash/project_465000123", "project_465000123", used_bytes=1_024),
        StorageQuota("/unknown/area"),
    ]

    lines = rendered_rows(render(storage_table(quotas)))

    assert lines[0] == ["Storage"]
    assert lines[1] == STORAGE_HEADERS
    assert lines[3] == ["/users/jdoe", "5.0 GiB", "20.0 GiB", "15.0 GiB", "25%", "-"]
    assert lines[4] == ["/flash/project_465000123", "1.0 KiB", "no quota", "-", "-", "-"]
    assert lines[5] == ["/unknown/area", "-", "-", "-", "-", "-"]
    assert len(lines) == 6


def test_storage_table_adds_notes_column_when_the_notes_differ() -> None:
    quotas = [
        StorageQuota("/scratch/p", "p", used_bytes=1, limit_bytes=2, note="over quota"),
        StorageQuota("/project/p", "p", used_bytes=1, limit_bytes=2),
        StorageQuota("/flash/p", "p", used_bytes=1, limit_bytes=2, note="as of 3 hours ago"),
    ]

    table = storage_table(quotas)

    assert headers(table) == STORAGE_HEADERS + ["Notes"]
    assert table.columns[-1].justify == "left"
    assert table.columns[-1].overflow == "fold"
    assert table.caption is None
    assert [row[-1] for row in rows(table)] == ["over quota", "", "as of 3 hours ago"]
    assert [row[-1] for row in styles(table)] == ["dim", "dim", "dim"]


@pytest.mark.parametrize("count", [1, 4])
def test_storage_table_note_shared_by_all_rows_becomes_the_caption(count: int) -> None:
    quotas = [StorageQuota(f"/area{i}", used_bytes=1, note="as of 3 hours ago") for i in range(count)]

    table = storage_table(quotas)

    assert headers(table) == STORAGE_HEADERS
    assert isinstance(table.caption, Text)
    assert table.caption.plain == "Note: as of 3 hours ago"
    assert table.caption.style == "dim"
    assert table.caption_justify == "left"
    assert len(rows(table)) == count


def test_storage_table_caption_is_rendered_below_the_table() -> None:
    quotas = [StorageQuota("/a", used_bytes=1, note="as of 1 day ago"), StorageQuota("/b", note="as of 1 day ago")]

    lines = render(storage_table(quotas)).splitlines()

    assert lines[-1].strip() == "Note: as of 1 day ago"
    assert "Notes" not in "\n".join(lines)


@pytest.mark.parametrize("notes", [["", None], ["", ""], [None, None]], ids=["mixed", "empty", "none"])
def test_storage_table_ignores_empty_notes(notes: List[Optional[str]]) -> None:
    table = storage_table([StorageQuota(f"/a{i}", note=note) for i, note in enumerate(notes)])

    assert headers(table) == STORAGE_HEADERS
    assert table.caption is None


@pytest.mark.parametrize("markup", MARKUP_LIKE)
def test_storage_table_shows_markup_like_paths_and_notes_literally(markup: str) -> None:
    quotas = [StorageQuota(f"/scratch/{markup}", markup, used_bytes=1, note=markup), StorageQuota("/other")]

    lines = rendered_rows(render(storage_table(quotas)))

    assert lines[3] == [f"/scratch/{markup}", "1 B", "no quota", "-", "-", "-", markup]


@pytest.mark.parametrize("markup", MARKUP_LIKE)
def test_storage_table_shows_markup_like_captions_literally(markup: str) -> None:
    out = render(storage_table([StorageQuota("/scratch/p", used_bytes=1, note=markup)]))

    assert out.splitlines()[-1].strip() == f"Note: {markup}"


def test_storage_table_cleans_every_remote_string() -> None:
    quotas = [
        StorageQuota(f"/scratch/\x1b[1mp{HOSTILE}", "p", used_bytes=1, note=f"over\nquota{HOSTILE}"),
        StorageQuota("/b\x07", note=None),
    ]

    table = storage_table(quotas)

    assert [row[0] for row in rows(table)] == ["/scratch/predlinkupcr", "/b"]
    assert [row[-1] for row in rows(table)] == ["over quotaredlinkupcr", ""]
    assert_terminal_safe(render(table))


def test_storage_table_cleans_the_caption() -> None:
    table = storage_table([StorageQuota("/a", note=f"as of\t1 day ago{HOSTILE}")])

    assert table.caption.plain == "Note: as of 1 day agoredlinkupcr"
    assert_terminal_safe(render(table))


# -- jobs -----------------------------------------------------------------------

JOBS_HEADERS = ["Job ID", "Name", "Partition", "State", "Time / limit", "Nodes", "Node list / reason"]
JOBS_HEADERS_WITH_ACCOUNT = JOBS_HEADERS[:2] + ["Account"] + JOBS_HEADERS[2:]


@pytest.mark.parametrize(
    "states, expected",
    [
        ([], "Jobs: 0 running, 0 pending"),
        (["RUNNING"], "Jobs: 1 running, 0 pending"),
        (["PENDING", "PD"], "Jobs: 0 running, 2 pending"),
        (["R", "PD", "CG"], "Jobs: 2 running, 1 pending"),
        (["SUSPENDED"], "Jobs: 0 running, 0 pending, 1 other"),
        (["RUNNING", "COMPLETING", "PENDING", "PENDING", "SUSPENDED"], "Jobs: 2 running, 2 pending, 1 other"),
        (["FAILED", "TIMEOUT", "running"], "Jobs: 1 running, 0 pending, 2 other"),
        (["SI", "RF", "RD", " R "], "Jobs: 2 running, 2 pending"),
    ],
)
def test_jobs_title_counts_running_pending_and_other_jobs(states: List[str], expected: str) -> None:
    jobs = [Job(job_id=str(i), state=state) for i, state in enumerate(states)]

    assert jobs_title(jobs) == expected


def test_jobs_table_columns_values_and_state_colours() -> None:
    table = jobs_table(fixture_jobs())

    assert table.title == "Jobs: 2 running, 2 pending, 1 other"
    assert table.title_justify == "left"
    assert table.title_style == "bold"
    assert table.box is box.SIMPLE_HEAD
    # The jobs use two accounts, so the account column is shown.
    assert headers(table) == JOBS_HEADERS_WITH_ACCOUNT
    assert [column.justify for column in table.columns] == ["left"] * 5 + ["right"] * 2 + ["left"]
    assert all(column.overflow == "fold" for column in table.columns)
    assert rows(table) == [
        ("1234567", "train|model [v2]", "project_465000123", "standard-g", "RUNNING", "1:02:03 / 2-00:00:00", "4",
         "nid[005000-005003]"),
        ("1234570", "cleanup", "project_465000123", "dev-g", "COMPLETING", "10:00 / 30:00", "1", "nid005100"),
        ("1234568", "eval", "project_465000123", "small-g", "PENDING", "0:00 / 1:00:00", "1", "(Priority)"),
        ("1234569_[1-10]", "sweep", "project_465000999", "small-g", "PENDING", "0:00 / 30:00", "-",
         "(AssocGrpGRESMinutes)"),
        ("1234571", "suspended-job", "project_465000123", "small", "SUSPENDED", "5:00 / 30:00", "1", "nid001001"),
    ]
    assert [row[4] for row in styles(table)] == ["green", "green", "yellow", "yellow", ""]
    assert all(style == "" for row in styles(table) for i, style in enumerate(row) if i != 4)


def test_jobs_table_hides_the_account_column_when_all_jobs_use_one_account() -> None:
    jobs = [job for job in fixture_jobs() if job.account == "project_465000123"]

    table = jobs_table(jobs)

    assert headers(table) == JOBS_HEADERS
    assert [column.justify for column in table.columns] == ["left"] * 4 + ["right"] * 2 + ["left"]
    assert rows(table)[0] == ("1234567", "train|model [v2]", "standard-g", "RUNNING", "1:02:03 / 2-00:00:00", "4",
                              "nid[005000-005003]")
    assert [row[3] for row in styles(table)] == ["green", "green", "yellow", ""]


@pytest.mark.parametrize(
    "accounts, shown",
    [
        ([], False),
        (["a"], False),
        (["a", "a", "a"], False),
        (["", ""], False),
        (["a", "b"], True),
        (["a", ""], True),
        (["a", "a", "b"], True),
    ],
)
def test_jobs_table_account_column_only_with_several_accounts(accounts: List[str], shown: bool) -> None:
    jobs = [Job(job_id=str(i), account=account, state="R") for i, account in enumerate(accounts)]

    table = jobs_table(jobs)

    assert ("Account" in headers(table)) is shown
    if shown:
        assert [row[2] for row in rows(table)] == accounts


def test_jobs_table_with_zero_nodes_shows_zero_not_dash() -> None:
    table = jobs_table([Job(job_id="1", state="PENDING", nodes=0), Job(job_id="2", state="PENDING")])

    assert headers(table) == JOBS_HEADERS
    assert [row[5] for row in rows(table)] == ["0", "-"]


def test_jobs_table_renders_names_with_pipes_and_brackets_literally() -> None:
    jobs = fixture_jobs() + [Job("1234572", "[bold]x[/bold]", "small", "p", "PENDING", "0:00", "[/red]", 2,
                                 reason="[/red]")]

    out = render(jobs_table(jobs))
    lines = rendered_rows(out)

    assert lines[0] == ["Jobs: 2 running, 3 pending, 1 other"]
    assert lines[1] == JOBS_HEADERS_WITH_ACCOUNT
    assert lines[3] == ["1234567", "train|model [v2]", "project_465000123", "standard-g", "RUNNING",
                        "1:02:03 / 2-00:00:00", "4", "nid[005000-005003]"]
    assert lines[6] == ["1234569_[1-10]", "sweep", "project_465000999", "small-g", "PENDING", "0:00 / 30:00", "-",
                        "(AssocGrpGRESMinutes)"]
    assert lines[8] == ["1234572", "[bold]x[/bold]", "p", "small", "PENDING", "0:00 / [/red]", "2", "[/red]"]


def test_jobs_table_cleans_every_remote_string() -> None:
    jobs = [
        Job(f"1\x1b[1m{HOSTILE}", f"name\n{HOSTILE}", f"part\x07{HOSTILE}", f"acc\x1b]0;t\x07{HOSTILE}",
            f"RUNNING\x1b[0m{HOSTILE}", f"1:00\r{HOSTILE}", f"2:00\t{HOSTILE}", 1, f"nid\x1bM{HOSTILE}"),
        Job("2", "x", "p", "other", "PENDING"),
    ]

    table = jobs_table(jobs)

    assert rows(table)[0] == (
        "1redlinkupcr",
        "name redlinkupcr",
        "accredlinkupcr",
        "partredlinkupcr",
        "RUNNINGredlinkupcr",
        "1:00redlinkupcr / 2:00 redlinkupcr",
        "1",
        "nidredlinkupcr",
    )
    assert_terminal_safe(render(table))


# -- render_report ----------------------------------------------------------------


@pytest.mark.parametrize(
    "name, system, expected",
    [
        ("lumi", "LUMI", "LUMI"),
        ("LUMI", "LUMI", "LUMI"),
        ("Leonardo", "leonardo", "leonardo"),
        ("lumi-gpu", "LUMI", "LUMI (lumi-gpu)"),
        ("mycluster", "Slurm cluster", "Slurm cluster (mycluster)"),
        ("[/red]", "[bold]X[/bold]", "[bold]X[/bold] ([/red])"),
    ],
)
def test_render_report_title_includes_name_only_when_it_differs_from_system(
    name: str, system: str, expected: str
) -> None:
    console = make_console()

    render_report(report(name=name, system=system, sections=[]), console, TODAY)

    assert title_line(output_of(console)) == expected


def test_render_report_title_is_a_bold_cyan_left_aligned_rule() -> None:
    console = RecordingConsole()

    render_report(report(name="lumi-gpu", sections=[]), console, TODAY)  # type: ignore[arg-type]

    rule, blank = console.renderables
    assert isinstance(rule, Rule)
    assert isinstance(rule.title, Text)
    assert rule.title.plain == "LUMI (lumi-gpu)"
    assert rule.title.style == "bold cyan"
    assert rule.align == "left"
    assert rule.style == "cyan"
    assert blank is None
    assert console.printed[-1] == ()


def test_render_report_cleans_the_title() -> None:
    data = report(name="my\x1b]0;x\x07cluster", system=f"Sys\ntem{HOSTILE}", sections=[])
    console = RecordingConsole()

    render_report(data, console, TODAY)  # type: ignore[arg-type]

    assert console.renderables[0].title.plain == "Sys temredlinkupcr (mycluster)"


def test_render_report_prints_all_sections_in_order() -> None:
    full = report(
        budgets=[lumi_budget()],
        storage=[StorageQuota("/users/jdoe", used_bytes=5 * GiB, limit_bytes=20 * GiB)],
        jobs=fixture_jobs(),
        warnings=["projectinfo: unknown line"],
    )
    console = RecordingConsole()

    render_report(full, console, TODAY)  # type: ignore[arg-type]

    printed = console.renderables
    assert [type(item) for item in printed] == [Rule, Table, Table, Table, Text, type(None)]
    assert [printed[1].title, printed[2].title, printed[3].title] == [
        "Compute allocations",
        "Storage",
        "Jobs: 2 running, 2 pending, 1 other",
    ]
    assert rows(printed[1]) == rows(compute_table(full.budgets, TODAY))
    assert rows(printed[2]) == rows(storage_table(full.storage))
    assert rows(printed[3]) == rows(jobs_table(full.jobs or []))
    assert printed[4].plain == "warning: projectinfo: unknown line"


def test_render_report_uses_the_given_day_for_the_end_column() -> None:
    console = RecordingConsole()

    render_report(report(budgets=[lumi_budget()], sections=["compute"]), console, dt.date(2027, 3, 1))  # type: ignore[arg-type]

    assert rows(console.renderables[1])[0][-1] == "2027-03-31 (30d)"


def test_render_report_dims_budgets_that_ended_before_the_given_day() -> None:
    console = RecordingConsole()

    render_report(report(budgets=[lumi_budget()], sections=["compute"]), console, dt.date(2027, 4, 1))  # type: ignore[arg-type]

    table = console.renderables[1]
    assert rows(table)[0][3:] == ("92,813 (97%)", "2027-03-31 (ended)")
    assert styles(table)[0][3:] == ("dim", "red")


def test_render_report_renders_full_report_text() -> None:
    full = report(
        budgets=[lumi_budget()],
        storage=[StorageQuota("/users/jdoe", used_bytes=5 * GiB, limit_bytes=20 * GiB)],
        jobs=fixture_jobs()[:1],
    )
    console = make_console()

    render_report(full, console, TODAY)
    out = output_of(console)

    assert title_line(out) == "LUMI"
    for expected in (
        "Compute allocations",
        "92,813 (97%)",
        "2027-03-31 (133d)",
        "Storage",
        "/users/jdoe",
        "Files / quota",
        "Jobs: 1 running, 0 pending",
        "train|model [v2]",
        "1:02:03 / 2-00:00:00",
    ):
        assert expected in out
    assert out.index("Compute allocations") < out.index("Storage") < out.index("Jobs: 1 running")
    assert "No " not in out
    assert "Note:" not in out
    assert "warning:" not in out and "error:" not in out
    assert out.endswith("\n\n")


@pytest.mark.parametrize(
    "sections, expected_titles",
    [
        ([], []),
        (["compute"], ["Compute allocations"]),
        (["storage"], ["Storage"]),
        (["jobs"], ["Jobs: 2 running, 2 pending, 1 other"]),
        (["compute", "jobs"], ["Compute allocations", "Jobs: 2 running, 2 pending, 1 other"]),
    ],
)
def test_render_report_shows_only_requested_sections(sections: List[str], expected_titles: List[str]) -> None:
    data = report(
        budgets=[lumi_budget()],
        storage=[StorageQuota("/users/jdoe", used_bytes=1)],
        jobs=fixture_jobs(),
        sections=sections,
    )
    console = RecordingConsole()

    render_report(data, console, TODAY)  # type: ignore[arg-type]

    tables = [item for item in console.renderables if isinstance(item, Table)]
    assert [table.title for table in tables] == expected_titles
    assert not any(isinstance(item, Text) for item in console.renderables)


def test_render_report_unrequested_empty_sections_print_no_placeholder() -> None:
    console = make_console()

    render_report(report(sections=["jobs"], jobs=[]), console, TODAY)
    out = output_of(console)

    assert "No jobs in the queue." in out
    assert "No compute allocations found." not in out
    assert "No storage quotas found." not in out


def test_render_report_empty_sections_without_errors_say_so() -> None:
    console = RecordingConsole()

    render_report(report(jobs=[]), console, TODAY)  # type: ignore[arg-type]

    messages = [item for item in console.renderables if isinstance(item, Text)]
    assert [(text.plain, text.style) for text in messages] == [
        ("No compute allocations found.", "dim"),
        ("No storage quotas found.", "dim"),
        ("No jobs in the queue.", "dim"),
    ]


def test_render_report_empty_sections_render_messages_in_order() -> None:
    console = make_console()

    render_report(report(jobs=[]), console, TODAY)

    assert output_of(console).splitlines()[1:] == [
        "No compute allocations found.",
        "No storage quotas found.",
        "No jobs in the queue.",
        "",
    ]


def test_render_report_jobs_not_collected_prints_nothing_for_jobs() -> None:
    console = RecordingConsole()

    render_report(report(jobs=None), console, TODAY)  # type: ignore[arg-type]

    plain = [item.plain for item in console.renderables if isinstance(item, Text)]
    assert plain == ["No compute allocations found.", "No storage quotas found."]


def test_render_report_failed_sections_print_no_placeholder_and_the_error_is_printed() -> None:
    failed = report(
        errors=["ssh: connect to host lumi.csc.fi port 22: Connection timed out"],
        failed_sections=["compute", "storage", "jobs"],
        jobs=None,
    )
    console = RecordingConsole()

    render_report(failed, console, TODAY)  # type: ignore[arg-type]

    printed = console.renderables
    assert [type(item) for item in printed] == [Rule, Text, type(None)]
    assert printed[1].plain == "error: ssh: connect to host lumi.csc.fi port 22: Connection timed out"
    assert printed[1].style == "bold red"


def test_render_report_failed_jobs_do_not_hide_the_other_placeholders() -> None:
    # Only the sections listed in ``failed_sections`` lack data; an error of
    # another kind (e.g. about jobs) must not hide "nothing found" messages.
    data = report(errors=["jobs: `squeue` timed out"], failed_sections=["jobs"], jobs=None)
    console = RecordingConsole()

    render_report(data, console, TODAY)  # type: ignore[arg-type]

    texts = [item for item in console.renderables if isinstance(item, Text)]
    assert [(text.plain, text.style) for text in texts] == [
        ("No compute allocations found.", "dim"),
        ("No storage quotas found.", "dim"),
        ("error: jobs: `squeue` timed out", "bold red"),
    ]


def test_render_report_errors_without_failed_sections_keep_the_placeholders() -> None:
    data = report(errors=["something went wrong"], jobs=[])
    console = RecordingConsole()

    render_report(data, console, TODAY)  # type: ignore[arg-type]

    plain = [item.plain for item in console.renderables if isinstance(item, Text)]
    assert plain == [
        "No compute allocations found.",
        "No storage quotas found.",
        "No jobs in the queue.",
        "error: something went wrong",
    ]


def test_render_report_compute_data_with_failed_storage_hides_storage_placeholder() -> None:
    partial = report(
        budgets=[lumi_budget()],
        storage=[],
        jobs=None,
        errors=["storage: `lfs quota` timed out"],
        failed_sections=["storage"],
    )
    console = RecordingConsole()

    render_report(partial, console, TODAY)  # type: ignore[arg-type]

    printed = console.renderables
    assert [type(item) for item in printed] == [Rule, Table, Text, type(None)]
    assert printed[1].title == "Compute allocations"
    assert printed[2].plain == "error: storage: `lfs quota` timed out"


def test_render_report_storage_data_with_failed_compute_hides_compute_placeholder() -> None:
    partial = report(
        budgets=[],
        storage=[StorageQuota("/users/jdoe", used_bytes=1)],
        jobs=None,
        errors=["compute: `lumi-allocations` failed (exit status 1)"],
        failed_sections=["compute"],
    )
    console = RecordingConsole()

    render_report(partial, console, TODAY)  # type: ignore[arg-type]

    printed = console.renderables
    assert [type(item) for item in printed] == [Rule, Table, Text, type(None)]
    assert printed[1].title == "Storage"
    assert printed[2].plain == "error: compute: `lumi-allocations` failed (exit status 1)"


@pytest.mark.parametrize(
    "failed_sections, expected",
    [
        (["compute"], ["No storage quotas found."]),
        (["storage"], ["No compute allocations found."]),
        (["compute", "storage"], []),
        ([], ["No compute allocations found.", "No storage quotas found."]),
    ],
)
def test_render_report_one_failed_section_keeps_the_other_placeholder(
    failed_sections: List[str], expected: List[str]
) -> None:
    data = report(sections=["compute", "storage"], failed_sections=failed_sections)
    console = make_console()

    render_report(data, console, TODAY)

    assert [line for line in output_of(console).splitlines()[1:] if line] == expected


def test_render_report_errors_do_not_hide_collected_data() -> None:
    partial = report(
        budgets=[lumi_budget()],
        storage=[StorageQuota("/users/jdoe", used_bytes=1)],
        jobs=None,
        errors=["jobs: `squeue` timed out"],
        failed_sections=["jobs"],
    )
    console = RecordingConsole()

    render_report(partial, console, TODAY)  # type: ignore[arg-type]

    printed = console.renderables
    assert [type(item) for item in printed] == [Rule, Table, Table, Text, type(None)]
    assert [printed[1].title, printed[2].title] == ["Compute allocations", "Storage"]
    assert printed[3].plain == "error: jobs: `squeue` timed out"


def test_render_report_empty_job_list_is_reported_even_with_errors_elsewhere() -> None:
    # ``jobs`` is ``None`` when the jobs could not be collected, so an empty
    # list means that squeue succeeded and the user has no jobs.
    partial = report(
        jobs=[],
        errors=["storage: `lfs quota` failed (exit status 1)"],
        failed_sections=["storage"],
    )
    console = RecordingConsole()

    render_report(partial, console, TODAY)  # type: ignore[arg-type]

    plain = [item.plain for item in console.renderables if isinstance(item, Text)]
    assert plain == [
        "No compute allocations found.",
        "No jobs in the queue.",
        "error: storage: `lfs quota` failed (exit status 1)",
    ]


def test_render_report_prints_warnings_before_errors() -> None:
    data = report(
        sections=[],
        warnings=["first warning", "second warning"],
        errors=["compute: boom", "storage: bang"],
    )
    console = RecordingConsole()

    render_report(data, console, TODAY)  # type: ignore[arg-type]

    texts = [item for item in console.renderables if isinstance(item, Text)]
    assert [(text.plain, text.style) for text in texts] == [
        ("warning: first warning", "yellow"),
        ("warning: second warning", "yellow"),
        ("error: compute: boom", "bold red"),
        ("error: storage: bang", "bold red"),
    ]


@pytest.mark.parametrize("markup", MARKUP_LIKE)
def test_render_report_prints_markup_like_warnings_and_errors_literally(markup: str) -> None:
    console = make_console()

    render_report(report(sections=[], warnings=[markup], errors=[markup]), console, TODAY)

    assert output_of(console).splitlines()[1:] == [f"warning: {markup}", f"error: {markup}", ""]


def test_render_report_cleans_warnings_and_errors() -> None:
    console = RecordingConsole()
    data = report(sections=[], warnings=[f"multi\nline\twarning{HOSTILE}"], errors=[f"\x1b[2Jerr\x00or{HOSTILE}"])

    render_report(data, console, TODAY)  # type: ignore[arg-type]

    texts = [item.plain for item in console.renderables if isinstance(item, Text)]
    assert texts == ["warning: multi line warningredlinkupcr", "error: errorredlinkupcr"]


def test_render_report_never_prints_escape_sequences_from_remote_strings() -> None:
    hostile_jobs = [
        Job(f"1{HOSTILE}", f"n{HOSTILE}", f"p{HOSTILE}", f"a{HOSTILE}", f"R{HOSTILE}", f"1{HOSTILE}", f"2{HOSTILE}", 1,
            f"r{HOSTILE}"),
        Job("2", "n", "p", "other-account", "PD"),
    ]
    data = report(
        name=f"name{HOSTILE}",
        system=f"system{HOSTILE}",
        budgets=[
            ComputeBudget(project=f"p{HOSTILE}", resource=f"r{HOSTILE}", unit=f"u{HOSTILE}", account=f"a{HOSTILE}",
                          total=10, used=1, note=f"n{HOSTILE}"),
            ComputeBudget(project="p2", resource="CPU", unit="core-h", note=f"other{HOSTILE}"),
        ],
        storage=[StorageQuota(f"/a{HOSTILE}", used_bytes=1, note=f"shared{HOSTILE}"),
                 StorageQuota(f"/b{HOSTILE}", note=f"shared{HOSTILE}")],
        jobs=hostile_jobs,
        warnings=[f"w{HOSTILE}"],
        errors=[f"e{HOSTILE}"],
    )
    console = make_console()

    render_report(data, console, TODAY)
    out = output_of(console)

    assert_terminal_safe(out)
    for expected in ("systemredlinkupcr (nameredlinkupcr)", "predlinkupcr (aredlinkupcr)",
                     "Note: sharedredlinkupcr", "warning: wredlinkupcr", "error: eredlinkupcr", "Account"):
        assert expected in out


def test_render_report_colours_in_a_truecolor_terminal() -> None:
    console = make_console(color=True)

    render_report(
        report(jobs=[], warnings=["careful"], errors=["boom"], failed_sections=["compute", "storage"]),
        console,
        TODAY,
    )
    out = output_of(console)

    assert "\x1b[1;36mLUMI\x1b[0m" in out
    assert "\x1b[2mNo jobs in the queue.\x1b[0m" in out
    assert "\x1b[33mwarning: careful\x1b[0m" in out
    assert "\x1b[1;31merror: boom\x1b[0m" in out
    assert "No compute allocations found." not in out


def test_render_report_colours_hostile_strings_without_their_own_sequences() -> None:
    console = make_console(color=True)

    render_report(report(sections=[], warnings=["\x1b[5mblink\x1b[0m"]), console, TODAY)
    out = output_of(console)

    assert "\x1b[33mwarning: blink\x1b[0m" in out
    assert "\x1b[5m" not in out


def test_render_report_tables_are_coloured_in_a_truecolor_terminal() -> None:
    console = make_console(color=True)
    data = report(
        budgets=[lumi_budget(end=dt.date(2026, 11, 25)), lumi_budget(project="old", end=dt.date(2026, 1, 1))],
        storage=[StorageQuota("/scratch/p", "p", used_bytes=99, limit_bytes=100)],
        jobs=[Job("1", state="RUNNING"), Job("2", state="PENDING")],
    )

    render_report(data, console, TODAY)
    out = output_of(console)

    # Right-aligned cells are padded inside their style.
    assert re.search(r"\x1b\[32m *92,813 \(97%\)\x1b\[0m", out)
    assert re.search(r"\x1b\[2m *92,813 \(97%\)\x1b\[0m", out)
    assert re.search(r"\x1b\[31m *2026-11-25 \(7d\)\x1b\[0m", out)
    assert re.search(r"\x1b\[31m *2026-01-01 \(ended\)\x1b\[0m", out)
    assert "\x1b[31m99%\x1b[0m" in out
    assert "\x1b[32mRUNNING\x1b[0m" in out
    assert "\x1b[33mPENDING\x1b[0m" in out


# -- render_reports ---------------------------------------------------------------


def test_render_reports_prints_each_report_as_it_arrives_and_returns_them_in_order() -> None:
    console = make_console()
    first = report(name="lumi", system="LUMI", sections=["jobs"], jobs=[])
    second = report(name="leonardo", system="Leonardo", sections=["jobs"], jobs=None, errors=["boom"])
    seen_before_second: List[str] = []

    def produce() -> Iterator[ClusterReport]:
        yield first
        seen_before_second.append(output_of(console))
        yield second

    rendered = render_reports(produce(), console, TODAY)

    assert isinstance(rendered, list)
    assert len(rendered) == 2
    assert rendered[0] is first and rendered[1] is second
    assert title_line(seen_before_second[0]) == "LUMI"
    assert "Leonardo" not in seen_before_second[0]
    out = output_of(console)
    assert out.startswith(seen_before_second[0])
    second_part = out[len(seen_before_second[0]):]
    assert title_line(second_part) == "Leonardo"
    assert second_part.splitlines()[1:] == ["error: boom", ""]


def test_render_reports_with_no_reports_prints_nothing() -> None:
    console = make_console()

    assert render_reports(iter([]), console, TODAY) == []
    assert output_of(console) == ""


# -- reports_to_json --------------------------------------------------------------


def test_reports_to_json_document_structure() -> None:
    lumi = report(
        gpus="4× AMD MI250X 128 GB (2 GCDs each)",
        budgets=[lumi_budget(start=dt.date(2026, 4, 1), note="LUMI-G", gpu_hours_per_unit=1.0)],
        storage=[StorageQuota("/users/jdoe", None, used_bytes=5 * GiB, limit_bytes=20 * GiB, files_used=10,
                              files_limit=100)],
        jobs=[fixture_jobs()[3]],
        warnings=["careful"],
        collected_at=NOW,
    )
    leonardo = report(
        name="leo",
        system="Leonardo",
        sections=["compute", "storage"],
        budgets=[ComputeBudget(project="euhpc_d00_001", resource="Booster", unit="core-h", account="EUHPC_D00_001_0",
                               month_total=200, month_used=50, end=dt.date(2027, 6, 30), gpu_hours_per_unit=0.125)],
        errors=["storage: boom"],
        failed_sections=["storage"],
    )

    text = reports_to_json([lumi, leonardo], generated_at=NOW.replace(microsecond=123_456))
    document = json.loads(text)

    assert list(document) == ["generated_at", "clusters"]
    assert document["generated_at"] == "2026-11-18T20:30:00"
    assert document["clusters"] == [
        {
            "name": "lumi",
            "system": "LUMI",
            "gpus": "4× AMD MI250X 128 GB (2 GCDs each)",
            "collected_at": "2026-11-18T20:30:00",
            "sections": ["compute", "storage", "jobs"],
            "failed_sections": [],
            "budgets": [
                {
                    "project": "project_465000123",
                    "account": None,
                    "resource": "GPU",
                    "unit": "GPU-h",
                    "total": 96_000,
                    "used": 3_187,
                    "remaining": 92_813,
                    "remaining_fraction": 92_813 / 96_000,
                    "month_total": None,
                    "month_used": None,
                    "month_remaining": None,
                    "month_remaining_fraction": None,
                    "start": "2026-04-01",
                    "end": "2027-03-31",
                    # Days left as of the collection date.
                    "days_left": 133,
                    "note": "LUMI-G",
                    "gpu_hours_per_unit": 1.0,
                    "gpu_hours_total": 96_000,
                    "gpu_hours_remaining": 92_813,
                    "month_gpu_hours_total": None,
                    "month_gpu_hours_remaining": None,
                }
            ],
            "storage": [
                {
                    "name": "/users/jdoe",
                    "project": None,
                    "used_bytes": 5 * GiB,
                    "limit_bytes": 20 * GiB,
                    "free_bytes": 15 * GiB,
                    "used_fraction": 0.25,
                    "files_used": 10,
                    "files_limit": 100,
                    "files_fraction": 0.1,
                    "note": None,
                }
            ],
            "jobs": [
                {
                    "job_id": "1234569_[1-10]",
                    "name": "sweep",
                    "partition": "small-g",
                    "account": "project_465000999",
                    "state": "PENDING",
                    "elapsed": "0:00",
                    "time_limit": "30:00",
                    "nodes": None,
                    "reason": "(AssocGrpGRESMinutes)",
                }
            ],
            "errors": [],
            "warnings": ["careful"],
        },
        {
            "name": "leo",
            "system": "Leonardo",
            "gpus": None,
            "collected_at": None,
            "sections": ["compute", "storage"],
            "failed_sections": ["storage"],
            "budgets": [
                {
                    "project": "euhpc_d00_001",
                    "account": "EUHPC_D00_001_0",
                    "resource": "Booster",
                    "unit": "core-h",
                    "total": None,
                    "used": None,
                    "remaining": None,
                    "remaining_fraction": None,
                    "month_total": 200,
                    "month_used": 50,
                    "month_remaining": 150,
                    "month_remaining_fraction": 0.75,
                    "start": None,
                    "end": "2027-06-30",
                    # Without a collection time there is no reference day.
                    "days_left": None,
                    "note": None,
                    "gpu_hours_per_unit": 0.125,
                    "gpu_hours_total": None,
                    "gpu_hours_remaining": None,
                    "month_gpu_hours_total": 25.0,
                    "month_gpu_hours_remaining": 18.75,
                }
            ],
            "storage": [],
            "jobs": None,
            "errors": ["storage: boom"],
            "warnings": [],
        },
    ]
    assert document["clusters"] == [lumi.to_dict(), leonardo.to_dict()]


def test_reports_to_json_days_left_use_the_collection_date_of_each_report() -> None:
    budget = lumi_budget(end=dt.date(2026, 12, 31))
    early = report(name="a", budgets=[budget], collected_at=dt.datetime(2026, 12, 1, 23, 59, 59))
    late = report(name="b", budgets=[budget], collected_at=dt.datetime(2027, 1, 2, 0, 0, 1))

    document = json.loads(reports_to_json([early, late], generated_at=NOW))

    assert [cluster["budgets"][0]["days_left"] for cluster in document["clusters"]] == [30, -2]
    assert [cluster["collected_at"] for cluster in document["clusters"]] == [
        "2026-12-01T23:59:59",
        "2027-01-02T00:00:01",
    ]


def test_reports_to_json_keeps_remote_strings_unmodified() -> None:
    # Escaping is the job of the JSON encoder; the terminal cleaning does not apply.
    data = report(warnings=["\x1b[31mred\x1b[0m\nnext"], sections=[])

    text = reports_to_json([data], generated_at=NOW)

    assert "\x1b" not in text
    assert "\\u001b[31mred\\u001b[0m\\nnext" in text
    assert json.loads(text)["clusters"][0]["warnings"] == ["\x1b[31mred\x1b[0m\nnext"]


def test_reports_to_json_is_indented_by_two_spaces() -> None:
    text = reports_to_json([], generated_at=NOW)

    assert text == '{\n  "generated_at": "2026-11-18T20:30:00",\n  "clusters": []\n}'


def test_reports_to_json_keeps_timezone_and_drops_sub_second_precision() -> None:
    generated_at = dt.datetime(2026, 11, 18, 20, 30, 5, 999_999, tzinfo=dt.timezone(dt.timedelta(hours=1)))

    document = json.loads(reports_to_json([], generated_at=generated_at))

    assert document["generated_at"] == "2026-11-18T20:30:05+01:00"


def test_reports_to_json_accepts_any_iterable_and_distinguishes_no_jobs_from_uncollected() -> None:
    reports = (report(name=name, jobs=jobs) for name, jobs in (("a", []), ("b", None)))

    document = json.loads(reports_to_json(reports, generated_at=NOW))

    assert [cluster["name"] for cluster in document["clusters"]] == ["a", "b"]
    assert [cluster["jobs"] for cluster in document["clusters"]] == [[], None]


def test_reports_to_json_escapes_non_ascii_and_round_trips() -> None:
    data = report(warnings=["quota über 90 %"])

    text = reports_to_json([data], generated_at=NOW)

    assert "\\u00fcber" in text
    assert json.loads(text)["clusters"][0]["warnings"] == ["quota über 90 %"]
