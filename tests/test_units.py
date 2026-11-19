"""Tests for :mod:`hpcmon.units`: parsing and formatting of numbers, sizes and dates.

Expected values are written out as literals (computed by hand from the unit
definitions) rather than recomputed with the module's own formulas.
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pytest

from hpcmon import units
from hpcmon.units import (
    format_amount,
    format_bytes,
    format_count,
    format_percent,
    is_missing,
    parse_count,
    parse_date,
    parse_int,
    parse_number,
    parse_size,
)
from tests.conftest import TODAY, read_fixture

KiB = 1024
MiB = 1024**2
GiB = 1024**3
TiB = 1024**4
PiB = 1024**5
EiB = 1024**6

#: Every spelling of a missing value used by the cluster tools.
MISSING_SPELLINGS = ["", "-", "--", "---", "n/a", "N/A", "na", "NA", "none", "None", "null", "unlimited"]


# --------------------------------------------------------------------------- #
# is_missing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("text", [None, "   ", "\t\n", " N/A ", "  --  "] + MISSING_SPELLINGS)
def test_is_missing_recognises_missing_spellings_and_whitespace(text):
    assert is_missing(text) is True


@pytest.mark.parametrize("text", ["0", "x", "NONE", "Unlimited", "NULL", "-1", "n.a.", "-0"])
def test_is_missing_is_case_sensitive_and_rejects_values(text):
    assert is_missing(text) is False


@pytest.mark.parametrize("text", ["----", "-----", " ------ ", "-" * 40, "\t--------\n"])
def test_is_missing_treats_any_run_of_dashes_as_missing(text):
    assert is_missing(text) is True


@pytest.mark.parametrize("text", ["- -", "--1", "-x-", "-- --", "—"])
def test_is_missing_dashes_mixed_with_other_characters_are_not_missing(text):
    assert is_missing(text) is False


@pytest.mark.parametrize("text", ["----", "------"])
def test_runs_of_dashes_are_missing_for_every_parser(text):
    assert parse_number(text) is None
    assert parse_int(text) is None
    assert parse_size(text) is None
    assert parse_count(text) is None
    assert parse_date(text) is None


def test_missing_constant_contains_exactly_the_documented_spellings():
    assert units.MISSING == frozenset(MISSING_SPELLINGS)


# --------------------------------------------------------------------------- #
# parse_number / parse_int
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, expected",
    [
        ("42", 42.0),
        (" 42 ", 42.0),
        ("1,234.5", 1234.5),
        ("1,000,000", 1000000.0),
        ("20.5%", 20.5),
        ("12 %", 12.0),
        (" 8.5% ", 8.5),
        ("-3.25", -3.25),
        ("+7", 7.0),
        ("0.000", 0.0),
        (".5", 0.5),
    ],
)
def test_parse_number_parses_decimals_with_commas_and_percent(text, expected):
    value = parse_number(text)
    assert isinstance(value, float)
    assert value == expected


@pytest.mark.parametrize("text", [None, " N/A ", "  "] + MISSING_SPELLINGS)
def test_parse_number_returns_none_for_missing_values(text):
    assert parse_number(text) is None


@pytest.mark.parametrize("text", ["abc", "%", "--%", "12abc", "1.2.3", "NONE", "Unlimited", "5 GB"])
def test_parse_number_raises_value_error_for_garbage(text):
    with pytest.raises(ValueError):
        parse_number(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("4", 4),
        ("1,234", 1234),
        ("3.7", 4),
        ("2.4", 2),
        ("-1.6", -2),
        ("50%", 50),
        (" 16 ", 16),
    ],
)
def test_parse_int_rounds_to_nearest_integer(text, expected):
    value = parse_int(text)
    assert type(value) is int
    assert value == expected


@pytest.mark.parametrize("text", [None, "-", "", "N/A", "unlimited"])
def test_parse_int_returns_none_for_missing_values(text):
    assert parse_int(text) is None


@pytest.mark.parametrize("text", ["abc", "(Priority)", "nid[001-004]", "nan"])
def test_parse_int_raises_value_error_for_non_numbers(text):
    with pytest.raises(ValueError):
        parse_int(text)


@pytest.mark.parametrize("text", ["inf", "-inf", "1e999"])
def test_parse_int_raises_value_error_for_infinite_values(text):
    with pytest.raises(ValueError):
        parse_int(text)


@pytest.mark.parametrize(
    "text",
    ["inf", "-inf", "+inf", "Infinity", " INF ", "nan", "NaN", "-nan", "nan%", "1e999", "-1e999", "1,000e999"],
)
def test_parse_number_raises_value_error_for_non_finite_values(text):
    with pytest.raises(ValueError) as excinfo:
        parse_number(text)
    assert str(excinfo.value) == f"not a finite number: {text!r}"


@pytest.mark.parametrize("text, expected", [("1e3", 1000.0), ("1.5E2", 150.0), ("1e308", 1e308), ("-2e-3", -0.002)])
def test_parse_number_accepts_finite_exponents(text, expected):
    assert parse_number(text) == expected


# --------------------------------------------------------------------------- #
# parse_size
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, expected",
    [
        # Upper case prefixes are binary, whatever the suffix.
        ("1K", KiB),
        ("1KB", KiB),
        ("1Kb", KiB),
        ("1KiB", KiB),
        ("1Kib", KiB),
        ("1Ki", KiB),
        ("1M", MiB),
        ("2MiB", 2 * MiB),
        ("1G", GiB),
        ("1GB", GiB),
        ("1T", TiB),
        ("1TiB", TiB),
        ("1P", PiB),
        ("1PB", PiB),
        ("1E", EiB),
        ("1EiB", EiB),
        # Lower case prefixes with an "i" are binary as well.
        ("1ki", KiB),
        ("1kiB", KiB),
        ("1kib", KiB),
        ("3gib", 3 * GiB),
        ("1ti", TiB),
        # Lower case prefixes without "i" are decimal.
        ("1k", 1000),
        ("1kB", 1000),
        ("1kb", 1000),
        ("1m", 10**6),
        ("1mB", 10**6),
        ("1g", 10**9),
        ("1t", 10**12),
        ("1p", 10**15),
        ("1e", 10**18),
        ("2.5kB", 2500),
        # Plain bytes.
        ("512B", 512),
        ("512b", 512),
        ("512 B", 512),
    ],
)
def test_parse_size_unit_prefixes(text, expected):
    value = parse_size(text)
    assert type(value) is int
    assert value == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("5.2G", 5583457485),
        (".5K", 512),
        ("5.K", 5120),
        ("+5 K", 5120),
        ("-1K", -KiB),
        ("-0.5M", -524288),
        ("1,024K", MiB),
        ("  7 MiB  ", 7 * MiB),
        ("1\tG", GiB),
        ("0K", 0),
        ("0.0001K", 0),
        ("1.0005K", 1025),
    ],
)
def test_parse_size_number_formats_signs_whitespace_and_rounding(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        # LUMI ``lfs quota`` style (lumi/workspaces.txt).
        ("20KiB", 20 * KiB),
        ("4KiB", 4 * KiB),
        ("100GiB", 100 * GiB),
        ("110GiB", 110 * GiB),
        ("20TiB", 20 * TiB),
        ("20.0010TiB", 21991332067148),
        ("1.00098TiB", 1100589149171),
        # Leonardo ``cindata`` style (leonardo/cindata.txt).
        ("87G", 87 * GiB),
        ("2T", 2 * TiB),
        ("4K", 4 * KiB),
        ("3.4G", 3650722202),
        ("60G", 60 * GiB),
        # JUPITER ``jutil`` style (jupiter/userdata.json).
        ("1187264KB", 1215758336),
        ("31457280KB", 32212254720),
    ],
)
def test_parse_size_real_tool_values(text, expected):
    assert parse_size(text) == expected


def test_parse_size_values_taken_from_fixtures_exist_verbatim():
    """The strings used above really occur in the captured tool outputs."""
    workspaces = read_fixture("lumi/workspaces.txt")
    for token in ("20KiB", "100GiB", "20TiB/20.0010TiB", "1.00098TiB", "4.00100M", "200K"):
        assert token in workspaces
    cindata = read_fixture("leonardo/cindata.txt")
    for token in (" 87G ", " 2T ", " 4K ", " 3.4G ", " 60G "):
        assert token in cindata


def test_parse_size_jupiter_entries_with_kb_default_unit():
    entries = json.loads(read_fixture("jupiter/userdata.json"))
    home = next(e for e in entries if e["filesystem"] == "exa_home")
    project = next(e for e in entries if e["filesystem"] == "exa_project1")

    assert parse_size(home["data-usage"], default_unit="KB") == 1187264 * KiB
    assert parse_size(home["data-soft-limit"], default_unit="KB") == 31457280 * KiB
    assert parse_size(project["data-usage"], default_unit="KB") == 198342876 * KiB
    # "-" means no limit.
    assert parse_size(project["data-soft-limit"], default_unit="KB") is None
    assert parse_count(home["inode-usage"]) == 4128
    assert parse_count(home["inode-soft-limit"]) == 120000
    assert parse_count(project["inode-soft-limit"]) is None


@pytest.mark.parametrize(
    "text, default_unit, expected",
    [
        ("512", "B", 512),
        ("1187264", "KB", 1215758336),
        ("3", "GiB", 3 * GiB),
        ("2", "kB", 2000),
        ("2", "b", 2),
        ("1.5", "M", 1572864),
        # An explicit unit always wins over the default.
        ("1M", "KB", MiB),
        ("7B", "GiB", 7),
    ],
)
def test_parse_size_bare_numbers_use_default_unit(text, default_unit, expected):
    assert parse_size(text, default_unit=default_unit) == expected


def test_parse_size_default_unit_is_bytes():
    assert parse_size("4096") == 4096


@pytest.mark.parametrize("text", [None, "-", "--", "", "   ", "N/A", "unlimited", "none"])
def test_parse_size_returns_none_for_missing_values(text):
    assert parse_size(text) is None
    assert parse_size(text, default_unit="KB") is None


@pytest.mark.parametrize("text", ["abc", "K", "GiB", "1.2.3G", "1e3", "G5", "12 34", "5 G B", "5G!", "--%", "Unlimited"])
def test_parse_size_rejects_unparseable_sizes(text):
    with pytest.raises(ValueError) as excinfo:
        parse_size(text)
    assert str(excinfo.value) == f"invalid size: {text!r}"


@pytest.mark.parametrize("text", ["5X", "5KK", "5Kbytes", "5KIB", "5KIb", "5bB", "5 iB", "5 KiBB", "5bytes", "5BB", "5z"])
def test_parse_size_rejects_unknown_units(text):
    with pytest.raises(ValueError) as excinfo:
        parse_size(text)
    assert str(excinfo.value) == f"invalid size unit in {text!r}"


@pytest.mark.parametrize("default_unit", ["Q", "KBytes", "x"])
def test_parse_size_rejects_invalid_default_unit_for_bare_numbers(default_unit):
    with pytest.raises(ValueError, match=re.escape("invalid size unit in '5'")):
        parse_size("5", default_unit=default_unit)


@pytest.mark.parametrize("text", ["1" + "0" * 400, "9" * 300 + "E"])
def test_parse_size_overflowing_sizes_raise_value_error(text):
    with pytest.raises(ValueError):
        parse_size(text)


def test_parse_size_error_message_keeps_original_text():
    with pytest.raises(ValueError) as excinfo:
        parse_size("  1,0 XB ")
    assert str(excinfo.value) == "invalid size unit in '  1,0 XB '"


# --------------------------------------------------------------------------- #
# parse_count
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, expected",
    [
        ("4512", 4512),
        ("0", 0),
        ("1,000", 1000),
        (" 80000 ", 80000),
        ("+5", 5),
        ("12.6", 13),
        ("100K", 100000),
        ("100k", 100000),
        ("101K", 101000),
        ("2M", 2000000),
        ("2.00100M", 2001000),
        ("1.00100M", 1001000),
        ("1m", 1000000),
        ("2G", 2 * 10**9),
        ("1T", 10**12),
        ("1P", 10**15),
        ("1E", 10**18),
        (".5K", 500),
        ("-2K", -2000),
        ("1.5 K", 1500),
    ],
)
def test_parse_count_uses_decimal_prefixes(text, expected):
    value = parse_count(text)
    assert type(value) is int
    assert value == expected


@pytest.mark.parametrize("text", [None, "-", "", "  ", "N/A", "unlimited"])
def test_parse_count_returns_none_for_missing_values(text):
    assert parse_count(text) is None


@pytest.mark.parametrize("text", ["abc", "K", "1.2.3", "1e3", "5 K B", "Unlimited"])
def test_parse_count_rejects_unparseable_counts(text):
    with pytest.raises(ValueError) as excinfo:
        parse_count(text)
    assert str(excinfo.value) == f"invalid count: {text!r}"


@pytest.mark.parametrize("text", ["1KB", "1Ki", "1B", "1b", "1X", "1kb", "1KK", "1KiB"])
def test_parse_count_rejects_size_units(text):
    with pytest.raises(ValueError) as excinfo:
        parse_count(text)
    assert str(excinfo.value) == f"invalid count unit in {text!r}"


@pytest.mark.parametrize("text", ["1" + "0" * 400, "9" * 300 + "E"])
def test_parse_count_overflowing_counts_raise_value_error(text):
    with pytest.raises(ValueError):
        parse_count(text)


# --------------------------------------------------------------------------- #
# parse_date
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "2026-11-18",
        "20261118",
        "18.11.2026",
        "2026/11/18",
        "  2026-11-18  ",
        "\t18.11.2026\n",
        # Timestamps: the time is ignored.
        "2026-11-18 17:20:01",
        "2026-11-18 00:00:00",
        "2026-11-18 23:59:59",
        "2026-11-18T00:00:00",
        "2026-11-18T23:59:59+02:00",
        "2026-11-18T12:00:00.123456Z",
        " 2026-11-18 17:20 ",
    ],
)
def test_parse_date_accepts_all_supported_formats(text):
    value = parse_date(text)
    assert type(value) is dt.date
    assert value == TODAY


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2027-05-19", dt.date(2027, 5, 19)),
        ("20280229", dt.date(2028, 2, 29)),
        ("01.03.2026", dt.date(2026, 3, 1)),
        ("2025/12/31", dt.date(2025, 12, 31)),
    ],
)
def test_parse_date_parses_day_and_month_in_the_right_order(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize("text", [None, "-", "--", "", " ", "N/A", "none", "null"])
def test_parse_date_returns_none_for_missing_values(text):
    assert parse_date(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "2026-13-01",
        "31.02.2026",
        "20270229",
        "18/11/2026",
        "11/18/2026",
        "tomorrow",
        "NONE",
        # Timestamps with an impossible date, or that are not timestamps.
        "2026-13-01 10:00:00",
        "2027-02-29T00:00:00",
        "2026-11-18T",
        "2026-11-18 T",
        "2026-11-18 noon",
        "2026-11-18x17:20:01",
        "18.11.2026 17:20:01",
        "20261118T172001",
    ],
)
def test_parse_date_rejects_unknown_formats_and_impossible_dates(text):
    with pytest.raises(ValueError) as excinfo:
        parse_date(text)
    assert str(excinfo.value) == f"invalid date: {text!r}"


# --------------------------------------------------------------------------- #
# format_bytes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "-"),
        (0, "0 B"),
        (1, "1 B"),
        (1023, "1023 B"),
        (1024, "1.0 KiB"),
        (1536, "1.5 KiB"),
        (1048514, "1023.9 KiB"),  # 1023.94 KiB: just below the rollover threshold
        (1048525, "1.0 MiB"),  # 1023.95 KiB: rounds up, so switch unit first
        (MiB, "1.0 MiB"),
        (1023 * MiB, "1023.0 MiB"),
        (1215758336, "1.1 GiB"),
        (5 * GiB, "5.0 GiB"),
        (64 * GiB, "64.0 GiB"),
        (1099500890357, "1.0 TiB"),  # 1023.99 GiB, not "1024.0 GiB"
        (TiB, "1.0 TiB"),
        (50 * TiB, "50.0 TiB"),
        (PiB, "1.0 PiB"),
        (EiB, "1024.0 PiB"),  # PiB is the largest unit
        (2048 * PiB, "2048.0 PiB"),
    ],
)
def test_format_bytes_uses_binary_prefixes(value, expected):
    assert format_bytes(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        (-1, "-1 B"),
        (-1023, "-1023 B"),
        (-1536, "-1.5 KiB"),
        (-3 * TiB, "-3.0 TiB"),
        (-1099500890357, "-1.0 TiB"),
        (-EiB, "-1024.0 PiB"),
    ],
)
def test_format_bytes_negative_values_keep_their_sign(value, expected):
    assert format_bytes(value) == expected


def test_format_bytes_never_prints_1024_of_a_unit():
    for value in range(1047552 - 64, 1048576 + 64):  # around 1023 KiB .. 1 MiB
        assert not format_bytes(value).startswith("1024.0 ")


# --------------------------------------------------------------------------- #
# format_amount
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "-"),
        (0, "0"),
        (0.0, "0"),
        (5, "5"),
        (12.0, "12"),
        (12.34, "12.3"),
        (99.94, "99.9"),
        (0.04, "0"),
        (99.96, "100"),
        (2.04, "2"),
        (100, "100"),
        (100.4, "100"),
        (1500.0, "1,500"),
        (1234.6, "1,235"),
        (1234567.0, "1,234,567"),
        (2500000, "2,500,000"),
        (-12.34, "-12.3"),
        (-150.7, "-151"),
        (-1234.6, "-1,235"),
        (-7.0, "-7"),
    ],
)
def test_format_amount(value, expected):
    assert format_amount(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        # Values that round to a whole number are shown without decimals ...
        (0.96, "1"),
        (1.04, "1"),
        (9.96, "10"),
        (12.96, "13"),
        (-0.96, "-1"),
        (-99.96, "-100"),
        # ... and values that round to 100 or more never get a decimal.
        (99.95001, "100"),
        (999.96, "1,000"),
        # Others keep one decimal.
        (0.06, "0.1"),
        (0.5, "0.5"),
        (99.949, "99.9"),
        (-0.4, "-0.4"),
    ],
)
def test_format_amount_decides_decimals_after_rounding(value, expected):
    assert format_amount(value) == expected


@pytest.mark.parametrize("value", [-0.04, -0.0])
def test_format_amount_never_prints_negative_zero(value):
    assert format_amount(value) == "0"


# --------------------------------------------------------------------------- #
# format_count
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "-"),
        (0, "0"),
        (5, "5"),
        (999, "999"),
        (1000, "1k"),
        (1500, "1.5k"),
        (4512, "4.5k"),
        (80000, "80k"),
        (100000, "100k"),
        (187342, "187.3k"),
        (999949, "999.9k"),
        (999950, "1M"),  # rolls over instead of printing "1000.0k"
        (1000000, "1M"),
        (1234567, "1.2M"),
        (2001000, "2M"),
        (999950000, "1G"),
        (1500000000, "1.5G"),
        (5000000000000, "5000G"),  # G is the largest unit
        (2500000000000000, "2500000G"),
    ],
)
def test_format_count_is_compact_with_decimal_prefixes(value, expected):
    assert format_count(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [(-1, "-1"), (-999, "-999"), (-1000, "-1k"), (-1500, "-1.5k"), (-1000000, "-1M"), (-2500000000, "-2.5G")],
)
def test_format_count_negative_values(value, expected):
    assert format_count(value) == expected


def test_format_count_never_prints_trailing_zero_or_1000_of_a_unit():
    for value in range(999_000, 1_001_000, 7):
        text = format_count(value)
        assert not text.startswith("1000")
        assert ".0" not in text


# --------------------------------------------------------------------------- #
# format_percent
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "fraction, expected",
    [
        (None, "-"),
        (0, "0.0%"),
        (0.0, "0.0%"),
        (0.00049, "0.0%"),
        (0.205, "20.5%"),
        (0.5, "50.0%"),
        (1, "100.0%"),
        (1.234, "123.4%"),
        (-0.1, "-10.0%"),
    ],
)
def test_format_percent(fraction, expected):
    assert format_percent(fraction) == expected
