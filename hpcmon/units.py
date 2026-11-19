"""Parsing and formatting of numbers, sizes and dates."""

from __future__ import annotations

import datetime as dt
import math
import re
from typing import Optional

#: Strings used by the various tools to denote a missing value.
MISSING = frozenset({"", "-", "--", "---", "n/a", "N/A", "na", "NA", "none", "None", "null", "unlimited"})

_SIZE_RE = re.compile(r"^(?P<num>[+-]?\d+(?:\.\d*)?|[+-]?\.\d+)\s*(?P<unit>[A-Za-z]*)$")
_PREFIXES = "kmgtpe"


def is_missing(text: Optional[str]) -> bool:
    """Whether ``text`` denotes a missing value (including any run of dashes)."""
    if text is None:
        return True
    stripped = text.strip()
    return stripped in MISSING or not stripped.strip("-")


def parse_number(text: Optional[str]) -> Optional[float]:
    """Parse a decimal number such as ``"1,234.5"`` or ``"20.5%"``.

    :return: The number, or ``None`` for missing values.
    :raises ValueError: if ``text`` is neither a number nor a missing value.
    """
    if is_missing(text):
        return None
    assert text is not None
    cleaned = text.strip().replace(",", "").rstrip("%").strip()
    value = float(cleaned)
    if not math.isfinite(value):
        raise ValueError(f"not a finite number: {text!r}")
    return value


def parse_int(text: Optional[str]) -> Optional[int]:
    """Like :func:`parse_number`, but rounds the result to an integer."""
    value = parse_number(text)
    return None if value is None else int(round(value))


def parse_size(text: Optional[str], default_unit: str = "B") -> Optional[int]:
    """Parse a size such as ``"5.2G"``, ``"1187264KB"``, ``"50 TiB"`` to bytes.

    Upper case prefixes (``K``, ``KB``, ``KiB``, ``M``...) are binary (powers of
    1024), as used by most HPC tools (``lfs quota``, ``jutil``, ``cindata``).
    A lower case prefix without ``i`` (``k``, ``kB``, ``m``...) is decimal
    (powers of 1000), following the ``cindata`` convention.  ``default_unit``
    is applied when ``text`` is a bare number.

    :return: The size in bytes, or ``None`` for missing values.
    :raises ValueError: if ``text`` cannot be parsed.
    """
    if is_missing(text):
        return None
    assert text is not None
    match = _SIZE_RE.match(text.strip().replace(",", ""))
    if not match:
        raise ValueError(f"invalid size: {text!r}")
    number = float(match.group("num"))
    unit = match.group("unit") or default_unit
    return _finite_int(number * _size_multiplier(unit, text), text)


def _finite_int(value: float, text: str) -> int:
    """Round ``value`` to an integer, rejecting values too large to represent."""
    if not math.isfinite(value):
        raise ValueError(f"number too large: {text!r}")
    return int(round(value))


def _size_multiplier(unit: str, text: str) -> int:
    if unit in ("B", "b"):
        return 1
    prefix = unit[0]
    rest = unit[1:]
    if prefix.lower() not in _PREFIXES or rest not in ("", "B", "b", "iB", "ib", "i"):
        raise ValueError(f"invalid size unit in {text!r}")
    power = _PREFIXES.index(prefix.lower()) + 1
    decimal = prefix.islower() and not rest.startswith("i")
    return (1000 if decimal else 1024) ** power


def parse_count(text: Optional[str]) -> Optional[int]:
    """Parse a count such as ``"100K"``, ``"2.00100M"`` or ``"4512"``.

    Prefixes are decimal (``K`` = 1000) as used for file counts.

    :return: The count, or ``None`` for missing values.
    :raises ValueError: if ``text`` cannot be parsed.
    """
    if is_missing(text):
        return None
    assert text is not None
    match = _SIZE_RE.match(text.strip().replace(",", ""))
    if not match:
        raise ValueError(f"invalid count: {text!r}")
    number = float(match.group("num"))
    unit = match.group("unit")
    if not unit:
        return _finite_int(number, text)
    if len(unit) != 1 or unit.lower() not in _PREFIXES:
        raise ValueError(f"invalid count unit in {text!r}")
    return _finite_int(number * 1000 ** (_PREFIXES.index(unit.lower()) + 1), text)


def parse_date(text: Optional[str]) -> Optional[dt.date]:
    """Parse a date in ``YYYY-MM-DD``, ``YYYYMMDD`` or ``DD.MM.YYYY`` format.

    :return: The date, or ``None`` for missing values.
    :raises ValueError: if ``text`` is not a date in a known format.
    """
    if is_missing(text):
        return None
    assert text is not None
    cleaned = text.strip()
    # Ignore the time of timestamps such as "2027-05-31 23:59:59".
    match = re.match(r"^(\d{4}-\d{2}-\d{2})[ T]\d", cleaned)
    if match:
        cleaned = match.group(1)
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%d.%m.%Y", "%Y/%m/%d"):
        try:
            return dt.datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"invalid date: {text!r}")


def format_bytes(value: Optional[int]) -> str:
    """Format a size in bytes using binary prefixes, e.g. ``"1.5 TiB"``."""
    if value is None:
        return "-"
    sign = "-" if value < 0 else ""
    amount = float(abs(value))
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    index = 0
    # Switch to the next unit before rounding could print e.g. "1024.0".
    while amount >= 1023.95 and index < len(units) - 1:
        amount /= 1024
        index += 1
    unit = units[index]
    if unit == "B":
        return f"{sign}{int(amount)} B"
    return f"{sign}{amount:.1f} {unit}"


def format_amount(value: Optional[float]) -> str:
    """Format a compute amount with thousands separators."""
    if value is None:
        return "-"
    if abs(round(value, 1)) >= 100 or float(round(value, 1)).is_integer():
        text = f"{value:,.0f}"
    else:
        text = f"{value:,.1f}"
    # Values that round to zero must not be shown as "-0".
    return text.lstrip("-") if not text.strip("-0.,") else text


def format_count(value: Optional[int]) -> str:
    """Format a count compactly, e.g. ``"1.2M"``."""
    if value is None:
        return "-"
    amount = float(value)
    units = ("", "k", "M", "G")
    index = 0
    while abs(amount) >= 999.95 and index < len(units) - 1:
        amount /= 1000
        index += 1
    unit = units[index]
    if unit == "":
        return f"{int(amount)}"
    text = f"{amount:.1f}"
    return f"{text[:-2] if text.endswith('.0') else text}{unit}"


def format_percent(fraction: Optional[float]) -> str:
    """Format a fraction (0-1) as a percentage."""
    if fraction is None:
        return "-"
    return f"{fraction * 100:.1f}%"
