"""Timestamp utilities for Aeon-V1.

Strategy
--------
- All JSON records store timestamps as timezone-aware UTC ISO-8601 strings:
    2026-04-28T21:07:13.050967+00:00
- Human-readable Markdown and CLI output converts to the configured local
  timezone (default: America/New_York) using zoneinfo (stdlib, Python 3.9+).

Requires the system timezone database or the `tzdata` PyPI package.
"""
from datetime import datetime
try:  # Python 3.11+
    from datetime import UTC
except ImportError:  # Python 3.10
    from datetime import timezone as _timezone
    UTC = _timezone.utc
from zoneinfo import ZoneInfo


def utc_now() -> datetime:
    """Return the current moment as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with +00:00 offset.

    Example: '2026-04-28T21:07:13.050967+00:00'
    """
    return datetime.now(UTC).isoformat()


def local_time_string(
    dt_or_iso: "str | datetime",
    timezone_name: str = "America/New_York",
) -> str:
    """Convert a UTC datetime or ISO string to a human-readable local string.

    Example: '2026-04-28 5:07 PM EDT'

    Args:
        dt_or_iso:     A timezone-aware datetime or an ISO-8601 string.
                       Naive datetimes are assumed to be UTC.
        timezone_name: IANA timezone name, e.g. 'America/New_York'.

    Returns:
        A string like '2026-04-28 5:07 PM EDT'.
    """
    if isinstance(dt_or_iso, str):
        dt = datetime.fromisoformat(dt_or_iso)
    else:
        dt = dt_or_iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    local_dt = dt.astimezone(ZoneInfo(timezone_name))
    hour = local_dt.strftime("%I").lstrip("0") or "12"
    return local_dt.strftime(f"{hour}:%M %p %Z")


def local_date_time_string(
    dt_or_iso: "str | datetime",
    timezone_name: str = "America/New_York",
) -> str:
    """Full date + time string in local timezone.

    Example: '2026-04-28 5:07 PM EDT'
    """
    if isinstance(dt_or_iso, str):
        dt = datetime.fromisoformat(dt_or_iso)
    else:
        dt = dt_or_iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    local_dt = dt.astimezone(ZoneInfo(timezone_name))
    hour = local_dt.strftime("%I").lstrip("0") or "12"
    return local_dt.strftime(f"%Y-%m-%d {hour}:%M %p %Z")


def local_now_string(timezone_name: str = "America/New_York") -> str:
    """Return the current local date + time as a human-readable string.

    Example: '2026-04-28 5:07 PM EDT'
    """
    return local_date_time_string(datetime.now(UTC), timezone_name)
