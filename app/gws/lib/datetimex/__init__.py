"""Date and time utilities.

These utilities are wrappers around the ``datetime`` module. Some functions also use
``pendulum`` (https://pendulum.eustace.io/), however all functions here return
stock ``datetime.datetime`` objects, and all returned objects are timezone-aware.

Functions in this package fall into these groups:

- time zones: ``time_zone``, ``is_valid_time_zone``, ``set_local_time_zone``,
- constructors and parsers: ``new``, ``now``, ``today``, ``parse``, ``from_string``, ``from_iso_string``, ``from_timestamp`` and others,
- formatters: ``to_iso_string``, ``to_iso_date_string``, ``to_basic_string``, ``to_string`` and others,
- converters: ``to_timestamp``, ``to_millis``, ``to_utc``, ``to_local``, ``to_time_zone``,
- predicates: ``is_date``, ``is_datetime``, ``is_utc``, ``is_local``,
- arithmetic: ``add``, ``difference``, ``total_difference``, ``next``, ``prev``,
- wrappers for ``pendulum`` helpers: ``start_of_<unit>`` and ``end_of_<unit>`` (for ``second``, ``minute``, ``hour``, ``day``, ``week``, ``month``, ``year``),
  ``day_of_week``, ``day_of_year``, ``week_of_month``, ``week_of_year``, ``days_in_month``,
- durations: ``parse_duration``, ``format_duration``.

Time zones are given as zoneinfo strings, like ``Europe/Berlin``. An empty string (the default)
means the local time zone. Alias names like ``CEST`` are not supported.

When a function accepts a date or time argument, it is converted to a datetime as follows:

- ``None`` means the current date and time,
- naive ``datetime`` objects are assumed to be in the local time zone,
- ``date`` objects are promoted to ``datetime`` with the time set to midnight UTC,
- ``time`` objects are promoted to ``datetime`` with today's date.

Parsers (``parse``, ``from_string`` etc.) attach the given time zone to naive input,
and a parsed date becomes midnight in that time zone.

When running in a docker container, there are several ways to set up the local time zone:

- by setting the config variable ``server.timeZone`` (see ``gws.config.parser``),
- by setting the ``TZ`` environment variable,
- by mounting a host zone info file to ``/etc/localtime``.

Example::

    import gws.lib.datetimex as datetimex

    d = datetimex.parse('2024-05-01T12:30:00', tz='Europe/Berlin')
    datetimex.to_iso_string(d)  # '2024-05-01T12:30:00+0200'
    datetimex.to_iso_string(datetimex.to_utc(d), with_tz='Z')  # '2024-05-01T10:30:00Z'

    next_week = datetimex.add(d, weeks=1)
    datetimex.parse_duration('1h30m')  # 5400
"""

from typing import Optional

import datetime as dt
import contextlib
import os
import re
import zoneinfo

import pendulum
import pendulum._helpers
import pendulum.helpers
import pendulum.parsing
import pendulum.parsing.exceptions

import gws
import gws.lib.osx


class Error(gws.Error):
    """Date and time error."""

    pass


UTC = zoneinfo.ZoneInfo('UTC')
"""The UTC time zone."""

_ZI_CACHE = {
    'utc': UTC,
    'UTC': UTC,
    'Etc/UTC': UTC,
}

_ZI_ALL = set(zoneinfo.available_timezones())


# Time zones


def is_valid_time_zone(tz: str) -> bool:
    """Check if a time zone string is valid.

    Args:
        tz: Time zone string, like ``Europe/Berlin``.

    Returns:
        ``True`` if the time zone is known.
    """

    return tz in _ZI_CACHE or tz in _ZI_ALL


def set_local_time_zone(tz: str):
    """Set the local time zone for the system.

    The time zone is set by linking ``/etc/localtime`` to the zone info file,
    which requires root privileges. Nothing is done if the time zone is already set.

    Args:
        tz: Time zone string, like ``Europe/Berlin``.

    Raises:
        ``Error``: If the time zone is invalid, or the process is not running as root.
    """
    new_zi = time_zone(tz)
    cur_zi = _zone_info_from_localtime()

    gws.log.debug(f'set_local_time_zone: cur={cur_zi} new={new_zi}')

    if new_zi == cur_zi:
        return
    _set_localtime_from_zone_info(new_zi)

    gws.log.debug(f'set_local_time_zone: cur={_zone_info_from_localtime()}')


def time_zone(tz: str = '') -> zoneinfo.ZoneInfo:
    """Get a ZoneInfo object for the specified time zone.

    The local time zone is determined from ``/etc/localtime``; if that fails, UTC is assumed.

    Args:
        tz: Time zone string, like ``Europe/Berlin``. An empty string means the local time zone.

    Returns:
        The ZoneInfo object.

    Raises:
        ``Error``: If the time zone is invalid.
    """

    if tz in _ZI_CACHE:
        return _ZI_CACHE[tz]

    if not tz:
        _ZI_CACHE[''] = _zone_info_from_localtime()
        return _ZI_CACHE['']

    return _zone_info_from_string(tz)


def _set_localtime_from_zone_info(zi):
    if os.getuid() != 0:
        raise Error('cannot set timezone, must be root')
    gws.lib.osx.run(['ln', '-fs', f'/usr/share/zoneinfo/{zi}', '/etc/localtime'])


def _zone_info_from_localtime():
    a = '/etc/localtime'

    try:
        p = os.readlink(a)
    except FileNotFoundError:
        gws.log.warning(f'time zone: {a!r} not found, assuming UTC')
        return UTC

    m = re.search(r'zoneinfo/(.+)$', p)
    if not m:
        gws.log.warning(f'time zone: {a!r}={p!r} invalid, assuming UTC')
        return UTC

    try:
        return zoneinfo.ZoneInfo(m.group(1))
    except zoneinfo.ZoneInfoNotFoundError:
        gws.log.warning(f'time zone: {a!r}={p!r} not found, assuming UTC')
        return UTC


def _zone_info_from_string(tz):
    if tz not in _ZI_ALL:
        raise Error(f'invalid time zone {tz!r}')
    try:
        return zoneinfo.ZoneInfo(tz)
    except zoneinfo.ZoneInfoNotFoundError as exc:
        raise Error(f'invalid time zone {tz!r}') from exc


def _zone_info_from_tzinfo(tzinfo: dt.tzinfo):
    if type(tzinfo) is zoneinfo.ZoneInfo:
        return tzinfo
    s = str(tzinfo)
    if s == '+0:0':
        return UTC
    try:
        return _zone_info_from_string(s)
    except Error:
        pass


# init from the env variable right now

if 'TZ' in os.environ:
    _set_localtime_from_zone_info(_zone_info_from_string(os.environ['TZ']))


# Constructors


def new(year, month, day, hour=0, minute=0, second=0, microsecond=0, fold=0, tz: str = '') -> dt.datetime:
    """Create a new datetime object with the specified components.

    Args:
        year: Year.
        month: Month.
        day: Day.
        hour: Hour.
        minute: Minute.
        second: Second.
        microsecond: Microsecond.
        fold: Fold value for ambiguous local times, see ``datetime.datetime``.
        tz: Time zone string, the local time zone by default.

    Returns:
        A timezone-aware datetime.

    Raises:
        ``Error``: If the time zone is invalid.
    """

    return dt.datetime(year, month, day, hour, minute, second, microsecond, fold=fold, tzinfo=time_zone(tz))


def now(tz: str = '') -> dt.datetime:
    """Get the current date and time.

    Args:
        tz: Time zone string, the local time zone by default.

    Returns:
        The current datetime in the given time zone.
    """

    return _now(time_zone(tz))


def now_utc() -> dt.datetime:
    """Get the current date and time in UTC.

    Returns:
        The current datetime in UTC.
    """

    return _now(UTC)


# for testing

_MOCK_NOW = None


@contextlib.contextmanager
def mock_now(d):
    """Context manager that makes all functions here use a fixed current date and time, for testing.

    Args:
        d: Datetime to use as the current date and time.
    """
    global _MOCK_NOW
    _MOCK_NOW = d
    yield
    _MOCK_NOW = None


def _now(tzinfo):
    return _MOCK_NOW or dt.datetime.now(tz=tzinfo)


def today(tz: str = '') -> dt.datetime:
    """Get today's date at midnight.

    Args:
        tz: Time zone string, the local time zone by default.

    Returns:
        A datetime at midnight of the current day in the given time zone.
    """

    return now(tz).replace(hour=0, minute=0, second=0, microsecond=0)


def today_utc() -> dt.datetime:
    """Get today's date at midnight in UTC.

    Returns:
        A datetime at midnight of the current day in UTC.
    """

    return now_utc().replace(hour=0, minute=0, second=0, microsecond=0)


def parse(s: str | dt.datetime | dt.date | None, tz: str = '') -> Optional[dt.datetime]:
    """Parse a string, datetime, or date into a datetime object.

    Strings are parsed like in ``from_string``. Dates become midnight in the given time zone.

    Args:
        s: Input to parse.
        tz: Time zone for timezone-naive inputs, the local time zone by default.

    Returns:
        A datetime, or ``None`` if the input is empty or cannot be parsed.
    """

    if not s:
        return None

    if isinstance(s, dt.datetime):
        return _ensure_tzinfo(s, tz)

    if isinstance(s, dt.date):
        return new(s.year, s.month, s.day, tz=tz)

    try:
        return from_string(str(s), tz)
    except Error:
        pass


def parse_time(s: str | dt.time | None, tz: str = '') -> Optional[dt.datetime]:
    """Parse a string or time into a datetime object with today's date.

    Strings are parsed like in ``from_iso_time_string``.

    Args:
        s: Input to parse.
        tz: Time zone for timezone-naive inputs, the local time zone by default.

    Returns:
        A datetime, or ``None`` if the input is empty or cannot be parsed.
    """

    if not s:
        return

    if isinstance(s, dt.time):
        return _datetime(_ensure_tzinfo(s, tz))

    try:
        return from_iso_time_string(str(s), tz)
    except Error:
        pass


def from_string(s: str, tz: str = '') -> dt.datetime:
    """Parse a date or datetime string.

    Accepts ISO 8601 and some other common formats understood by ``pendulum``.
    A date without time becomes midnight in the given time zone.

    Args:
        s: Date or datetime string.
        tz: Time zone for timezone-naive inputs, the local time zone by default.

    Returns:
        A datetime.

    Raises:
        ``Error``: If the string cannot be parsed or is not a date or datetime.
    """

    return _pend_parse_datetime(s.strip(), tz, iso_only=False)


def from_iso_string(s: str, tz: str = '') -> dt.datetime:
    """Parse an ISO 8601 date or datetime string.

    A date without time becomes midnight in the given time zone.

    Args:
        s: ISO 8601 date or datetime string.
        tz: Time zone for timezone-naive inputs, the local time zone by default.

    Returns:
        A datetime.

    Raises:
        ``Error``: If the string cannot be parsed or is not a date or datetime.
    """

    return _pend_parse_datetime(s.strip(), tz, iso_only=True)


def from_iso_time_string(s: str, tz: str = '') -> dt.datetime:
    """Parse an ISO 8601 time string into a datetime with today's date.

    Args:
        s: ISO 8601 time string.
        tz: Time zone for timezone-naive inputs, the local time zone by default.

    Returns:
        A datetime.

    Raises:
        ``Error``: If the string cannot be parsed or is not a time.
    """

    return _pend_parse_time(s.strip(), tz, iso_only=True)


def from_timestamp(n: float, tz: str = '') -> dt.datetime:
    """Create a datetime from a Unix timestamp.

    Args:
        n: Unix timestamp, in seconds since the epoch.
        tz: Time zone string, the local time zone by default.

    Returns:
        A datetime in the given time zone.
    """

    return dt.datetime.fromtimestamp(n, tz=time_zone(tz))


# Formatters


def to_iso_string(d: Optional[dt.date] = None, with_tz='+', sep='T') -> str:
    """Convert a date or time to an ISO 8601 datetime string.

    Args:
        d: Date or time to convert, the current date and time by default.
        with_tz: Time zone suffix: ``"+"`` for ``+hhmm``, ``":"`` for ``+hh:mm``,
            ``"Z"`` for ``Z`` if the offset is zero and ``+hhmm`` otherwise. An empty value omits the time zone.
        sep: Separator between date and time.

    Returns:
        A string like ``2024-05-01T12:30:00+0200``.
    """

    d = _datetime(d)
    s = d.strftime(f'%Y-%m-%d{sep}%H:%M:%S')
    if not with_tz:
        return s
    tz = d.strftime('%z')
    if with_tz == 'Z' and tz == '+0000':
        return s + 'Z'
    if with_tz == ':' and len(tz) == 5:
        return s + tz[:3] + ':' + tz[3:]
    return s + tz


def to_iso_date_string(d: Optional[dt.date] = None) -> str:
    """Convert a date to an ISO date string.

    Args:
        d: Date to convert, the current date and time by default.

    Returns:
        A string like ``2024-05-01``.
    """

    return _datetime(d).strftime('%Y-%m-%d')


def to_basic_string(d: Optional[dt.date] = None, with_ms=False) -> str:
    """Convert a date to a compact string without separators.

    Args:
        d: Date to convert, the current date and time by default.
        with_ms: Append milliseconds as three digits.

    Returns:
        A string like ``20240501123000``, or ``20240501123000123`` with milliseconds.
    """

    d = _datetime(d)
    s = d.strftime('%Y%m%d%H%M%S')
    if with_ms:
        s += f'{d.microsecond // 1000:03d}'
    return s


def to_iso_time_string(d: Optional[dt.date] = None, with_tz='+') -> str:
    """Convert a date to an ISO 8601 time string.

    Args:
        d: Date to convert, the current date and time by default.
        with_tz: Time zone suffix: ``"+"`` for ``+hhmm``, ``"Z"`` for ``Z`` if the offset is zero
            and ``+hhmm`` otherwise. An empty value omits the time zone.

    Returns:
        A string like ``12:30:00+0200``.
    """

    fmt = '%H:%M:%S'
    if with_tz:
        fmt += '%z'
    s = _datetime(d).strftime(fmt)
    if with_tz == 'Z' and s.endswith('+0000'):
        s = s[:-5] + 'Z'
    return s


def to_string(fmt: str, d: Optional[dt.date] = None) -> str:
    """Convert a date to a string using a custom format.

    Args:
        fmt: ``strftime`` format string.
        d: Date to convert, the current date and time by default.

    Returns:
        The formatted string.
    """

    return _datetime(d).strftime(fmt)


def time_to_iso_string(d: Optional[dt.date | dt.time] = None) -> str:
    """Convert a date or time to a time string without time zone.

    Args:
        d: Datetime or time to convert. For a date or ``None``, ``00:00:00`` is returned.

    Returns:
        A string like ``12:30:00``.
    """

    if isinstance(d, (dt.datetime, dt.time)):
        return f'{d.hour:02d}:{d.minute:02d}:{d.second:02d}'
    return f'00:00:00'


# Converters


def to_timestamp(d: Optional[dt.date] = None) -> int:
    """Convert a date to a Unix timestamp.

    Args:
        d: Date to convert, the current date and time by default.

    Returns:
        Whole seconds since the epoch.
    """

    return int(_datetime(d).timestamp())


def to_millis(d: Optional[dt.date] = None) -> int:
    """Convert a date to milliseconds since the Unix epoch.

    Args:
        d: Date to convert, the current date and time by default.

    Returns:
        Whole milliseconds since the epoch.
    """

    return int(_datetime(d).timestamp() * 1000)


def to_utc(d: Optional[dt.date] = None) -> dt.datetime:
    """Convert a date to the UTC time zone.

    Args:
        d: Date to convert, the current date and time by default.

    Returns:
        A datetime in UTC.
    """

    return _datetime(d).astimezone(time_zone('UTC'))


def to_local(d: Optional[dt.date] = None) -> dt.datetime:
    """Convert a date to the local time zone.

    Args:
        d: Date to convert, the current date and time by default.

    Returns:
        A datetime in the local time zone.
    """

    return _datetime(d).astimezone(time_zone(''))


def to_time_zone(tz: str, d: Optional[dt.date] = None) -> dt.datetime:
    """Convert a date to a specific time zone.

    Args:
        tz: Target time zone string.
        d: Date to convert, the current date and time by default.

    Returns:
        A datetime in the target time zone.

    Raises:
        ``Error``: If the time zone is invalid.
    """

    return _datetime(d).astimezone(time_zone(tz))


# Predicates


def is_date(x) -> bool:
    """Check if an object is a date.

    Args:
        x: Object to check.

    Returns:
        ``True`` if the object is a ``date``. Since ``datetime`` is a subclass of ``date``, this is also ``True`` for datetimes.
    """

    return isinstance(x, dt.date)


def is_datetime(x) -> bool:
    """Check if an object is a datetime.

    Args:
        x: Object to check.

    Returns:
        ``True`` if the object is a ``datetime``.
    """

    return isinstance(x, dt.datetime)


def is_utc(d: dt.datetime) -> bool:
    """Check if a datetime is in the UTC time zone.

    Args:
        d: Datetime to check. A naive datetime is assumed to be local.

    Returns:
        ``True`` if the time zone of the datetime is UTC.
    """

    return _zone_info_from_tzinfo(gws.u.require(_datetime(d).tzinfo)) == UTC


def is_local(d: dt.datetime) -> bool:
    """Check if a datetime is in the local time zone.

    Args:
        d: Datetime to check. A naive datetime is assumed to be local.

    Returns:
        ``True`` if the time zone of the datetime is the local time zone.
    """

    return _zone_info_from_tzinfo(gws.u.require(_datetime(d).tzinfo)) == time_zone('')


# Arithmetic


def add(d: Optional[dt.date] = None, years=0, months=0, days=0, weeks=0, hours=0, minutes=0, seconds=0, microseconds=0) -> dt.datetime:
    """Add a duration to a date.

    Negative values subtract.

    Args:
        d: Base date, the current date and time by default.
        years: Years to add.
        months: Months to add.
        days: Days to add.
        weeks: Weeks to add.
        hours: Hours to add.
        minutes: Minutes to add.
        seconds: Seconds to add.
        microseconds: Microseconds to add.

    Returns:
        The resulting datetime.
    """

    return pendulum.helpers.add_duration(
        _datetime(d),
        years=years,
        months=months,
        days=days,
        weeks=weeks,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        microseconds=microseconds,
    )


class Diff:
    """Difference between two dates, as returned by ``difference`` and ``total_difference``."""

    years: int
    """Years."""
    months: int
    """Months."""
    weeks: int
    """Weeks."""
    days: int
    """Days."""
    hours: int
    """Hours."""
    minutes: int
    """Minutes."""
    seconds: int
    """Seconds."""
    microseconds: int
    """Microseconds."""

    def __repr__(self):
        return repr(vars(self))


def difference(d1: dt.date, d2: Optional[dt.date] = None) -> Diff:
    """Compute the difference between two dates, broken down into components.

    The components add up to the whole difference, for example ``1 year, 2 months, 1 week, 3 days``.

    Args:
        d1: The start date.
        d2: The end date, the current date and time by default.

    Returns:
        The difference from ``d1`` to ``d2``.
    """

    pd = _precise_diff(d1, d2)
    df = Diff()

    df.years = pd.years
    df.months = pd.months
    df.weeks = _sign(pd.days) * (abs(pd.days) // 7)
    df.days = _sign(pd.days) * (abs(pd.days) % 7)
    df.hours = pd.hours
    df.minutes = pd.minutes
    df.seconds = pd.seconds
    df.microseconds = pd.microseconds

    return df


def total_difference(d1: dt.date, d2: Optional[dt.date] = None) -> Diff:
    """Compute the total difference between two dates in each unit.

    Each component holds the whole difference expressed in that unit,
    for example, for a difference of one year and two months, ``years`` is 1 and ``months`` is 14.

    Args:
        d1: The start date.
        d2: The end date, the current date and time by default.

    Returns:
        The difference from ``d1`` to ``d2``.
    """

    pd = _precise_diff(d1, d2)
    total = (_utc(_datetime(d2)) - _utc(_datetime(d1))).total_seconds()
    df = Diff()

    df.years = pd.years
    df.months = pd.years * 12 + pd.months
    df.weeks = _sign(pd.total_days) * (abs(pd.total_days) // 7)
    df.days = pd.total_days
    df.hours = int(total / 3600)
    df.minutes = int(total / 60)
    df.seconds = int(total)
    df.microseconds = df.seconds * 1_000_000

    return df


def _precise_diff(d1, d2):
    # NB pendulum's compiled `precise_diff` is broken, use the pure python version
    return pendulum._helpers.precise_diff(_datetime(d1), _datetime(d2))


def _utc(d: dt.datetime) -> dt.datetime:
    return d.astimezone(dt.timezone.utc)


def _sign(n: int) -> int:
    return -1 if n < 0 else 1


# Wrappers for useful pendulum utilities

# fmt:off

def start_of_second(d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).start_of('second'))
def start_of_minute(d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).start_of('minute'))
def start_of_hour  (d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).start_of('hour'))
def start_of_day   (d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).start_of('day'))
def start_of_week  (d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).start_of('week'))
def start_of_month (d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).start_of('month'))
def start_of_year  (d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).start_of('year'))


def end_of_second(d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).end_of('second'))
def end_of_minute(d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).end_of('minute'))
def end_of_hour  (d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).end_of('hour'))
def end_of_day   (d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).end_of('day'))
def end_of_week  (d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).end_of('week'))
def end_of_month (d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).end_of('month'))
def end_of_year  (d: Optional[dt.date] = None) -> dt.datetime: return _unpend(_pend(d).end_of('year'))


def day_of_week   (d: Optional[dt.date] = None) -> int: return _pend(d).day_of_week
def day_of_year   (d: Optional[dt.date] = None) -> int: return _pend(d).day_of_year
def week_of_month (d: Optional[dt.date] = None) -> int: return _pend(d).week_of_month
def week_of_year  (d: Optional[dt.date] = None) -> int: return _pend(d).week_of_year
def days_in_month (d: Optional[dt.date] = None) -> int: return _pend(d).days_in_month


# fmt:on

_WD = {
    0: pendulum.WeekDay.MONDAY,
    1: pendulum.WeekDay.TUESDAY,
    2: pendulum.WeekDay.WEDNESDAY,
    3: pendulum.WeekDay.THURSDAY,
    4: pendulum.WeekDay.FRIDAY,
    5: pendulum.WeekDay.SATURDAY,
    6: pendulum.WeekDay.SUNDAY,
    'monday': pendulum.WeekDay.MONDAY,
    'tuesday': pendulum.WeekDay.TUESDAY,
    'wednesday': pendulum.WeekDay.WEDNESDAY,
    'thursday': pendulum.WeekDay.THURSDAY,
    'friday': pendulum.WeekDay.FRIDAY,
    'saturday': pendulum.WeekDay.SATURDAY,
    'sunday': pendulum.WeekDay.SUNDAY,
}


def next(day: int | str, d: Optional[dt.date] = None, keep_time=False) -> dt.datetime:
    """Get the next occurrence of a specific weekday.

    Args:
        day: Day of the week, ``0`` to ``6`` for Monday to Sunday, or a lowercase weekday name like ``monday``.
        d: Starting date, the current date and time by default.
        keep_time: Keep the time of the starting date, otherwise the time is set to midnight.

    Returns:
        The datetime of the next occurrence after the starting date.
    """

    return _unpend(_pend(d).next(_WD[day], keep_time))


def prev(day: int | str, d: Optional[dt.date] = None, keep_time=False) -> dt.datetime:
    """Get the previous occurrence of a specific weekday.

    Args:
        day: Day of the week, ``0`` to ``6`` for Monday to Sunday, or a lowercase weekday name like ``monday``.
        d: Starting date, the current date and time by default.
        keep_time: Keep the time of the starting date, otherwise the time is set to midnight.

    Returns:
        The datetime of the previous occurrence before the starting date.
    """

    return _unpend(_pend(d).previous(_WD[day], keep_time))


# Duration

_DURATION_UNITS = {
    'w': 3600 * 24 * 7,
    'd': 3600 * 24,
    'h': 3600,
    'm': 60,
    's': 1,
}


def parse_duration(s: str) -> int:
    """Convert a duration string to seconds.

    The string consists of numbers followed by units ``w``, ``d``, ``h``, ``m`` or ``s``,
    like ``1w2d3h4m5s``. A trailing number without a unit is taken as seconds.

    Args:
        s: Duration string, or an integer number of seconds, which is returned as is.

    Returns:
        The duration in seconds.

    Raises:
        ``Error``: If the string is not a valid duration.
    """

    if isinstance(s, int):
        return s

    p = None
    r = 0

    for n, v in re.findall(r'(\d+)|(\D+)', str(s).strip()):
        if n:
            p = int(n)
            continue
        v = v.strip()
        if p is None or v not in _DURATION_UNITS:
            raise Error('invalid duration', s)
        r += p * _DURATION_UNITS[v]
        p = None

    if p:
        r += p

    return r


def format_duration(s: int) -> str:
    """Format a duration in seconds to a string.

    Args:
        s: Duration in seconds.

    Returns:
        A string like ``1d 2h 30m``, or ``0s`` for a zero duration.
    """

    r = ''

    for u, v in _DURATION_UNITS.items():
        n = s // v
        if n:
            r += f'{n}{u} '
            s -= n * v

    return r.strip() or '0s'

##

# conversions


def _datetime(d: dt.date | dt.time | None) -> dt.datetime:
    # ensure a valid datetime object

    if d is None:
        return now()

    if isinstance(d, dt.datetime):
        # if a value is a naive datetime, assume the local tz
        # see https://www.postgresql.org/docs/current/datatype-datetime.html#DATATYPE-DATETIME-INPUT-TIME-STAMPS:
        # > Conversions between timestamp without time zone and timestamp with time zone normally assume
        # > that the timestamp without time zone value should be taken or given as timezone local time.
        return _ensure_tzinfo(d, tz='')

    if isinstance(d, dt.date):
        # promote date to midnight UTC
        return dt.datetime(d.year, d.month, d.day, tzinfo=UTC)

    if isinstance(d, dt.time):
        # promote time to today's time
        n = _now(d.tzinfo)
        return dt.datetime(n.year, n.month, n.day, d.hour, d.minute, d.second, d.microsecond, d.tzinfo, fold=d.fold)

    raise Error(f'invalid datetime value {d!r}')


def _ensure_tzinfo(d, tz: str):
    # attach tzinfo if not set

    if not d.tzinfo:
        return d.replace(tzinfo=time_zone(tz))

    # try to convert 'their' tzinfo (might be an unnamed dt.timezone or pendulum.FixedTimezone) to zoneinfo
    zi = _zone_info_from_tzinfo(d.tzinfo)
    if zi:
        return d.replace(tzinfo=zi)

    # failing that, keep existing tzinfo
    return d


# pendulum.DateTime <-> python datetime


def _pend(d: dt.date | None) -> pendulum.DateTime:
    return pendulum.instance(_datetime(d))


def _unpend(p: pendulum.DateTime) -> dt.datetime:
    return dt.datetime(
        p.year,
        p.month,
        p.day,
        p.hour,
        p.minute,
        p.second,
        p.microsecond,
        tzinfo=p.tzinfo,
        fold=p.fold,
    )


# NB using private APIs


def _pend_parse_datetime(s, tz, iso_only):
    try:
        if iso_only:
            d = pendulum.parsing.parse_iso8601(s)
        else:
            # do not normalize
            d = pendulum.parsing._parse(s)
    except (ValueError, pendulum.parsing.exceptions.ParserError) as exc:
        raise Error(f'invalid date {s!r}') from exc

    if isinstance(d, dt.datetime):
        return _ensure_tzinfo(d, tz)
    if isinstance(d, dt.date):
        return new(d.year, d.month, d.day, tz=tz)

    # times and durations not accepted
    raise Error(f'invalid date {s!r}')


def _pend_parse_time(s, tz, iso_only):
    try:
        if iso_only:
            d = pendulum.parsing.parse_iso8601(s)
        else:
            # do not normalize
            d = pendulum.parsing._parse(s)
    except (ValueError, pendulum.parsing.exceptions.ParserError) as exc:
        raise Error(f'invalid time {s!r}') from exc

    if isinstance(d, dt.time):
        return _datetime(_ensure_tzinfo(d, tz))

    # dates and durations not accepted
    raise Error(f'invalid time {s!r}')
