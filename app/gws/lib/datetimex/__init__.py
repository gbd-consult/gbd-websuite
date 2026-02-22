"""Date and time utilities.

These utilities are wrappers around the `datetime` module,
some functions also use `pendulum` (https://pendulum.eustace.io/),
however all functions here return strictly stock ``datetime.datetime`` objects.

``date`` objects are silently promoted to ``datetime`` with time set to midnight UTC.
``time`` objects are silently promoted to ``datetime`` in the local timezone with the today's date.

This module always returns timezone-aware objects.

When constructing an object (e.g. from a string), the default time zone should be passed
as a zoneinfo string (like ``Europe/Berlin``). An empty zoneinfo string (default) means the local time zone.
Alias names like ``CEST`` are not supported.

Naive datetime arguments are assumed to be in the local time zone.

When running in a docker container, there are several ways to set up the local time zone:

- by setting the config variable ``server.timeZone`` (see `gws.config.parser`)
- by setting the ``TZ`` environment variable
- mounting a host zone info file to ``/etc/localtime``
"""

from typing import Optional

import datetime as dt
import contextlib
import os
import re
import zoneinfo

import pendulum
import pendulum.helpers
import pendulum.parsing
import pendulum.parsing.exceptions

import gws
import gws.lib.osx


class Error(gws.Error):
    pass


UTC = zoneinfo.ZoneInfo('UTC')

_ZI_CACHE = {
    'utc': UTC,
    'UTC': UTC,
    'Etc/UTC': UTC,
}

_ZI_ALL = set(zoneinfo.available_timezones())


# Time zones


def set_local_time_zone(tz: str):
    """Set the local time zone for the system.

    Args:
        tz: Time zone string (e.g. 'Europe/Berlin')
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

    Args:
        tz: Time zone string (e.g. 'Europe/Berlin'). Empty string returns local time zone.
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
        year: Year component
        month: Month component
        day: Day component
        hour: Hour component (default 0)
        minute: Minute component (default 0)
        second: Second component (default 0)
        microsecond: Microsecond component (default 0)
        fold: Fold component for ambiguous times (default 0)
        tz: Time zone string (default empty for local time zone)
    """

    return dt.datetime(year, month, day, hour, minute, second, microsecond, fold=fold, tzinfo=time_zone(tz))


def now(tz: str = '') -> dt.datetime:
    """Get the current date and time.

    Args:
        tz: Time zone string (default empty for local time zone)
    """

    return _now(time_zone(tz))


def now_utc() -> dt.datetime:
    """Get the current date and time in UTC."""

    return _now(UTC)


# for testing

_MOCK_NOW = None


@contextlib.contextmanager
def mock_now(d):
    global _MOCK_NOW
    _MOCK_NOW = d
    yield
    _MOCK_NOW = None


def _now(tzinfo):
    return _MOCK_NOW or dt.datetime.now(tz=tzinfo)


def today(tz: str = '') -> dt.datetime:
    """Get today's date at midnight.

    Args:
        tz: Time zone string (default empty for local time zone)
    """

    return now(tz).replace(hour=0, minute=0, second=0, microsecond=0)


def today_utc() -> dt.datetime:
    """Get today's date at midnight in UTC."""

    return now_utc().replace(hour=0, minute=0, second=0, microsecond=0)


def parse(s: str | dt.datetime | dt.date | None, tz: str = '') -> Optional[dt.datetime]:
    """Parse a string, datetime, or date into a datetime object.

    Args:
        s: Input to parse (string, datetime, date, or None)
        tz: Default time zone string for timezone-naive inputs
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
    """Parse a string or time into a datetime object.

    Args:
        s: Input to parse (string, time, or None)
        tz: Default time zone string for timezone-naive inputs
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
    """Parse a datetime string using flexible parsing.

    Args:
        s: Date/time string to parse
        tz: Default time zone string for timezone-naive inputs
    """

    return _pend_parse_datetime(s.strip(), tz, iso_only=False)


def from_iso_string(s: str, tz: str = '') -> dt.datetime:
    """Parse an ISO 8601 datetime string.

    Args:
        s: ISO 8601 date/time string to parse
        tz: Default time zone string for timezone-naive inputs
    """

    return _pend_parse_datetime(s.strip(), tz, iso_only=True)


def from_iso_time_string(s: str, tz: str = '') -> dt.datetime:
    """Parse an ISO 8601 time string.

    Args:
        s: ISO 8601 time string to parse
        tz: Default time zone string for timezone-naive inputs
    """

    return _pend_parse_time(s.strip(), tz, iso_only=True)


def from_timestamp(n: float, tz: str = '') -> dt.datetime:
    """Create a datetime from a Unix timestamp.

    Args:
        n: Unix timestamp (seconds since epoch)
        tz: Time zone string (default empty for local time zone)
    """

    return dt.datetime.fromtimestamp(n, tz=time_zone(tz))


# Formatters


def to_iso_string(d: Optional[dt.date] = None, with_tz='+', sep='T') -> str:
    """Convert a date or time to an ISO string.

    Args:
        d: Date or time to convert. If None, the current date and time is used.
        with_tz: If set, append the time zone information. Can be "Z" (for UTC), ":" (for ISO 8601 format) or "+" (for +hhmm format).
        sep: Separator between date and time. Default is "T".
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
    """Convert a date to an ISO date string (YYYY-MM-DD format).

    Args:
        d: Date to convert (default current date/time)
    """

    return _datetime(d).strftime('%Y-%m-%d')


def to_basic_string(d: Optional[dt.date] = None, with_ms=False) -> str:
    """Convert a date to a basic string format (YYYYMMDDHHMMSS).

    Args:
        d: Date to convert (default current date/time)
    """

    d = _datetime(d)
    s = d.strftime('%Y%m%d%H%M%S')
    if with_ms:
        s += f'{d.microsecond // 1000:03d}'
    return s


def to_iso_time_string(d: Optional[dt.date] = None, with_tz='+') -> str:
    """Convert a date to an ISO time string.

    Args:
        d: Date to convert (default current date/time)
        with_tz: Include timezone information ('+' for +hhmm, 'Z' for UTC as Z, False for no timezone)
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
        fmt: strftime format string
        d: Date to convert (default current date/time)
    """

    return _datetime(d).strftime(fmt)


def time_to_iso_string(d: Optional[dt.date | dt.time] = None, with_tz='+') -> str:
    """Convert a date or time to an ISO time string.

    Args:
        d: Date or time to convert (default returns 00:00:00)
        with_tz: Include timezone information (unused in current implementation)
    """

    if isinstance(d, (dt.datetime, dt.time)):
        return f'{d.hour:02d}:{d.minute:02d}:{d.second:02d}'
    return f'00:00:00'


# Converters


def to_timestamp(d: Optional[dt.date] = None) -> int:
    """Convert a date to a Unix timestamp.

    Args:
        d: Date to convert (default current date/time)
    """

    return int(_datetime(d).timestamp())


def to_millis(d: Optional[dt.date] = None) -> int:
    """Convert a date to milliseconds since Unix epoch.

    Args:
        d: Date to convert (default current date/time)
    """

    return int(_datetime(d).timestamp() * 1000)


def to_utc(d: Optional[dt.date] = None) -> dt.datetime:
    """Convert a date to UTC timezone.

    Args:
        d: Date to convert (default current date/time)
    """

    return _datetime(d).astimezone(time_zone('UTC'))


def to_local(d: Optional[dt.date] = None) -> dt.datetime:
    """Convert a date to local timezone.

    Args:
        d: Date to convert (default current date/time)
    """

    return _datetime(d).astimezone(time_zone(''))


def to_time_zone(tz: str, d: Optional[dt.date] = None) -> dt.datetime:
    """Convert a date to a specific timezone.

    Args:
        tz: Target timezone string
        d: Date to convert (default current date/time)
    """

    return _datetime(d).astimezone(time_zone(tz))


# Predicates


def is_date(x) -> bool:
    """Check if an object is a date.

    Args:
        x: Object to check
    """

    return isinstance(x, dt.date)


def is_datetime(x) -> bool:
    """Check if an object is a datetime.

    Args:
        x: Object to check
    """

    return isinstance(x, dt.datetime)


def is_utc(d: dt.datetime) -> bool:
    """Check if a datetime is in UTC timezone.

    Args:
        d: Datetime to check
    """

    return _zone_info_from_tzinfo(gws.u.require(_datetime(d).tzinfo)) == UTC


def is_local(d: dt.datetime) -> bool:
    """Check if a datetime is in the local timezone.

    Args:
        d: Datetime to check
    """

    return _zone_info_from_tzinfo(gws.u.require(_datetime(d).tzinfo)) == time_zone('')


# Arithmetic


def add(d: Optional[dt.date] = None, years=0, months=0, days=0, weeks=0, hours=0, minutes=0, seconds=0, microseconds=0) -> dt.datetime:
    """Add time components to a datetime.

    Args:
        d: Base datetime (default current date/time)
        years: Years to add
        months: Months to add
        days: Days to add
        weeks: Weeks to add
        hours: Hours to add
        minutes: Minutes to add
        seconds: Seconds to add
        microseconds: Microseconds to add
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
    """Difference between two dates."""

    years: int
    months: int
    weeks: int
    days: int
    hours: int
    minutes: int
    seconds: int
    microseconds: int

    def __repr__(self):
        return repr(vars(self))


def difference(d1: dt.date, d2: Optional[dt.date] = None) -> Diff:
    """Compute the difference between two dates.

    Args:
        d1: The first date.
        d2: The second date. If None, the current date and time is used.
    """

    iv = pendulum.Interval(_datetime(d1), _datetime(d2), absolute=False)
    df = Diff()

    df.years = iv.years
    df.months = iv.months
    df.weeks = iv.weeks
    df.days = iv.remaining_days
    df.hours = iv.hours
    df.minutes = iv.minutes
    df.seconds = iv.remaining_seconds
    df.microseconds = iv.microseconds

    return df


def total_difference(d1: dt.date, d2: Optional[dt.date] = None) -> Diff:
    """Compute the total difference between two dates in each unit.

    Args:
        d1: First date
        d2: Second date (default current date/time)
    """

    iv = pendulum.Interval(_datetime(d1), _datetime(d2), absolute=False)
    df = Diff()

    df.years = iv.in_years()
    df.months = iv.in_months()
    df.weeks = iv.in_weeks()
    df.days = iv.in_days()
    df.hours = iv.in_hours()
    df.minutes = iv.in_minutes()
    df.seconds = iv.in_seconds()
    df.microseconds = df.seconds * 1_000_000

    return df


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
        day: Day of week (0-6 for Monday-Sunday or weekday name string)
        d: Starting date (default current date/time)
        keep_time: Whether to keep the time component
    """

    return _unpend(_pend(d).next(_WD[day], keep_time))


def prev(day: int | str, d: Optional[dt.date] = None, keep_time=False) -> dt.datetime:
    """Get the previous occurrence of a specific weekday.

    Args:
        day: Day of week (0-6 for Monday-Sunday or weekday name string)
        d: Starting date (default current date/time)
        keep_time: Whether to keep the time component
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
    """Convert duration string to seconds.

    Args:
        s: Duration string (e.g. '1w2d3h4m5s') or integer seconds
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
        s: Duration in seconds
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
