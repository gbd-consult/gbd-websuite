import datetime as dt
import re
import zoneinfo
from typing import Optional

import pendulum
import pendulum.parsing

import gws.lib.osx

_ZONES = {
    '': zoneinfo.ZoneInfo('UTC'),
    'utc': zoneinfo.ZoneInfo('UTC'),
    'UTC': zoneinfo.ZoneInfo('UTC'),
}


# System time zone

def set_system_time_zone(tz: str = 'UTC'):
    zi = _get_zi(tz)

    if zi == _get_system_zi():
        return

    gws.lib.osx.run(['ln', '-fs', f'/usr/share/zoneinfo/{zi}', '/etc/localtime'])

    _ZONES['local'] = zi


def get_system_time_zone():
    if not _ZONES.get('local'):
        _ZONES['local'] = _get_system_zi()
    return _ZONES['local']


def _get_system_zi():
    d = dt.datetime.now().astimezone()
    return zoneinfo.ZoneInfo(str(d.tzinfo))


def _set_default_zi(d: dt.datetime | dt.time, tz: str):
    zi = d.tzinfo
    if not zi:
        return d.replace(tzinfo=_get_zi(tz))
    if not isinstance(zi, zoneinfo.ZoneInfo):
        return d.replace(tzinfo=zoneinfo.ZoneInfo(str(zi)))
    return d


def _get_zi(tz: str) -> zoneinfo.ZoneInfo:
    if tz in _ZONES:
        return _ZONES[tz]
    if tz.lower() == 'local':
        return get_system_time_zone()
    try:
        return zoneinfo.ZoneInfo(tz)
    except zoneinfo.ZoneInfoNotFoundError as exc:
        raise ValueError(f'invalid time zone {tz!r}') from exc


# Constructors


def make_datetime(year, month=None, day=None, hour=0, minute=0, second=0, microsecond=0, tz: str = 'UTC') -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, second, microsecond, tzinfo=_get_zi(tz))


def make_date(year, month=None, day=None) -> dt.date:
    return dt.date(year, month, day)


def now(tz: str = 'UTC') -> dt.datetime:
    return dt.datetime.now(_get_zi(tz))


def now_local() -> dt.datetime:
    return now('local')


def parse(s, tz: str = 'UTC') -> Optional[dt.date]:
    if not s:
        return None

    if isinstance(s, dt.datetime):
        return _set_default_zi(s, tz)

    try:
        return from_iso_string(s, tz)
    except ValueError:
        pass
    try:
        return from_dmy_string(s)
    except ValueError:
        pass
    try:
        return from_string(s, tz)
    except ValueError:
        pass


def from_string(s: str, tz: str = 'UTC') -> dt.datetime:
    try:
        d = pendulum.parse(s)
    except pendulum.parsing.exceptions.ParserError as exc:
        raise ValueError(f'invalid date {s!r}') from exc
    return _set_default_zi(d, tz)


def from_iso_string(s: str, tz: str = 'UTC') -> dt.date:
    try:
        d = dt.datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f'invalid date {s!r}') from exc

    if len(s) == 10:
        return dt.date(d.year, d.month, d.day)

    return _set_default_zi(d, tz)


_DMY_RE = r'''(?x)
    ^
        (?P<d> \d{1,2})
        [./\s]
        (?P<m> \d{1,2})
        [./\s]
        (?P<Y> \d{2,4})
    $
'''


def from_dmy_string(s: str) -> dt.date:
    m = re.match(_DMY_RE, s)
    if not m:
        raise ValueError(f'invalid date {s!r}')

    g = m.groupdict()

    try:
        return dt.date(int(g['Y']), int(g['m']), int(g['d']))
    except ValueError as exc:
        raise ValueError(f'invalid date {s!r}') from exc


def from_timestamp(ts: float, tz: str = 'UTC') -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, tz=_get_zi(tz))


# Formatters

def to_iso_string(d: dt.date, with_tz='+', sep='T') -> str:
    if isinstance(d, dt.datetime):
        fmt = f'%Y-%m-%d{sep}%H:%M:%S'
        if with_tz:
            fmt += '%z'
        s = d.strftime(fmt)
        if with_tz == 'Z' and s.endswith('+0000'):
            s = s[:-5] + 'Z'
        return s

    return to_iso_date_string(d)


def to_iso_date_string(d: dt.date) -> str:
    fmt = '%Y-%m-%d'
    return d.strftime(fmt)


def to_int_string(d: dt.datetime) -> str:
    return d.strftime("%Y%m%d%H%M%S")


# Converters

def to_timestamp(d: dt.datetime) -> int:
    return int(d.timestamp())


def to_utc(d: dt.datetime) -> dt.datetime:
    return d.astimezone(_get_zi('UTC'))


def to_local(d: dt.datetime) -> dt.datetime:
    return d.astimezone(get_system_time_zone())


def to_time_zone(d: dt.datetime, tz: str) -> dt.datetime:
    return d.astimezone(_get_zi(tz))


def to_datetime(d: dt.date, tz: str = 'UTC') -> dt.datetime:
    return dt.datetime(d.year, d.month, d.day, tzinfo=_get_zi(tz))


def to_date(d: dt.date) -> dt.date:
    return dt.date(d.year, d.month, d.day)


# Predicates

def is_date(x) -> bool:
    return isinstance(x, dt.date)


def is_datetime(x) -> bool:
    return isinstance(x, dt.datetime)


# Arithmetic

def add(
        d: dt.datetime,
        years=0, months=0, days=0, weeks=0, hours=0, minutes=0, seconds=0, microseconds=0
) -> dt.datetime:
    d = pendulum.helpers.add_duration(
        d,
        years=years, months=months, days=days,
        weeks=weeks, hours=hours, minutes=minutes, seconds=seconds, microseconds=microseconds
    )
    return d


class Diff:
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


def relative_difference(dt1: dt.datetime, dt2: dt.datetime) -> Diff:
    iv = pendulum.Interval(dt1, dt2, absolute=False)
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


def absolute_difference(dt1: dt.datetime, dt2: dt.datetime) -> Diff:
    iv = pendulum.Interval(dt1, dt2, absolute=False)
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


_TRUNC_KEYS = ['month', 'day', 'hour', 'minute', 'second', 'microsecond']


# https://www.postgresql.org/docs/current/functions-datetime.html#FUNCTIONS-DATETIME-TRUNC

def trunc(d: dt.datetime, year=None, month=None, day=None, hour=None, minute=None, second=None) -> dt.datetime:
    args = [d.year, 1, 1, 0, 0, 0, 0]
    i = 1

    for k in _TRUNC_KEYS:
        if k == key:
            break
        args[i] = getattr(d, k)
        i += 1

    dt2 = dt.datetime(*args, tzinfo=d.tzinfo)
    return _set_default_zi(dt2, 'UTC')


# Time

def parse_time(s, tz: str = 'UTC') -> Optional[dt.time]:
    if isinstance(s, dt.datetime):
        return _set_default_zi(s.timetz(), tz)
    if isinstance(s, dt.time):
        return _set_default_zi(s, tz)
    try:
        return time_from_iso_string(str(s), tz)
    except ValueError:
        pass


def time_from_iso_string(s: str, tz: str = 'UTC') -> dt.time:
    d = dt.time.fromisoformat(s)
    return _set_default_zi(d, tz)


def time_to_iso_string(d: dt.time, with_tz='+') -> str:
    fmt = '%H:%M:%S'
    if with_tz:
        fmt += '%z'
    s = d.strftime(fmt)
    if with_tz == 'Z' and s.endswith('+0000'):
        s = s[:-5] + 'Z'
    return s
