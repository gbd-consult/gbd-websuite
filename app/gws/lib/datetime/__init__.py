import datetime as dt
import re
import zoneinfo
from typing import Optional

import pendulum
import pendulum.parsing

import gws.lib.osx

_ZONES = {
    'utc': zoneinfo.ZoneInfo('UTC'),
    'UTC': zoneinfo.ZoneInfo('UTC'),
}


# System time zone

def set_system_time_zone(tz: str = 'UTC'):
    zi = time_zone(tz)

    if zi == _system_zi():
        return

    gws.lib.osx.run(['ln', '-fs', f'/usr/share/zoneinfo/{zi}', '/etc/localtime'])

    _ZONES['local'] = zi


def system_time_zone():
    if not _ZONES.get('local'):
        _ZONES['local'] = _system_zi()
    return _ZONES['local']


def _system_zi():
    d = dt.datetime.now().astimezone()
    return zoneinfo.ZoneInfo(str(d.tzinfo))


def _set_default_zi(d: dt.datetime | dt.time, tz: str):
    zi = d.tzinfo
    if not zi:
        return d.replace(tzinfo=time_zone(tz))
    if not isinstance(zi, zoneinfo.ZoneInfo):
        return d.replace(tzinfo=zoneinfo.ZoneInfo(str(zi)))
    return d


def time_zone(tz: str) -> zoneinfo.ZoneInfo:
    if tz in _ZONES:
        return _ZONES[tz]
    if tz.lower() == 'local':
        return system_time_zone()
    try:
        return zoneinfo.ZoneInfo(tz)
    except zoneinfo.ZoneInfoNotFoundError as exc:
        raise ValueError(f'invalid time zone {tz!r}') from exc


# Constructors


def new(year, month=None, day=None, hour=0, minute=0, second=0, microsecond=0, tz: str = 'UTC') -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, second, microsecond, tzinfo=time_zone(tz))


def now(tz: str = 'UTC') -> dt.datetime:
    return dt.datetime.now(time_zone(tz))


def now_local() -> dt.datetime:
    return dt.datetime.now(system_time_zone())


def parse(s, tz: str = 'UTC') -> Optional[dt.datetime]:
    if not s:
        return None

    if isinstance(s, dt.datetime):
        return _set_default_zi(s, tz)

    try:
        return from_iso_string(s, tz)
    except ValueError:
        pass
    try:
        return from_dmy_string(s, tz)
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


def from_iso_string(s: str, tz: str = 'UTC') -> dt.datetime:
    try:
        d = dt.datetime.fromisoformat(s)
    except ValueError as exc:
        raise ValueError(f'invalid date {s!r}') from exc
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


def from_dmy_string(s: str, tz: str = 'UTC') -> dt.datetime:
    m = re.match(_DMY_RE, s)
    if not m:
        raise ValueError(f'invalid date {s!r}')
    g = m.groupdict()
    try:
        return dt.datetime(int(g['Y']), int(g['m']), int(g['d']), tzinfo=time_zone(tz))
    except ValueError as exc:
        raise ValueError(f'invalid date {s!r}') from exc


def from_timestamp(ts: float, tz: str = 'UTC') -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, tz=time_zone(tz))


# Formatters

def to_iso_string(d: dt.datetime, with_tz='+', sep='T') -> str:
    fmt = f'%Y-%m-%d{sep}%H:%M:%S'
    if with_tz:
        fmt += '%z'
    s = d.strftime(fmt)
    if with_tz == 'Z' and s.endswith('+0000'):
        s = s[:-5] + 'Z'
    return s


def to_iso_date_string(d: dt.datetime) -> str:
    fmt = '%Y-%m-%d'
    return d.strftime(fmt)


def to_int_string(d: dt.datetime) -> str:
    return d.strftime("%Y%m%d%H%M%S")


# Converters

def to_timestamp(d: dt.datetime) -> int:
    return int(d.timestamp())


def to_utc(d: dt.datetime) -> dt.datetime:
    return d.astimezone(time_zone('UTC'))


def to_local(d: dt.datetime) -> dt.datetime:
    return d.astimezone(system_time_zone())


def to_time_zone(d: dt.datetime, tz: str) -> dt.datetime:
    return d.astimezone(time_zone(tz))


# Predicates

def is_date(x) -> bool:
    return isinstance(x, dt.date)


def is_datetime(x) -> bool:
    return isinstance(x, dt.datetime)


def is_utc(d: dt.datetime) -> bool:
    return d.tzinfo == _ZONES['UTC']


def is_local(d: dt.datetime) -> bool:
    return d.tzinfo == system_time_zone()


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


def difference(dt1: dt.datetime, dt2: dt.datetime) -> Diff:
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


def total_difference(dt1: dt.datetime, dt2: dt.datetime) -> Diff:
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


# Wrappers for useful pendulum utilities

def _wrap(d: dt.datetime) -> pendulum.datetime:
    return pendulum.instance(d)


def _unwrap(p: pendulum.datetime):
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


# @formatter:off

def start_of_second(d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).start_of('second'))
def start_of_minute(d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).start_of('minute'))
def start_of_hour  (d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).start_of('hour'))
def start_of_day   (d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).start_of('day'))
def start_of_week  (d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).start_of('week'))
def start_of_month (d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).start_of('month'))
def start_of_year  (d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).start_of('year'))


def end_of_second(d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).end_of('second'))
def end_of_minute(d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).end_of('minute'))
def end_of_hour  (d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).end_of('hour'))
def end_of_day   (d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).end_of('day'))
def end_of_week  (d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).end_of('week'))
def end_of_month (d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).end_of('month'))
def end_of_year  (d: dt.datetime) -> dt.datetime: return _unwrap(_wrap(d).end_of('year'))


def day_of_week   (d: dt.datetime) -> int: return _wrap(d).day_of_week
def day_of_year   (d: dt.datetime) -> int: return _wrap(d).day_of_year
def week_of_month (d: dt.datetime) -> int: return _wrap(d).week_of_month
def week_of_year  (d: dt.datetime) -> int: return _wrap(d).week_of_year
def days_in_month (d: dt.datetime) -> int: return _wrap(d).days_in_month

# @formatter:on


_WD = {
    0: pendulum.WeekDay.MONDAY,
    1: pendulum.WeekDay.TUESDAY,
    2: pendulum.WeekDay.WEDNESDAY,
    3: pendulum.WeekDay.THURSDAY,
    4: pendulum.WeekDay.FRIDAY,
    5: pendulum.WeekDay.SATURDAY,
    6: pendulum.WeekDay.SUNDAY,
}


def next_weekday(d: dt.datetime, weekday: int, keep_time=True) -> dt.datetime:
    return _unwrap(_wrap(d).next(_WD[weekday], keep_time))


def prev_weekday(d: dt.datetime, weekday: int, keep_time=True) -> dt.datetime:
    return _unwrap(_wrap(d).previous(_WD[weekday], keep_time))


# Time

def new_time(hour=0, minute=0, second=0, microsecond=0, tz: str = 'UTC') -> dt.time:
    return dt.time(hour, minute, second, microsecond, tzinfo=time_zone(tz))


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


# Duration

_DURATION_UNITS = {
    'w': 3600 * 24 * 7,
    'd': 3600 * 24,
    'h': 3600,
    'm': 60,
    's': 1,
}


def parse_duration(s: str) -> int:
    """Converts weeks, days, hours or minutes to seconds.

    Args:
        s: Time of duration.

    Returns:
        Input as seconds.
    Raises:
        ``ValueError``: if the duration is invalid.
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
            raise ValueError('invalid duration', s)
        r += p * _DURATION_UNITS[v]
        p = None

    if p:
        r += p

    return r
