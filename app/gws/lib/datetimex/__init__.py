"""Date and time utilities.

These utilities are wrappers around the `datetime` module,
some functions also use `pendulum` (https://pendulum.eustace.io/),
however we work strictly with stock ``datetime.datetime`` and ``.time`` objects here.

"Naive" datetime objects are not supported. All objects constructed in this module
have the ``tzinfo`` attribute. When constructing an object (e.g. from a string),
the default time zone must be passed in as a zoneinfo string (like ``Europe/Berlin``).
An empty string (default) means the local time zone.

It is an error to pass a naive object as an argument.

When running in a docker container, there are several ways to set up the local time zone:

- by setting the config variable ``server.timeZone`` (see `gws.config.parser`)
- by setting the ``TZ`` environment variable
- mounting a host zone info file to ``/etc/localtime``
"""

from typing import Optional

import datetime as dt
import re
import os
import zoneinfo

import pendulum
import pendulum.parsing

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
    new_zi = time_zone(tz)
    cur_zi = _zone_info_from_localtime()

    gws.log.debug(f'set_local_time_zone: cur={cur_zi} new={new_zi}')

    if new_zi == cur_zi:
        return
    _set_localtime_from_zone_info(new_zi)

    gws.log.debug(f'set_local_time_zone: cur={_zone_info_from_localtime()}')


def time_zone(tz: str = '') -> zoneinfo.ZoneInfo:
    if tz in _ZI_CACHE:
        return _ZI_CACHE[tz]

    if not tz:
        _ZI_CACHE[''] = _zone_info_from_localtime()
        return _ZI_CACHE['']

    return _zone_info_from_string(tz)


def _set_localtime_from_zone_info(zi):
    if os.getuid() != 0:
        gws.log.warning('time zone: cannot set timezone, must be root')
        return
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
    _set_localtime_from_zone_info(
        _zone_info_from_string(os.environ['TZ']))


# Constructors


def new(year, month, day, hour=0, minute=0, second=0, microsecond=0, fold=0, tz: str = '') -> dt.datetime:
    return dt.datetime(year, month, day, hour, minute, second, microsecond, fold=fold, tzinfo=time_zone(tz))


def now(tz: str = '') -> dt.datetime:
    return dt.datetime.now(time_zone(tz))


def now_utc() -> dt.datetime:
    return dt.datetime.now(time_zone('UTC'))


def today(tz: str = '') -> dt.datetime:
    return now(tz).replace(hour=0, minute=0, second=0, microsecond=0)


def today_utc() -> dt.datetime:
    return now_utc().replace(hour=0, minute=0, second=0, microsecond=0)


def parse(s, tz: str = '') -> Optional[dt.datetime]:
    if not s:
        return None

    if isinstance(s, dt.datetime):
        return _ensure_tzinfo(s, tz)

    if isinstance(s, dt.date):
        return dt.datetime(s.year, s.month, s.day, tzinfo=time_zone(tz))

    try:
        return from_string(s, tz)
    except Error:
        pass


def from_string(s: str, tz: str = '') -> dt.datetime:
    return _pend_parse(s, tz, iso_only=False)


def from_iso_string(s: str, tz: str = '') -> dt.datetime:
    return _pend_parse(s, tz, iso_only=True)


def from_timestamp(ts: float, tz: str = '') -> dt.datetime:
    return dt.datetime.fromtimestamp(ts, tz=time_zone(tz))


# Formatters

def to_iso_string(d: Optional[dt.datetime] = None, with_tz='+', sep='T') -> str:
    fmt = f'%Y-%m-%d{sep}%H:%M:%S'
    if with_tz:
        fmt += '%z'
    s = _check(d).strftime(fmt)
    if with_tz == 'Z' and s.endswith('+0000'):
        s = s[:-5] + 'Z'
    return s


def to_iso_date_string(d: Optional[dt.datetime] = None) -> str:
    return _check(d).strftime('%Y-%m-%d')


def to_basic_string(d: Optional[dt.datetime] = None) -> str:
    return _check(d).strftime("%Y%m%d%H%M%S")


# Converters

def to_timestamp(d: Optional[dt.datetime] = None) -> int:
    return int(_check(d).timestamp())


def to_millis(d: Optional[dt.datetime] = None) -> int:
    return int(_check(d).timestamp() * 1000)


def to_utc(d: Optional[dt.datetime] = None) -> dt.datetime:
    return _check(d).astimezone(time_zone('UTC'))


def to_local(d: Optional[dt.datetime] = None) -> dt.datetime:
    return _check(d).astimezone(time_zone(''))


def to_time_zone(tz: str, d: Optional[dt.datetime] = None) -> dt.datetime:
    return _check(d).astimezone(time_zone(tz))


# Predicates

def is_date(x) -> bool:
    return isinstance(x, dt.date)


def is_datetime(x) -> bool:
    return isinstance(x, dt.datetime)


def is_utc(d: dt.datetime) -> bool:
    return _zone_info_from_tzinfo(_check(d).tzinfo) == time_zone('utc')


def is_local(d: dt.datetime) -> bool:
    return _zone_info_from_tzinfo(_check(d).tzinfo) == time_zone('')


# Arithmetic

def add(
        d: Optional[dt.datetime] = None,
        years=0, months=0, days=0, weeks=0, hours=0, minutes=0, seconds=0, microseconds=0
) -> dt.datetime:
    return pendulum.helpers.add_duration(
        _check(d),
        years=years, months=months, days=days,
        weeks=weeks, hours=hours, minutes=minutes,
        seconds=seconds, microseconds=microseconds
    )


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


def difference(d1: dt.datetime, d2: Optional[dt.datetime] = None) -> Diff:
    iv = pendulum.Interval(_check(d1), _check(d2), absolute=False)
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


def total_difference(d1: dt.datetime, d2: Optional[dt.datetime] = None) -> Diff:
    iv = pendulum.Interval(_check(d1), _check(d2), absolute=False)
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

# @formatter:off

def start_of_second(d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).start_of('second'))
def start_of_minute(d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).start_of('minute'))
def start_of_hour  (d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).start_of('hour'))
def start_of_day   (d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).start_of('day'))
def start_of_week  (d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).start_of('week'))
def start_of_month (d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).start_of('month'))
def start_of_year  (d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).start_of('year'))


def end_of_second(d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).end_of('second'))
def end_of_minute(d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).end_of('minute'))
def end_of_hour  (d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).end_of('hour'))
def end_of_day   (d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).end_of('day'))
def end_of_week  (d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).end_of('week'))
def end_of_month (d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).end_of('month'))
def end_of_year  (d: Optional[dt.datetime] = None) -> dt.datetime: return _unpend(_pend(d).end_of('year'))


def day_of_week   (d: Optional[dt.datetime] = None) -> int: return _pend(d).day_of_week
def day_of_year   (d: Optional[dt.datetime] = None) -> int: return _pend(d).day_of_year
def week_of_month (d: Optional[dt.datetime] = None) -> int: return _pend(d).week_of_month
def week_of_year  (d: Optional[dt.datetime] = None) -> int: return _pend(d).week_of_year
def days_in_month (d: Optional[dt.datetime] = None) -> int: return _pend(d).days_in_month


# @formatter:on

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


def next(day: int | str, d: Optional[dt.datetime] = None, keep_time=False) -> dt.datetime:
    return _unpend(_pend(d).next(_WD[day], keep_time))


def prev(day: int | str, d: Optional[dt.datetime] = None, keep_time=False) -> dt.datetime:
    return _unpend(_pend(d).previous(_WD[day], keep_time))


# Time

def new_time(hour=0, minute=0, second=0, microsecond=0, fold=0, tz: str = '') -> dt.time:
    return dt.time(hour, minute, second, microsecond, fold=fold, tzinfo=time_zone(tz))


def parse_time(s, tz: str = '') -> Optional[dt.time]:
    if isinstance(s, dt.datetime):
        return _ensure_tzinfo(s.timetz(), tz)
    if isinstance(s, dt.time):
        return _ensure_tzinfo(s, tz)
    try:
        return time_from_iso_string(str(s), tz)
    except Error:
        pass


def time_from_iso_string(s: str, tz: str = '') -> dt.time:
    return _pend_parse_time(s, tz, iso_only=True)


def time_to_iso_string(t: Optional[dt.time] = None, with_tz='+') -> str:
    fmt = '%H:%M:%S'
    if with_tz:
        fmt += '%z'
    s = _check_time(t).strftime(fmt)
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
            raise Error('invalid duration', s)
        r += p * _DURATION_UNITS[v]
        p = None

    if p:
        r += p

    return r


##

def _check(d: dt.datetime | None) -> dt.datetime:
    if d is None:
        return now()
    if isinstance(d, dt.datetime):
        if not d.tzinfo:
            raise Error('naive datetime detected')
        return d
    raise Error('invalid datetime value')


def _check_time(t: dt.time | None) -> dt.time:
    if t is None:
        return now().timetz()
    if isinstance(t, dt.time):
        if not t.tzinfo:
            raise Error('naive time detected')
        return t
    raise Error('invalid time value')


# pendulum.DateTime <-> python datetime

def _pend(d: dt.datetime | None) -> pendulum.DateTime:
    return pendulum.instance(_check(d))


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


def _pend_parse(s, tz, iso_only):
    try:
        if iso_only:
            d = pendulum.parsing.parse_iso8601(s, tz=time_zone(tz))
        else:
            # do not normalize
            d = pendulum.parsing._parse(s)
    except (ValueError, pendulum.parsing.ParserError) as exc:
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
            d = pendulum.parsing.parse_iso8601(s, tz=time_zone(tz))
        else:
            # do not normalize
            d = pendulum.parsing._parse(s)
    except (ValueError, pendulum.parsing.ParserError) as exc:
        raise Error(f'invalid time {s!r}') from exc

    if isinstance(d, dt.time):
        return _ensure_tzinfo(d, tz)

    # times and durations not accepted
    raise Error(f'invalid date {s!r}')


def _ensure_tzinfo(d: dt.datetime | dt.time, tz: str):
    if not d.tzinfo:
        return d.replace(tzinfo=time_zone(tz))

    # try to convert 'their' tzinfo (might be an unnamed dt.timezone or pendulum.FixedTimezone) to zoneinfo
    zi = _zone_info_from_tzinfo(d.tzinfo)
    if zi:
        return d.replace(tzinfo=zi)

    # failing that, keep existing tzinfo
    return d
