from typing import Optional

import datetime
import calendar
import re
import time

import gws.lib.osx


def set_system_time_zone(tz):
    if not tz:
        return
    if tz != 'UTC' and not re.match(r'^(\w+)/(\w+)$', tz):
        raise ValueError('invald timezone', tz)
    gws.lib.osx.run(['ln', '-fs', f'/usr/share/zoneinfo/{tz}', '/etc/localtime'])


def to_iso_string(d: datetime.datetime, with_tz='+', sep='T') -> str:
    fmt = f'%Y-%m-%d{sep}%H:%M:%S'
    if with_tz:
        fmt += '%z'
    s = d.strftime(fmt)
    if with_tz == 'Z' and s.endswith('+0000'):
        s = s[:-5] + 'Z'
    return s


def to_iso_date_string(d: datetime.datetime) -> str:
    fmt = '%Y-%m-%d'
    return d.strftime(fmt)


def to_iso_local_string(d: datetime.datetime, with_tz='+', sep='T') -> str:
    return to_iso_string(d.astimezone(), with_tz, sep)


def to_timestamp(d: datetime.datetime) -> int:
    return int(calendar.timegm(d.timetuple()))


def to_int_string(d: datetime.datetime) -> str:
    return d.strftime("%Y%m%d%H%M%S")


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def now_iso(with_tz='+', sep='T') -> str:
    return to_iso_string(now(), with_tz, sep)


def to_utc(d: datetime.datetime) -> datetime.datetime:
    return d.astimezone(datetime.timezone.utc)


def from_timestamp(ts) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)


def utime() -> float:
    return time.time()


def timestamp() -> int:
    return int(time.time())


def timestamp_msec() -> int:
    return int(time.time() * 1000)


def is_date(x) -> bool:
    return isinstance(x, datetime.date)


def is_datetime(x) -> bool:
    return isinstance(x, datetime.datetime)


# @TODO

def parse(s) -> Optional[datetime.datetime]:
    if not s:
        return None
    if isinstance(s, datetime.datetime):
        return s
    s = str(s)
    if re.match(r'^\d{4}', s):
        return from_iso(s)
    if re.match(r'^\d{1,2}', s):
        return from_dmy(s)
    return None


_DMY_RE = r'''(?x)
    ^
        (?P<d> \d{1,2})
        [./\s]
        (?P<m> \d{1,2})
        [./\s]
        (?P<Y> \d{2,4})
    $
'''


def from_dmy(s: str) -> Optional[datetime.datetime]:
    m = re.match(_DMY_RE, s)
    if not m:
        raise ValueError(f'invalid date {s!r}')
    g = m.groupdict()
    return datetime.datetime(
        int(g['Y']),
        int(g['m']),
        int(g['d']),
        0, 0, 0,
        tzinfo=datetime.timezone.utc
    )


_ISO_DATE_RE = r'''(?x)
    ^

    # date
    (?P<Y> \d{4}) - (?P<m> \d{1,2}) - (?P<d> \d{1,2})

    # time?
    (
        # separator
        [ T]

        # time
        (?P<H> \d{1,2}) : (?P<M> \d{1,2}) : (?P<S> \d{1,2})

        # fraction?
        (
            \.
            (?P<f> \d+)
        )?

        # time zone?
        (
            Z
            |
            (
                (?P<zsign> [+-]) (?P<zh> \d{2}) :? (?P<zm> \d{2})
            )
        )?

    )?
    $
'''


def from_iso(s: str) -> Optional[datetime.datetime]:
    m = re.match(_ISO_DATE_RE, s)
    if not m:
        raise ValueError(f'invalid date {s!r}')

    g = m.groupdict()
    tz = datetime.timezone.utc

    if g['zsign']:
        sec = int(g['zh']) * 3600 + int(g['zm']) * 60
        if g['zsign'] == '-':
            sec = -sec
        tz = datetime.timezone(datetime.timedelta(seconds=sec))

    return datetime.datetime(
        int(g['Y']),
        int(g['m']),
        int(g['d']),
        int(g['H'] or 0),
        int(g['M'] or 0),
        int(g['S'] or 0),
        int(g['f'] or 0),
        tz
    )


_ISO_TIME_RE = r'''(?x)
    ^
    (?P<H> \d{1,2}) 
    (
        \s* : \s* 
        (?P<M> \d{1,2}) 
        (
            \s* : \s* 
            (?P<S> \d{1,2} )
        )?
    )?
    $
'''


def parse_time(s: str) -> Optional[datetime.time]:
    m = re.match(_ISO_TIME_RE, s)
    if not m:
        raise ValueError(f'invalid time {s!r}')
    g = m.groupdict()
    return datetime.time(
        int(g['H'] or 0),
        int(g['M'] or 0),
        int(g['S'] or 0)
    )


def to_iso_time_string(tt: datetime.time) -> str:
    return '{:02d}:{:02d}:{:02d}'.format(
        tt.hour,
        tt.minute,
        tt.second
    )
