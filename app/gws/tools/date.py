from typing import Optional
import re
import datetime
import time
import babel.dates

import gws.tools.os2


def set_system_time_zone(tz):
    if not tz:
        return
    if tz != 'UTC' and not re.match(r'^(\w+)/(\w+)$', tz):
        raise ValueError('invald timezone', tz)
    gws.tools.os2.run(['ln', '-fs', f'/usr/share/zoneinfo/{tz}', '/etc/localtime'])


def to_iso(d: datetime.datetime, with_tz='+', sep='T') -> str:
    fmt = f'%Y-%m-%d{sep}%H:%M:%S'
    if with_tz:
        fmt += '%z'
    s = d.strftime(fmt)
    if with_tz == 'Z' and s.endswith('+0000'):
        s = s[:-5] + 'Z'
    return s


def to_iso_date(d: datetime.datetime) -> str:
    fmt = '%Y-%m-%d'
    return d.strftime(fmt)


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def now_iso(with_tz='+', sep='T') -> str:
    return to_iso(now(), with_tz, sep)


def to_utc(d: datetime.datetime) -> datetime.datetime:
    return d.astimezone(datetime.timezone.utc)


def from_timestamp(ts) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)


def utime() -> float:
    return time.time()


def timestamp() -> int:
    return int(time.time())


def is_date(x) -> bool:
    return isinstance(x, datetime.date)


def is_datetime(x) -> bool:
    return isinstance(x, datetime.datetime)


_iso_re = r'''(?x)
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
    m = re.match(_iso_re, s)
    if not m:
        return None

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


class DateFormatter:
    def __init__(self, locale):
        self.locale = locale

    def format(self, fmt, d=None):
        d = babel.dates.parse_date(d, self.locale) if d else datetime.datetime.now()
        return babel.dates.format_date(d, locale=self.locale, format=fmt)

    @property
    def short(self):
        return self.format('short')

    @property
    def medium(self):
        return self.format('medium')

    @property
    def long(self):
        return self.format('medium')


class TimeFormatter:
    def __init__(self, locale):
        self.locale = locale

    def format(self, fmt, d=None):
        d = babel.dates.parse_time(d, self.locale) if d else datetime.datetime.now()
        return babel.dates.format_time(d, locale=self.locale, format=fmt)

    @property
    def short(self):
        return self.format('short')

    @property
    def medium(self):
        return self.format('medium')

    @property
    def long(self):
        return self.format('medium')
