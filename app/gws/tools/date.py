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


def to_iso(d):
    return d.strftime("%Y-%m-%d %H:%M:%S")


def to_isotz(d):
    if not d.tzinfo:
        return d.strftime("%Y-%m-%d %H:%M:%S+0000")
    return d.strftime("%Y-%m-%d %H:%M:%S%z")


def utc_from_timestamp(s):
    return datetime.datetime.fromtimestamp(s, tz=datetime.timezone.utc)


def now():
    return datetime.datetime.now()


def now_iso():
    return to_iso(now())


def from_timestamp(ts):
    return datetime.datetime.fromtimestamp(ts)


def utime():
    return time.time()


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
