import re
import datetime
import babel.dates
from . import shell


def set_system_time_zone(tz):
    if not tz:
        return
    if tz != 'UTC' and not re.match(r'^(\w+)/(\w+)$', tz):
        raise ValueError('invald timezone', tz)
    shell.run(['ln', '-fs', f'/usr/share/zoneinfo/{tz}', '/etc/localtime'])


def to_iso(d):
    return d.strftime("%Y-%m-%d %H:%M:%S")


def now():
    return datetime.datetime.now()


def now_iso():
    return to_iso(now())


def from_timestamp(ts):
    return datetime.datetime.fromtimestamp(ts)


class DateFormatter:
    def __init__(self, locale):
        self.locale = locale

    def _run(self, d, fmt):
        d = babel.dates.parse_date(d, self.locale) if d else datetime.datetime.now()
        return babel.dates.format_date(d, locale=self.locale, format=fmt)

    def short(self, d=None):
        return self._run(d, 'short')

    def medium(self, d=None):
        return self._run(d, 'medium')

    def long(self, d=None):
        return self._run(d, 'long')


class TimeFormatter:
    def __init__(self, locale):
        self.locale = locale

    def _run(self, d, fmt):
        d = babel.dates.parse_time(d, self.locale) if d else datetime.datetime.now()
        return babel.dates.format_time(d, locale=self.locale, format=fmt)

    def short(self, d=None):
        return self._run(d, 'short')

    def medium(self, d=None):
        return self._run(d, 'medium')

    def long(self, d=None):
        return self._run(d, 'long')
