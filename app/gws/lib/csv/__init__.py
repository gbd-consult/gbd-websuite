"""Common csv writer helper."""

import decimal
import datetime

import gws
import gws.lib.intl
import gws.types as t

gws.ext.new.helper('csv')


class Config(gws.Config):
    """CSV format settings"""

    delimiter: str = ','
    """field delimiter"""
    encoding: str = 'utf8'
    """encoding for CSV files"""
    formulaHack: bool = True
    """prepend numeric strings with an equal sign"""
    quote: str = '"'
    """quote sign"""
    quoteAll: bool = False
    """quote all fields"""
    rowDelimiter: str = '\n'
    """row delimiter"""


class Object(gws.Node):
    delimiter: str
    encoding: str
    formulaHack: bool
    quote: str
    quoteAll: bool
    rowDelimiter: str

    def configure(self):
        self.delimiter = self.cfg('delimiter')
        self.encoding = self.cfg('encoding')
        self.formulaHack = self.cfg('formulaHack')
        self.quote = self.cfg('quote')
        self.quoteAll = self.cfg('quoteAll')
        self.rowDelimiter = self.cfg('rowDelimiter', default='\n').replace('CR', '\r').replace('LF', '\n')

    def writer(self, locale_uid: t.Optional[str] = None):
        if not locale_uid and self.root.app.localeUids:
            locale_uid = self.root.app.localeUids[0]
        return _Writer(self, locale_uid)


class _Writer:
    def __init__(self, helper, locale_uid):
        self.h: Object = helper
        self.rows = []
        self.headers = ''
        locale_uid = locale_uid or 'en_CA'
        self.numberFormatter = gws.lib.intl.NumberFormatter(locale_uid)
        self.dateFormatter = gws.lib.intl.DateFormatter(locale_uid)
        self.timeFormatter = gws.lib.intl.TimeFormatter(locale_uid)

    def write_headers(self, headers: list[str]):
        self.headers = self.h.delimiter.join(self._quote(s) for s in headers)
        return self

    def write_row(self, row: list):
        self.rows.append(self.h.delimiter.join(self._format(v) for v in row))
        return self

    def to_str(self):
        rows = []
        if self.headers:
            rows.append(self.headers)
        rows.extend(self.rows)
        return self.h.rowDelimiter.join(rows)

    def to_bytes(self, encoding=None):
        return self.to_str().encode(encoding or self.h.encoding, errors='replace')

    def _format(self, val):
        if val is None:
            return self._quote('')

        if isinstance(val, (float, decimal.Decimal)):
            s = self.numberFormatter.format(gws.lib.intl.NumberFormat.decimal, val)
            return self._quote(s) if self.h.quoteAll else s

        if isinstance(val, int):
            s = str(val)
            return self._quote(s) if self.h.quoteAll else s

        if isinstance(val, (datetime.datetime, datetime.date)):
            s = self.dateFormatter.format(gws.lib.intl.DateTimeFormat.short, val)
            return self._quote(s)

        if isinstance(val, datetime.time):
            s = self.timeFormatter.format(gws.lib.intl.DateTimeFormat.short, val)
            return self._quote(s)

        val = gws.to_str(val)

        if val and val.isdigit() and self.h.formulaHack:
            val = '=' + self._quote(val)

        return self._quote(val)

    def _quote(self, val):
        q = self.h.quote
        s = gws.to_str(val).replace(q, q + q)
        return q + s + q
