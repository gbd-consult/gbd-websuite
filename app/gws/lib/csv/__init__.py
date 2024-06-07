"""Common csv writer helper."""

from typing import Optional

import decimal
import datetime

import gws
import gws.lib.intl

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
    """field delimiter"""
    encoding: str
    """encoding for CSV files"""
    formulaHack: bool
    """prepend numeric strings with an equal sign"""
    quote: str
    """quote sign"""
    quoteAll: bool
    """quote all fields"""
    rowDelimiter: str
    """row delimiter"""

    def configure(self):
        """Configures the Objects attributes with the given config.
        """
        self.delimiter = self.cfg('delimiter')
        self.encoding = self.cfg('encoding')
        self.formulaHack = self.cfg('formulaHack')
        self.quote = self.cfg('quote')
        self.quoteAll = self.cfg('quoteAll')
        self.rowDelimiter = self.cfg('rowDelimiter', default='\n').replace('CR', '\r').replace('LF', '\n')

    def writer(self, locale: gws.Locale):
        """Creates a `_Writer` object.

        Args:
            locale: Locale to use.

        Returns:
            A `_Writer` object.
        """

        return _Writer(self, locale)


class _Writer:
    def __init__(self, helper, locale: gws.Locale):
        self.h: Object = helper
        self.rows = []
        self.headers = ''
        f = gws.lib.intl.get_formatters(locale)
        self.dateFormatter = f[0]
        self.timeFormatter = f[1]
        self.numberFormatter = f[2]

    def write_headers(self, headers: list[str]):
        """Writes headers into the header attribute.

        Args:
            headers: Multiple header names.

        Returns:
            The updated `_Writer` object.
        """

        self.headers = self.h.delimiter.join(self._quote(s) for s in headers)
        return self

    def write_row(self, row: list):
        """Writes entries into rows attribute.

        Args:
            row: Row entries.

        Returns:
            The updated `_Writer` object.
        """
        self.rows.append(self.h.delimiter.join(self._format(v) for v in row))
        return self

    def to_str(self):
        """Converts the headers and rows to a string using the given row delimiter.

        Returns:
            A table.
        """
        rows = []
        if self.headers:
            rows.append(self.headers)
        rows.extend(self.rows)
        return self.h.rowDelimiter.join(rows)

    def to_bytes(self, encoding=None):
        """Converts the table the writer object describes to bytes.

        Returns:
            The table as bytes.
        """
        return self.to_str().encode(encoding or self.h.encoding, errors='replace')

    def _format(self, val):
        if val is None:
            return self._quote('')

        if isinstance(val, (float, decimal.Decimal)):
            s = self.numberFormatter.decimal(val)
            return self._quote(s) if self.h.quoteAll else s

        if isinstance(val, int):
            s = str(val)
            return self._quote(s) if self.h.quoteAll else s

        if isinstance(val, (datetime.datetime, datetime.date)):
            s = self.dateFormatter.short(val)
            return self._quote(s)

        if isinstance(val, datetime.time):
            s = self.timeFormatter.short(val)
            return self._quote(s)

        val = gws.u.to_str(val)

        if val and val.isdigit() and self.h.formulaHack:
            val = '=' + self._quote(val)

        return self._quote(val)

    def _quote(self, val):
        q = self.h.quote
        s = gws.u.to_str(val).replace(q, q + q)
        return q + s + q
