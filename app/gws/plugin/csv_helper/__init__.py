"""Common csv writer helper."""

from typing import BinaryIO

import decimal
import datetime

import gws
import gws.lib.intl

gws.ext.new.helper('csv')


class FormatConfig(gws.Config):
    """CSV format settings"""

    delimiter: str = ','
    """Field delimiter."""
    encoding: str = 'utf8'
    """Text encoding."""
    formulaHack: bool = True
    """Prepend numeric strings with an equals sign."""
    quote: str = '"'
    """Quote character."""
    quoteAll: bool = False
    """Quote all fields."""
    rowDelimiter: str = 'LF'
    """Row delimiter."""


class Config(gws.Config):
    """CSV helper."""

    format: FormatConfig
    """CSV format settings."""


class Format(gws.Data):
    delimiter: str
    encoding: str
    formulaHack: bool
    quote: str
    quoteAll: bool
    rowDelimiter: str


class Object(gws.Node):
    format: Format

    def configure(self):
        self.format = Format(
            delimiter=self.cfg('format.delimiter', default=','),
            encoding=self.cfg('format.encoding', default='utf8'),
            formulaHack=self.cfg('format.formulaHack', default=True),
            quote=self.cfg('format.quote', default='"'),
            quoteAll=self.cfg('format.quoteAll', default=False),
            rowDelimiter=self.cfg('format.rowDelimiter', default='LF').replace('CR', '\r').replace('LF', '\n'),
        )

    def writer(self, locale: gws.Locale, stream_to: BinaryIO = None) -> '_Writer':
        """Creates a new csv Writer.

        Args:
            locale: Locale to use.
            stream_to: Stream to write to.
        """

        return _Writer(self, locale, stream_to)


class _Writer:
    def __init__(self, helper, locale: gws.Locale, stream_to: BinaryIO = None):
        self.helper: Object = helper
        self.format = self.helper.format
        self.stream_to = stream_to
        self.eol = self.format.rowDelimiter.encode(self.format.encoding)

        self.rows = []
        self.headers = ''

        f = gws.lib.intl.formatters(locale)
        self.dateFormatter = f[0]
        self.timeFormatter = f[1]
        self.numberFormatter = f[2]

    def write_headers(self, headers: list[str]):
        """Writes headers into the header attribute.

        Args:
            headers: Multiple header names.
        """

        self.headers = self.format.delimiter.join(self._quote(s) for s in headers)
        if self.stream_to:
            self.stream_to.write(self.headers.encode(self.format.encoding) + self.eol)
        return self

    def write_row(self, row: list):
        """Writes entries into rows attribute.

        Args:
            row: Row entries.
        """

        s = self.format.delimiter.join(self._format(v) for v in row)
        if self.stream_to:
            self.stream_to.write(s.encode(self.format.encoding) + self.eol)
        else:
            self.rows.append(s)
        return self

    def to_str(self):
        """Converts the headers and rows to a CSV string."""

        rows = []
        if self.headers:
            rows.append(self.headers)
        rows.extend(self.rows)
        return self.format.rowDelimiter.join(rows)

    def to_bytes(self, encoding=None):
        """Converts the table the writer object describes to a CSV byte string."""

        return self.to_str().encode(encoding or self.format.encoding, errors='replace')

    def _format(self, val):
        if val is None:
            return self._quote('')

        if isinstance(val, (float, decimal.Decimal)):
            s = self.numberFormatter.decimal(val)
            return self._quote(s) if self.format.quoteAll else s

        if isinstance(val, int):
            s = str(val)
            return self._quote(s) if self.format.quoteAll else s

        if isinstance(val, (datetime.datetime, datetime.date)):
            s = self.dateFormatter.short(val)
            return self._quote(s)

        if isinstance(val, datetime.time):
            s = self.timeFormatter.short(val)
            return self._quote(s)

        val = gws.u.to_str(val)

        if val and val.isdigit() and self.format.formulaHack:
            val = '=' + self._quote(val)

        return self._quote(val)

    def _quote(self, val):
        q = self.format.quote
        s = gws.u.to_str(val).replace(q, q + q)
        return q + s + q
