"""CSV helper.

The ``csv`` helper writes CSV data with configurable formatting, for example
for the ALKIS export. ``writer`` creates a writer, which writes headers and
rows either into memory or directly into a binary stream.

Values are formatted according to their type:

- ``None`` becomes an empty quoted string,
- integers are written as they are; floats and decimals are formatted with
  the number formatter of the locale. Numbers are only quoted if
  ``quoteAll`` is set,
- dates, datetimes and times are formatted in the short format of the locale
  and quoted,
- other values are converted to strings and quoted. If ``formulaHack`` is
  set, digit-only strings are written as formulas (``="0123"``), so that
  spreadsheet programs keep leading zeros.

The helper is created with default settings if it is not configured.

Example::

    helpers+ {
        type "csv"
        format {
            delimiter ";"
            encoding "cp1252"
            rowDelimiter "CRLF"
        }
    }

Usage in Python::

    helper = cast(gws.plugin.csv_helper.Object, root.app.helper('csv'))
    w = helper.writer(gws.lib.intl.locale('de_DE'))
    w.write_headers(['name', 'area'])
    w.write_row(['Parcel 1', 123.4])
    data = w.to_bytes()
"""

from typing import BinaryIO

import decimal
import datetime

import gws
import gws.lib.intl


class FormatConfig(gws.Config):
    """CSV format settings."""

    delimiter: str = ','
    """Field delimiter."""
    encoding: str = 'utf8'
    """Text encoding."""
    formulaHack: bool = True
    """Write digit-only strings as formulas."""
    quote: str = '"'
    """Quote character."""
    quoteAll: bool = False
    """Quote all fields."""
    rowDelimiter: str = 'LF'
    """Row delimiter."""


@gws.ext.config.helper('csv')
class Config(gws.Config):
    """Format settings for CSV exports."""

    format: FormatConfig
    """CSV format settings."""


class Format(gws.Data):
    """CSV format settings used by the writer."""

    delimiter: str
    """Field delimiter."""
    encoding: str
    """Text encoding."""
    formulaHack: bool
    """Write digit-only strings as formulas."""
    quote: str
    """Quote character."""
    quoteAll: bool
    """Quote all fields, including numbers."""
    rowDelimiter: str
    """Row delimiter, with ``CR`` and ``LF`` replaced by the actual characters."""


@gws.ext.object.helper('csv')
class Object(gws.Node):
    """CSV helper."""

    format: Format
    """Format settings."""

    def configure(self) -> None:
        self.format = Format(
            delimiter=self.cfg('format.delimiter', default=','),
            encoding=self.cfg('format.encoding', default='utf8'),
            formulaHack=self.cfg('format.formulaHack', default=True),
            quote=self.cfg('format.quote', default='"'),
            quoteAll=self.cfg('format.quoteAll', default=False),
            rowDelimiter=self.cfg('format.rowDelimiter', default='LF').replace('CR', '\r').replace('LF', '\n'),
        )

    def writer(self, locale: gws.Locale, stream_to: BinaryIO = None) -> '_Writer':
        """Create a CSV writer.

        Args:
            locale: Locale for formatting numbers, dates and times.
            stream_to: Binary stream to write to. If ``None``, the data is kept in memory.

        Returns:
            A new writer with the format settings of this helper.
        """

        return _Writer(self, locale, stream_to)


class _Writer:
    """CSV writer.

    Writes headers and rows either directly into a binary stream or into
    memory. Data kept in memory is returned by ``to_str`` and ``to_bytes``.
    """

    def __init__(self, helper: 'Object', locale: gws.Locale, stream_to: BinaryIO = None) -> None:
        """Create a CSV writer.

        Args:
            helper: The CSV helper with the format settings.
            locale: Locale for formatting numbers, dates and times.
            stream_to: Binary stream to write to. If ``None``, the data is kept in memory.
        """
        self.helper: Object = helper
        self.format = self.helper.format
        self.stream_to = stream_to
        self.eol = self.format.rowDelimiter.encode(self.format.encoding)

        self.headers = []
        self.str_rows = []
        self.str_headers = ''

        f = gws.lib.intl.formatters(locale)
        self.dateFormatter = f[0]
        self.timeFormatter = f[1]
        self.numberFormatter = f[2]

    def write_headers(self, headers: list[str]) -> '_Writer':
        """Write the header row.

        The headers also define the column order for ``write_dict``.

        Args:
            headers: Column names.

        Returns:
            The writer itself, for chaining.
        """

        self.headers = headers
        self.str_headers = self.format.delimiter.join(self._quote(s) for s in headers)
        if self.stream_to:
            self.stream_to.write(self.str_headers.encode(self.format.encoding) + self.eol)
        return self

    def write_row(self, row: list) -> '_Writer':
        """Write a data row.

        Args:
            row: Values of the row.

        Returns:
            The writer itself, for chaining.
        """

        s = self.format.delimiter.join(self._format(v) for v in row)
        if self.stream_to:
            self.stream_to.write(s.encode(self.format.encoding) + self.eol)
        else:
            self.str_rows.append(s)
        return self

    def write_dict(self, d: dict) -> '_Writer':
        """Write a data row from a dict.

        If no headers are written yet, the keys of the dict are written as
        headers first. Values are taken in the order of the headers, missing
        values are empty.

        Args:
            d: Values by column name.

        Returns:
            The writer itself, for chaining.
        """

        if not self.headers:
            self.write_headers(list(d.keys()))
        return self.write_row([d.get(h, '') for h in self.headers])

    def to_str(self) -> str:
        """Return the data kept in memory as a string.

        Returns:
            The header row and the data rows, joined with the row delimiter.
            When writing into a stream, only the header row is kept in memory.
        """

        rows = []
        if self.headers:
            rows.append(self.str_headers)
        rows.extend(self.str_rows)
        return self.format.rowDelimiter.join(rows)

    def to_bytes(self, encoding: str = None) -> bytes:
        """Return the data kept in memory as bytes.

        Characters that cannot be encoded are replaced.

        Args:
            encoding: Text encoding. If ``None``, the format encoding is used.

        Returns:
            The encoded CSV data.
        """

        return self.to_str().encode(encoding or self.format.encoding, errors='replace')

    def _format(self, val) -> str:
        """Format a value according to its type."""
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

    def _quote(self, val) -> str:
        """Quote a value, doubling the quote characters in it."""
        q = self.format.quote
        s = gws.u.to_str(val).replace(q, q + q)
        return q + s + q
