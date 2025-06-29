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

    def configure(self) -> None:
        """Configure the CSV helper with format settings from config.

        Sets up the format attribute with values from configuration or defaults.
        """
        self.format = Format(
            delimiter=self.cfg('format.delimiter', default=','),
            encoding=self.cfg('format.encoding', default='utf8'),
            formulaHack=self.cfg('format.formulaHack', default=True),
            quote=self.cfg('format.quote', default='"'),
            quoteAll=self.cfg('format.quoteAll', default=False),
            rowDelimiter=self.cfg('format.rowDelimiter', default='LF').replace('CR', '\r').replace('LF', '\n'),
        )

    def writer(self, locale: gws.Locale, stream_to: BinaryIO = None) -> '_Writer':
        """Creates a new CSV Writer.

        Args:
            locale: Locale to use for formatting values.
            stream_to: Optional binary stream to write to. If None, data is stored in memory.

        Returns:
            A new _Writer instance configured with this helper's format settings.
        """

        return _Writer(self, locale, stream_to)


class _Writer:
    def __init__(self, helper: 'Object', locale: gws.Locale, stream_to: BinaryIO = None) -> None:
        """Initialize a CSV writer.

        Args:
            helper: The CSV helper object containing format settings.
            locale: Locale to use for formatting values.
            stream_to: Optional binary stream to write to. If None, data is stored in memory.
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
        """Writes headers to the CSV output.

        Args:
            headers: List of header column names.

        Returns:
            Self for method chaining.
        """

        self.headers = headers
        self.str_headers = self.format.delimiter.join(self._quote(s) for s in headers)
        if self.stream_to:
            self.stream_to.write(self.str_headers.encode(self.format.encoding) + self.eol)
        return self

    def write_row(self, row: list) -> '_Writer':
        """Writes a row of data to the CSV output.

        Args:
            row: List of values to write as a single row.

        Returns:
            Self for method chaining.
        """

        s = self.format.delimiter.join(self._format(v) for v in row)
        if self.stream_to:
            self.stream_to.write(s.encode(self.format.encoding) + self.eol)
        else:
            self.str_rows.append(s)
        return self

    def write_dict(self, d: dict) -> '_Writer':
        """Writes a dict of data to the CSV output.

        Args:
            d: Dictionary where keys are column names and values are the data.

        Returns:
            Self for method chaining.
        """

        if not self.headers:
            self.write_headers(list(d.keys()))
        return self.write_row([d.get(h, '') for h in self.headers])

    def to_str(self) -> str:
        """Converts the headers and rows to a CSV string.

        Returns:
            A string containing the complete CSV data.
        """

        rows = []
        if self.headers:
            rows.append(self.str_headers)
        rows.extend(self.str_rows)
        return self.format.rowDelimiter.join(rows)

    def to_bytes(self, encoding: str = None) -> bytes:
        """Converts the CSV data to a byte string.

        Args:
            encoding: Optional encoding to use. If None, uses the format's encoding.

        Returns:
            Byte string representation of the CSV data.
        """

        return self.to_str().encode(encoding or self.format.encoding, errors='replace')

    def _format(self, val) -> str:
        """Format a value for CSV output according to its type.

        Args:
            val: The value to format.

        Returns:
            Formatted string representation of the value.
        """
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
        """Quote a value according to CSV quoting rules.

        Doubles any quote characters in the value and wraps the result in quotes.

        Args:
            val: The value to quote.

        Returns:
            Quoted string.
        """
        q = self.format.quote
        s = gws.u.to_str(val).replace(q, q + q)
        return q + s + q
