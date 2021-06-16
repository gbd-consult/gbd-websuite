"""Common csv writer helper."""

import gws
import gws.types as t


# @TODO use locale-dependent formatting

class Config(t.WithType):
    """CSV format settings"""

    decimal: str = '.'  #: decimal sign
    delimiter: str = ','  #: field delimiter
    encoding: str = 'utf8'  #: encoding for CSV files
    formulaHack: bool = True  #: prepend numeric strings with an equal sign
    precision: int = 2  #: precision for floats
    quote: str = '"'  #: quote sign
    quoteAll: bool = False #: quote all fields
    rowDelimiter: str = '\n'  #: row delimiter


class Object(gws.Object):
    def configure(self):
        super().configure()

        self.decimal = self.var('decimal')
        self.delimiter = self.var('delimiter')
        self.encoding = self.var('encoding')
        self.formula_hack = self.var('formulaHack')
        self.precision = self.var('precision')
        self.quote = self.var('quote')
        self.quote_all = self.var('quoteAll')
        self.row_delimiter = self.var('rowDelimiter').replace('CR', '\r').replace('LF', '\n')

    def writer(self):
        return _Writer(self)


class _Writer:
    def __init__(self, helper):
        self.h: Object = helper
        self.rows = []
        self.headers = ''

    def write_headers(self, headers: t.List[str]):
        self.headers = self.h.delimiter.join(self._quote(s) for s in headers)
        return self

    def write_attributes(self, attributes: t.List[t.Attribute]):
        self.rows.append(self.h.delimiter.join(self._format(a.value, a.type) for a in attributes))
        return self

    def as_str(self):
        rows = []
        if self.headers:
            rows.append(self.headers)
        rows.extend(self.rows)
        return self.h.row_delimiter.join(rows)

    def as_bytes(self, encoding=None):
        return self.as_str().encode(encoding or self.h.encoding, errors='replace')

    def _format(self, val, type):
        if val is None:
            return self._quote('')

        if type == t.AttributeType.float:
            s = '{:.{prec}f}'.format(float(val), prec=self.h.precision)
            s = s.replace('.', self.h.decimal)
            return self._quote(s) if self.h.quote_all else s

        if type == t.AttributeType.int:
            s = str(val)
            return self._quote(s) if self.h.quote_all else s

        val = gws.as_str(val)

        if val and val.isdigit() and self.h.formula_hack:
            val = '=' + self._quote(val)

        return self._quote(val)

    def _quote(self, val):
        q = self.h.quote
        s = gws.as_str(val).replace(q, q + q)
        return q + s + q
