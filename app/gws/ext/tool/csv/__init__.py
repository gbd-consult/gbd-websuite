"""Common csv writer tool."""

import gws
import gws.types as t


# @TODO use locale-dependent formatting

class Config(t.WithType):
    """CSV format settings"""

    decimal: str = '.'  #: decimal sign
    delimiter: str = ','  #: field delimiter
    encoding: str = 'utf8'  #: encoding for CSV files
    precision: int = 2  #: precision for floats
    quote: str = '"'  #: quote sign
    rowDelimiter: str = '\n'  #: row delimiter
    formulaHack: bool = True  #: prepend numeric strings with an equal sign


class Object(gws.Object):
    def configure(self):
        super().configure()

        self.decimal = self.var('decimal')
        self.delimiter = self.var('delimiter')
        self.encoding = self.var('encoding')
        self.precision = self.var('precision')
        self.quote = self.var('quote')
        self.row_delimiter = self.var('rowDelimiter').replace('CR', '\r').replace('LF', '\n')
        self.formula_hack = self.var('formulaHack')

    def writer(self):
        return _Writer(self)


class _Writer:
    def __init__(self, tool):
        self.tool: Object = tool
        self.rows = []
        self.headers = ''

    def write_headers(self, headers: t.List[str]):
        self.headers = self.tool.delimiter.join(self._quote(s) for s in headers)

    def write_attributes(self, attributes: t.List[t.Attribute]):
        self.rows.append(self.tool.delimiter.join(self._format(a.value, a.type) for a in attributes))

    def as_str(self):
        rows = []
        if self.headers:
            rows.append(self.headers)
        rows.extend(self.rows)
        return self.tool.row_delimiter.join(rows)

    def as_bytes(self, encoding=None):
        return self.as_str().encode(encoding or self.tool.encoding)

    def _format(self, val, type):
        if val is None:
            return ''

        if type == t.AttributeType.float:
            s = '{:.{prec}f}'.format(float(val), prec=self.tool.precision)
            return s.replace('.', self.tool.decimal)

        if type == t.AttributeType.int:
            return str(val)

        val = gws.as_str(val)

        if val and val.isdigit() and self.tool.formula_hack:
            q = self.tool.quote
            val = '=' + q + val + q

        return self._quote(val)

    def _quote(self, val):
        q = self.tool.quote
        s = gws.as_str(val).replace(q, q + q)
        return q + s + q
