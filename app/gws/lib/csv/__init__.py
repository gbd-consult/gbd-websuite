"""Common csv writer helper."""

import gws
import gws.types as t

gws.ext.new.helper('csv')


# @TODO use locale-dependent formatting

class Config(gws.Config):
    """CSV format settings"""

    decimal: str = '.' 
    """decimal sign"""
    delimiter: str = ',' 
    """field delimiter"""
    encoding: str = 'utf8' 
    """encoding for CSV files"""
    formulaHack: bool = True 
    """prepend numeric strings with an equal sign"""
    precision: int = 2 
    """precision for floats"""
    quote: str = '"' 
    """quote sign"""
    quoteAll: bool = False 
    """quote all fields"""
    rowDelimiter: str = '\n' 
    """row delimiter"""



class Object(gws.Node):
    def configure(self):
        self.decimal = self.var('decimal')
        self.delimiter = self.var('delimiter')
        self.encoding = self.var('encoding')
        self.formula_hack = self.var('formulaHack')
        self.precision = self.var('precision')
        self.quote = self.var('quote')
        self.quote_all = self.var('quoteAll')
        self.row_delimiter = self.var('rowDelimiter', default='\n').replace('CR', '\r').replace('LF', '\n')

    def writer(self):
        return _Writer(self)


class _Writer:
    def __init__(self, helper):
        self.h: Object = helper
        self.rows = []
        self.headers = ''

    def write_headers(self, headers: list[str]):
        self.headers = self.h.delimiter.join(self._quote(s) for s in headers)
        return self

    def write_attributes(self, attributes: list[gws.Attribute]):
        self.rows.append(self.h.delimiter.join(self._format(a.value, a.type) for a in attributes))
        return self

    def to_str(self):
        rows = []
        if self.headers:
            rows.append(self.headers)
        rows.extend(self.rows)
        return self.h.row_delimiter.join(rows)

    def to_bytes(self, encoding=None):
        return self.to_str().encode(encoding or self.h.encoding, errors='replace')

    def _format(self, val, type):
        if val is None:
            return self._quote('')

        if type == gws.AttributeType.float:
            s = '{:.{prec}f}'.format(float(val), prec=self.h.precision)
            s = s.replace('.', self.h.decimal)
            return self._quote(s) if self.h.quote_all else s

        if type == gws.AttributeType.int:
            s = str(val)
            return self._quote(s) if self.h.quote_all else s

        val = gws.to_str(val)

        if val and val.isdigit() and self.h.formula_hack:
            val = '=' + self._quote(val)

        return self._quote(val)

    def _quote(self, val):
        q = self.h.quote
        s = gws.to_str(val).replace(q, q + q)
        return q + s + q
