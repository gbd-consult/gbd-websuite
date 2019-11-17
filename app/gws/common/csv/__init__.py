import gws
import gws.types as t


class Config(t.Config):
    """CSV export settings"""

    decimal: str = '.'
    delimiter: str = ','  #: field delimiter
    encoding: str = 'utf8'  #: encoding for CSV files
    quote: str = '"'  #: quote sign
    quoteAll: bool = False #: quote all fields


class Object(gws.PublicObject):
    def configure(self):
        super().configure()

        self.decimal = self.var('decimal')
        self.delimiter = self.var('delimiter')
        self.encoding = self.var('encoding')
        self.quote = self.var('quote')
        self.quote_all = self.var('quoteAll')

    def write(self, path, headers, rows):
        with open(path, 'wb') as fp:
            if headers:
                self._write_row(fp, headers)
            for r in rows:
                self._write_row(fp, r)

    def _write_row(self, fp, row):
        s = self.delimiter.join(self._field(v) for v in row) + '\r\n'
        b = s.encode(self.encoding)
        fp.write(b)

    def _field(self, s):
        if s is None:
            s = ''

        if self.quote_all:
            s = str(s)

        if isinstance(s, float):
            s = str('%.2f' % s)
            s = s.replace('.', self.decimal)
            return s

        if isinstance(s, int):
            return str(s)

        q = self.quote
        s = str(s).replace(q, q + q)
        return q + s + q
