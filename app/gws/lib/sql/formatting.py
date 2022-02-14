"""SQL Formatting utility."""

import string

import gws
import gws.types as t

_comma = ','.join


class Formatter:
    def __init__(self):
        self.placeholder = ''
        self.always_true = ''

    # see PQescapeInternal @ https://github.com/postgres/postgres/blob/master/src/interfaces/libpq/fe-exec.c

    def _ident(self, s: str) -> str:
        s = s.encode('utf8').decode('utf8')
        if not s:
            raise ValueError('empty identifier')
        for c in s:
            if ord(c) < 32 or ord(c) >= 0xFFFF or c == '\'':
                raise ValueError('invalid character in identifier')
        return '"' + s.replace('"', '""') + '"'

    #

    def format_name(self, arg, vals):
        return self._ident(arg)

    def format_names(self, arg, vals):
        text = []
        for a in arg:
            text.append(self._ident(a))
        return _comma(text)

    def format_qname(self, arg, vals):
        if not isinstance(arg, str):
            a, b = arg
            s = self._ident(b)
            if a:
                s = self._ident(a) + '.' + s
            return s

        if '.' in arg:
            a, b = arg.split('.', 1)
            return self._ident(a) + '.' + self._ident(b)

        return self._ident(arg)

    def format_sql(self, arg, vals):
        if arg is None:
            return ''
        if not isinstance(arg, gws.Sql):
            raise ValueError('Sql instance expected')
        s, v = _apply(self, arg.text, arg.args, arg.kwargs)
        vals.extend(v)
        return s

    def format_value(self, arg, vals):
        if isinstance(arg, gws.Sql):
            return self.format_sql(arg, vals)
        vals.append(arg)
        return self.placeholder

    def format_values(self, arg, vals):
        aa = arg.values() if isinstance(arg, dict) else arg
        text = []
        for a in aa:
            text.append(self.format_value(a, vals))
        return _comma(text)

    def format_int(self, arg, vals):
        return str(int(arg))

    def format_items(self, arg, vals):
        text = []
        for key, val in arg.items():
            text.append(self._ident(key) + '=' + self.format_value(val, vals))
        return _comma(text)

    def format_like(self, arg, vals):
        fmt, s = arg
        s = s.replace('\\', '\\\\')
        s = s.replace('%', '\\%')
        s = s.replace('_', '\\_')
        return self.format_value(fmt.replace('*', s), vals)

    def format_and(self, arg, vals):
        text = []
        for a in arg:
            text.append('(' + self.format_value(a, vals) + ')')
        return ' AND '.join(text)

    def format_or(self, arg, vals):
        text = []
        for a in arg:
            text.append('(' + self.format_value(a, vals) + ')')
        return ' OR '.join(text)

    _keyword = '_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    def format_keyword(self, arg, vals):
        if isinstance(arg, str) and all(c in self._keyword for c in arg):
            return arg
        raise ValueError(f'invalid keyword {arg!r}')


#

class PostgresFormatter(Formatter):
    def __init__(self):
        super().__init__()
        self.placeholder = '%s'
        self.always_true = 'true'


class SqliteFormatter(Formatter):
    def __init__(self):
        super().__init__()
        self.placeholder = '?'
        self.always_true = 'true'


#

_string_formatter = string.Formatter()


def format(formatter, src, args, kwargs):
    try:
        return _apply(formatter, src, args, kwargs)
    except Exception as exc:
        raise ValueError('sql formatting error') from exc


def _apply(formatter, src, args, kwargs):
    auto_num = 0

    text = []
    vals = []

    for literal_text, field_name, format_spec, conversion in _string_formatter.parse(src):

        text.append(literal_text)
        if field_name is None:
            continue

        if not field_name:
            arg = args[auto_num]
            auto_num += 1
        elif field_name.isdigit():
            arg = args[int(field_name)]
        else:
            arg = kwargs[field_name]

        if conversion:
            arg = _string_formatter.convert_field(arg, conversion)

        format_spec = format_spec or 'value'
        s = getattr(formatter, 'format_' + format_spec)(arg, vals)
        if s:
            text.append(s)

    return ''.join(text), vals
