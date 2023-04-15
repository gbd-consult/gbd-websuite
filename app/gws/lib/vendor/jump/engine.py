"""Basic runtime"""

import html
import json
import re
import string
import builtins
from collections import abc

from . import compiler

builtins_dct = {k: getattr(builtins, k) for k in dir(builtins)}


class RuntimeError(ValueError):
    def __init__(self, message, path: str, line: int):
        self.path = path
        self.line = line
        message += ' in ' + path + ':' + str(line)
        super().__init__(message)
        self.message = message


class Environment:
    def __init__(self, engine, paths, args, errorhandler):
        self.engine = engine
        self.buf = []
        self.haserr = False
        self.paths = paths
        self.ARGS = self.prepare(args)

        self.engine_functions = {}

        for a in dir(self.engine):
            if '_' in a:
                cmd, _, name = a.partition('_')
                if cmd in compiler.C.DEF_COMMANDS:
                    self.engine_functions[name] = getattr(self.engine, a)

        if errorhandler:
            def err(exc, pos):
                try:
                    ok = errorhandler(exc, self.paths[pos[0]], pos[1], self)
                except:
                    self.haserr = True
                    raise
                if not ok:
                    self.haserr = True
                    raise
        else:
            def err(exc, pos):
                self.haserr = True
                if isinstance(exc, RuntimeError):
                    raise exc
                raise RuntimeError(str(exc), self.paths[pos[0]], pos[1]) from exc
        self.error = err

    def pushbuf(self):
        self.buf.append([])

    def popbuf(self):
        return ''.join(str(s) for s in self.buf.pop())

    def echo(self, s):
        if s is not None:
            self.buf[-1].append(s)

    def print(self, *args, end='\n'):
        self.buf[-1].append(' '.join(str(s) for s in args if s is not None) + end)

    def get(self, name):
        if name in self.ARGS:
            return self.ARGS[name]
        if name in self.engine_functions:
            return self.engine_functions[name]
        if name in builtins_dct:
            return builtins_dct[name]
        raise NameError(f'name {name!r} is not defined')

    def attr(self, obj, prop):
        try:
            return obj[prop]
        except:
            return getattr(obj, prop)

    def attrs(self, obj, props):
        for prop in props:
            obj = self.attr(obj, prop)
        return obj

    def iter(self, arg, size):
        if not arg:
            return ''

        if size == 1:
            if isinstance(arg, abc.Collection):
                return arg
            try:
                return [k for k in arg]
            except TypeError:
                pass
            return vars(arg)

        if size == 2:
            if isinstance(arg, abc.Mapping):
                return list(arg.items())
            try:
                return [k for k in arg]
            except TypeError:
                pass
            return list(vars(arg).items())

        try:
            return [k for k in arg]
        except TypeError:
            pass
        return vars(arg)

    def isempty(self, x):
        if isinstance(x, str):
            return len(x.strip()) == 0
        if isinstance(x, (int, float)):
            return False
        return not bool(x)

    def prepare(self, args):
        if isinstance(args, dict):
            return args
        if not args:
            return {}
        try:
            return vars(args)
        except TypeError:
            return {}


class BaseEngine:
    """Basic runtime."""

    def environment(self, paths, args, errorhandler):
        return Environment(self, paths, args, errorhandler)

    def parse(self, text, **options):
        return compiler.do('parse', self, options, text, None)

    def parse_path(self, path, **options):
        return compiler.do('parse', self, options, None, path)

    def translate(self, text, **options):
        return compiler.do('translate', self, options, text, None)

    def translate_path(self, path, **options):
        return compiler.do('translate', self, options, None, path)

    def compile(self, text, **options):
        return compiler.do('compile', self, options, text, None)

    def compile_path(self, path, **options):
        return compiler.do('compile', self, options, None, path)

    def call(self, template_fn, args=None, error=None):
        return template_fn(self, args, error)

    def render(self, text, args=None, error=None, **options):
        template_fn = self.compile(text, **options)
        return self.call(template_fn, args, error)

    def render_path(self, path, args=None, error=None, **options):
        template_fn = self.compile_path(path, **options)
        return self.call(template_fn, args, error)


class Engine(BaseEngine):
    """Basic runtime with default filters"""

    def def_raw(self, val):
        return _str(val)

    def def_safe(self, val):
        return _str(val)

    def def_as_int(self, val):
        return int(val)

    def def_as_float(self, val):
        return float(val)

    def def_as_str(self, val):
        if isinstance(val, bytes):
            return val.decode('utf8')
        return _str(val)

    def def_xml(self, val):
        return _xml(val, False)

    def def_xmlq(self, val):
        return _xml(val, True)

    def def_html(self, val):
        return _xml(val, False)

    def def_htmlq(self, val):
        return _xml(val, True)

    def def_h(self, val):
        return _xml(val, False)

    def def_unhtml(self, val):
        return html.unescape(str(val))

    def def_nl2br(self, val):
        return _str(val).replace('\n', '<br/>')

    def def_nl2p(self, val):
        s = re.sub(r'\n[ \t]*\n\s*', '\0', _str(val))
        return '\n'.join(f'<p>' + p.strip() + '</p>' for p in s.split('\0'))

    def def_url(self, val):
        # @TODO
        return _str(val)

    def def_strip(self, val):
        return _str(val).strip()

    def def_upper(self, val):
        return _str(val).upper()

    def def_lower(self, val):
        return _str(val).lower()

    def def_titlecase(self, val):
        return _str(val).title()

    # based on: https://daringfireball.net/2010/07/improved_regex_for_matching_urls
    linkify_re = r'''(?xi)
        \b
        (
            https?://
            |
            www\d?\.
        )
        (
            [^\s()<>{}\[\]]
            |
            \( [^\s()]+ \)
        )+
        (
            \( [^\s()]+ \)
            |
            [^\s`!()\[\]{};:'".,<>?«»“”‘’]
        )
    '''

    def def_linkify(self, val, target=None, rel=None, cut=None, ellipsis=None):
        def _repl(m):
            url = m.group(0)

            attr = 'href="{}"'.format(self.def_url(url))
            if target:
                attr += ' target="{}"'.format(target)
            if rel:
                attr += ' rel="{}"'.format(rel)

            text = url
            if cut:
                text = self.def_cut(text, cut, ellipsis)

            return f'<a {attr}>{_xml(text)}</a>'

        return re.sub(self.linkify_re, _repl, str(val))

    def def_format(self, val, fmt):
        if fmt[0] not in ':!':
            fmt = ':' + fmt
        return _formatter.format('{' + fmt + '}', val)

    def def_cut(self, val, n, ellip=None):
        val = _str(val)
        if len(val) <= n:
            return val
        return val[:n] + (ellip or '')

    def def_shorten(self, val, n, ellip=None):
        val = _str(val)
        if len(val) <= n:
            return val
        return val[:(n + 1) >> 1] + (ellip or '') + val[-(n >> 1):]

    def def_json(self, val, pretty=False):

        def dflt(obj):
            try:
                return vars(obj)
            except TypeError:
                return str(obj)

        if not pretty:
            return json.dumps(val, default=dflt)
        return json.dumps(val, default=dflt, indent=4, sort_keys=True)

    def def_slice(self, val, a, b):
        return val[a:b]

    def def_join(self, val, delim=''):
        return str(delim).join(_str(x) for x in val)

    def def_spaces(self, val):
        return ' '.join(_str(x).strip() for x in val)

    def def_commas(self, val):
        return ','.join(_str(x) for x in val)

    def def_split(self, val, delim=None):
        return _str(val).split(delim)

    def def_lines(self, val, strip=False):
        s = _str(val).split('\n')
        if strip:
            return [ln.strip() for ln in s]
        return s

    def def_sort(self, val):
        return sorted(val)


##


class _Formatter(string.Formatter):
    def format_field(self, val, spec):
        if spec:
            s = spec[-1]
            if s in 'bcdoxXn':
                val = int(val)
            elif s in 'eEfFgGn%':
                val = float(val)
            elif s == 's':
                val = _str(val)
        return format(val, spec)


_formatter = _Formatter()


def _str(x):
    return '' if x is None else str(x)


def _xml(x, quote=False):
    x = _str(x)
    x = x.replace('&', '&amp;')
    x = x.replace('<', '&lt;')
    x = x.replace('>', '&gt;')
    if quote:
        x = x.replace('"', '&#x22;')
        x = x.replace('\'', '&#x27;')
    return x
