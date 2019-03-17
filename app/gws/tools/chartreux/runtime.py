"""Basic runtime"""

import html
import json
import re
import string
import builtins


class Error(ValueError):
    pass


class _Undef(str):
    def get(cls, key):
        return cls

    def __getitem__(cls, item):
        return cls

    def __getattr__(cls, item):
        return cls

    def __bool__(self):
        return False


class _Formatter(string.Formatter):
    def format_field(cls, val, spec):
        if spec:
            s = spec[-1]
            if s in 'bcdoxXn':
                val = int(val)
            elif s in 'eEfFgGn%':
                val = float(val)
            elif s == 's':
                val = str(val)
        return format(val, spec)


class BaseRuntime:
    """Basic template"""

    undef = _Undef()
    error_class = Error

    @classmethod
    def prepare(cls, context, context_vars):
        if not context:
            context = {}
        elif not isinstance(context, dict):
            try:
                context = vars(context)
            except TypeError:
                context = {}

        return context, vars(builtins)


    @classmethod
    def get(cls, obj, prop):
        try:
            return obj[prop]
        except:
            pass
        return getattr(obj, prop)

    @classmethod
    def iter1(cls, x):
        if isinstance(x, (dict, list, tuple, set)):
            return x
        try:
            return [k for k in x]
        except TypeError:
            pass
        return vars(x).keys()

    @classmethod
    def iter2(cls, x):
        if isinstance(x, dict):
            return x.items()
        if isinstance(x, (list, tuple, set)):
            return list(enumerate(x))
        try:
            return [(k, v) for k, v in x]
        except TypeError:
            pass
        return vars(x).items()

    @classmethod
    def isempty(cls, x):
        if isinstance(x, str):
            return len(x.strip()) == 0
        if isinstance(x, (int, float)):
            return False
        return not bool(x)


class Runtime(BaseRuntime):
    """Basic template with some useful filters"""

    formatter = _Formatter()

    @classmethod
    def filter_raw(cls, val):
        return val

    @classmethod
    def filter_as_int(cls, val):
        return int(val)

    @classmethod
    def filter_as_float(cls, val):
        return float(val)

    @classmethod
    def filter_as_str(cls, val):
        return str(val)

    @classmethod
    def filter_html(cls, val):
        return html.escape(str(val))

    @classmethod
    def filter_h(cls, val):
        return html.escape(str(val))

    @classmethod
    def filter_unhtml(cls, val):
        return html.unescape(str(val))

    @classmethod
    def filter_nl2br(cls, val):
        return str(val).replace('\n', '<br/>')

    @classmethod
    def filter_url(cls, val):
        # @TODO
        return str(val)

    @classmethod
    def filter_strip(cls, val):
        return str(val).strip()

    @classmethod
    def filter_upper(cls, val):
        return str(val).upper()

    @classmethod
    def filter_lower(cls, val):
        return str(val).lower()

    # @TODO need a better expr
    linkify_re = r'https?://\S+[\w/]'

    @classmethod
    def filter_linkify(cls, val, target=None, cut=None, ellipsis=None):
        def _repl(m):
            url = m.group(0)

            attr = 'href="{}"'.format(cls.filter_url(url))
            if target:
                attr += ' target="{}"'.format(target)
                if target == '_blank':
                    attr += ' rel="noopener noreferrer"'

            text = url
            if cut:
                text = cls.filter_cut(text, cut, ellipsis)

            return f'<a {attr}>{cls.filter_html(text)}</a>'

        return re.sub(cls.linkify_re, _repl, str(val))

    @classmethod
    def filter_format(cls, val, fmt):
        return cls.formatter.format(fmt, val)

    @classmethod
    def filter_cut(cls, val, n, ellipsis=None):
        val = str(val)
        if len(val) <= n:
            return val
        return val[:n] + (ellipsis or '')

    @classmethod
    def filter_json(cls, val, pretty=False):
        # NB: allow objects to be dumped
        if not pretty:
            return json.dumps(val, default=vars)
        return json.dumps(val, default=vars, indent=4, sort_keys=True)

    @classmethod
    def filter_slice(cls, val, a, b):
        return val[a:b]

    @classmethod
    def filter_join(cls, val, delim=', '):
        return str(delim).join(str(x) for x in val)

    @classmethod
    def filter_split(cls, val, delim=None):
        return str(val).split(delim)

    @classmethod
    def filter_lines(cls, val):
        return str(val).splitlines()

    @classmethod
    def filter_sort(cls, val):
        return sorted(val)
