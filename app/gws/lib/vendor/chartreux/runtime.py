"""Basic runtime"""

import html
import json
import re
import string
import builtins


class Error(ValueError):
    pass


class _Undef(str):
    def get(self, key):
        return self

    def __getitem__(self, item):
        return self

    def __getattr__(self, item):
        return self

    def __bool__(self):
        return False


class _Formatter(string.Formatter):
    def format_field(self, val, spec):
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
    """Basic runtime."""

    undef = _Undef()
    error_class = Error

    def __init__(self):
        self.buf = []

    def prepare(self, context, context_vars):
        if not context:
            context = {}
        elif not isinstance(context, dict):
            try:
                context = vars(context)
            except TypeError:
                context = {}

        return context, vars(builtins)

    def get(self, obj, prop):
        try:
            return obj[prop]
        except:
            pass
        return getattr(obj, prop)

    def iter1(self, x):
        if isinstance(x, (dict, list, tuple, set)):
            return x
        try:
            return [k for k in x]
        except TypeError:
            pass
        return vars(x).keys()

    def iter2(self, x):
        if isinstance(x, dict):
            return x.items()
        if isinstance(x, (list, tuple, set)):
            return list(enumerate(x))
        try:
            return [(k, v) for k, v in x]
        except TypeError:
            pass
        return vars(x).items()

    def isempty(self, x):
        if isinstance(x, str):
            return len(x.strip()) == 0
        if isinstance(x, (int, float)):
            return False
        return not bool(x)

    def pushbuf(self):
        self.buf.append([])

    def popbuf(self):
        b = self.buf.pop()
        try:
            return "".join(b)
        except TypeError:
            return "".join(str(s) for s in b if s is not None)

    def prints(self, s):
        self.buf[-1].append(s)

    def printa(self, *a):
        self.buf[-1].append(' '.join(str(s) for s in a))


class Runtime(BaseRuntime):
    """Basic runtime with default filters"""

    formatter = _Formatter()

    def filter_raw(self, val):
        return val

    def filter_as_int(self, val):
        return int(val)

    def filter_as_float(self, val):
        return float(val)

    def filter_as_str(self, val):
        return str(val)

    def filter_xml(self, val):
        val = str(val)
        val = val.replace('&', '&amp;')
        val = val.replace('<', '&lt;')
        val = val.replace('>', '&gt;')
        return val

    def filter_xmlquote(self, val):
        val = str(val)
        val = val.replace('&', '&amp;')
        val = val.replace('<', '&lt;')
        val = val.replace('>', '&gt;')
        val = val.replace('"', '&quot;')
        val = val.replace('\'', '&apos;')
        return val

    def filter_html(self, val):
        return html.escape(str(val))

    def filter_h(self, val):
        return html.escape(str(val))

    def filter_unhtml(self, val):
        return html.unescape(str(val))

    def filter_nl2br(self, val):
        return str(val).replace('\n', '<br/>')

    def filter_url(self, val):
        # @TODO
        return str(val)

    def filter_strip(self, val):
        return str(val).strip()

    def filter_upper(self, val):
        return str(val).upper()

    def filter_lower(self, val):
        return str(val).lower()

    def filter_title(self, val):
        return str(val).title()

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


    def filter_linkify(self, val, target=None, rel=None, cut=None, ellipsis=None):
        def _repl(m):
            url = m.group(0)

            attr = 'href="{}"'.format(self.filter_url(url))
            if target:
                attr += ' target="{}"'.format(target)
            if rel:
                attr += ' rel="{}"'.format(rel)

            text = url
            if cut:
                text = self.filter_cut(text, cut, ellipsis)

            return f'<a {attr}>{self.filter_html(text)}</a>'

        return re.sub(self.linkify_re, _repl, str(val))

    def filter_format(self, val, fmt):
        return self.formatter.format(fmt, val)

    def filter_cut(self, val, n, ellipsis=None):
        val = str(val)
        if len(val) <= n:
            return val
        return val[:n] + (ellipsis or '')

    def filter_json(self, val, pretty=False):
        # NB: allow objects to be dumped
        if not pretty:
            return json.dumps(val, default=vars)
        return json.dumps(val, default=vars, indent=4, sort_keys=True)

    def filter_slice(self, val, a, b):
        return val[a:b]

    def filter_join(self, val, delim=', '):
        return str(delim).join(str(x) for x in val)

    def filter_split(self, val, delim=None):
        return str(val).split(delim)

    def filter_lines(self, val):
        return str(val).splitlines()

    def filter_sort(self, val):
        return sorted(val)
