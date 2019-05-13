"""Base template"""

import html
import json
import re
import string
import io


class DotDict(dict):
    def __getattr__(self, item):
        return self[item]


class DefaultValue(str):
    def get(self, key):
        return self

    def __getitem__(self, item):
        return self

    def __getattr__(self, item):
        return self


class Formatter(string.Formatter):
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


class Error(ValueError):
    pass


class Base:
    """Basic template"""

    def _render(self, context):
        pass

    def render(self, context, **opts):
        self.errors = []
        self.buf = []
        self.path = ''
        self.default_value = DefaultValue()

        context = self.prepare(context)

        self.pushbuf()
        try:
            self._render(context)
        except Exception as e:
            self.err(0, e)
        res = self.popbuf()

        self.buf = None
        return res

    def prepare(self, context):
        if isinstance(context, dict):
            return context
        if not context:
            return {}
        try:
            return vars(context)
        except Exception as e:
            self.err(0, e)
            return {}

    def put(self, s):
        if s is not None:
            self.buf[-1].append(s)

    def pushbuf(self):
        self.buf.append([])

    def popbuf(self):
        if not self.buf:
            return ''
        try:
            # if we're lucky, there are only strings in the buffer
            s = ''.join(self.buf[-1])
        except TypeError:
            s = ''.join(str(x) for x in self.buf[-1])
        self.buf.pop()
        return s

    defensive = False

    def err(self, lineno, exc):
        if self.defensive:
            self.errors.append('%s in %r:%s' % (repr(exc), self.path, lineno))
        else:
            raise Error('error in %r:%s' % (self.path, lineno)) from exc

    def get(self, obj, prop, lineno):
        try:
            try:
                return obj[prop]
            except KeyError as e:
                self.err(lineno, e)
                return self.default_value
            except TypeError:
                pass
            try:
                return getattr(obj, prop)
            except AttributeError as e:
                self.err(lineno, e)
                return self.default_value
        except Exception as e:
            self.err(lineno, e)
            return self.default_value

    def iter1(self, x):
        if isinstance(x, dict):
            return x.keys()
        if isinstance(x, (list, tuple)):
            return x
        try:
            return vars(x).keys()
        except:
            pass
        return [k for k in x]

    def iter2(self, x):
        if isinstance(x, dict):
            return x.items()
        if isinstance(x, (list, tuple)):
            return list(enumerate(x))
        try:
            return vars(x).items()
        except:
            pass
        return [(k, v) for k, v in x]

    def isempty(self, x):
        if isinstance(x, str):
            return len(x.strip()) == 0
        if isinstance(x, (int, float)):
            return False
        return not bool(x)

    def setpath(self, x):
        self.path = x


class Template(Base):
    """Basic template with some useful filters"""

    def filter_raw(self, val):
        return val

    def filter_as_int(self, val):
        return int(val)

    def filter_as_float(self, val):
        return float(val)

    def filter_as_str(self, val):
        return str(val)

    def filter_html(self, val):
        return html.escape(str(val))

    def filter_h(self, val):
        return html.escape(str(val))

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

    # @TODO need a better expr
    linkify_re = r'https?://\S+[\w/]'

    def filter_linkify(self, val, target=None, cut=None, ell='\u2026'):
        def _repl(m):
            url = m.group(0)

            attr = 'href="%s"' % self.filter_url(url)
            if target:
                attr += ' target="%s"' % target
                if target == '_blank':
                    attr += ' rel="noopener noreferrer"'

            text = url
            if cut:
                text = self.filter_cut(text, cut, ell)

            return f'<a {attr}>{text}</a>'

        return re.sub(self.linkify_re, _repl, str(val))

    def filter_html2(self, val):
        return html.escape(html.unescape(str(val)))

    def filter_format(self, val, fmt):
        if not hasattr(self, '_formatter'):
            self._formatter = Formatter()
        return self._formatter.format(fmt, val)

    def filter_f(self, s, f):
        return f % s

    def filter_cut(self, val, n, ell='\u2026'):
        val = str(val)
        if len(val) <= n:
            return val
        return val[:n] + (ell or '')

    def filter_json(self, val, pretty=False):
        # NB: allow objects to be dumped
        if not pretty:
            return json.dumps(val, default=vars)
        return json.dumps(val, default=vars, indent=4, sort_keys=True)

    def filter_slice(self, val, a, b):
        return val[a:b]
