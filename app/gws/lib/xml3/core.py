import xml.parsers.expat
import gws
import gws.types as t
from . import known_namespaces


def from_path(path, compact_ws=False, keep_ws=False, sort_atts=False, strip_ns=False, to_lower=False) -> gws.XmlElement:
    with open(path, encoding='utf8') as fp:
        xmlstr = fp.read()
    return _Parser(compact_ws, keep_ws, sort_atts, strip_ns, to_lower).parse(xmlstr)


def from_string(xmlstr, compact_ws=False, keep_ws=False, sort_atts=False, strip_ns=False, to_lower=False) -> gws.XmlElement:
    return _Parser(compact_ws, keep_ws, sort_atts, strip_ns, to_lower).parse(xmlstr)


def element(name='', attributes=None, children=None, text='', tail='') -> gws.XmlElement:
    return gws.XmlElement(
        name=name, attributes=attributes or {}, children=children or [], text=text, tail=tail)


def tag(names: str, *args, **kwargs) -> gws.XmlElement:
    root = el = None

    for name in names.split():
        new = element(name)
        if not el:
            root = new
        else:
            el.children.append(new)
        el = new

    if not el:
        raise Error('invalid name for tag', names)

    for arg in args:
        if arg is None:
            continue
        if isinstance(arg, gws.XmlElement):
            el.children.append(arg)
        elif isinstance(arg, str):
            _tag_add_text(el, arg)
        elif isinstance(arg, (int, float, bool)):
            _tag_add_text(el, str(arg).lower())
        elif isinstance(arg, dict):
            for k, v in arg.items():
                if v is not None:
                    el.attributes[k] = v
        else:
            try:
                it = iter(arg)  # type: ignore
            except TypeError:
                raise Error('invalid argument for tag', arg)
            el.children.append(tag(*it))

    for k, v in kwargs.items():
        if v is not None:
            el.attributes[k] = v

    return root


def _tag_add_text(el, s):
    if not s:
        return
    if not el.children:
        el.text += s
    else:
        el.children[-1].tail += s


class Error(Exception):
    pass


def to_string(
        el: gws.XmlElement,
        compact_ws=False,
        keep_ws=False,
        to_lower=False,
        with_xml=False,
        with_xmlns=False,
        with_schemas=False,
) -> str:
    def text(s):
        if isinstance(s, bool):
            s = str(s).lower()
        else:
            s = str(s)
        if not keep_ws:
            s = s.strip()
        if compact_ws:
            s = _compact_ws(s)
        return _encode(s)
        s = s.replace('\n', '&#xa;')
        return s

    def to_str(el, with_xmlns):
        atts = {}
        for key, val in el.attributes.items():
            if val is None:
                continue
            if to_lower:
                key = key.lower()
            atts[key] = text(val)

        if with_xmlns:
            atts.update(
                namespaces.declarations(
                    default=el.attributes.get('xmlns'),
                    for_element=el,
                    with_schemas=with_schemas))

        head_pos = len(buf)
        buf.append('')

        s = text(el.text)
        if s:
            buf.append(s)

        for c in el.children:
            to_str(c, False)

        name = el.name
        if to_lower:
            name = name.lower()

        head = name
        if atts:
            head += ' ' + ' '.join(f'{k}="{v}"' for k, v in atts.items())

        if len(buf) > head_pos + 1:
            buf[head_pos] = '<' + head + '>'
            buf.append('</' + name + '>')
        else:
            buf[head_pos] += '<' + head + '/>'

        s = text(el.tail)
        if s:
            buf.append(s)

    buf = ['']
    to_str(el, with_xmlns)
    if with_xml:
        buf[0] = _XML_DECL
    return ''.join(buf)


##


def attr(el, name, default=None):
    if name in el.attributes:
        return el.attributes[name]
    v = _filter(name, atts=el.attributes)
    if v is not None:
        return v
    return default


def all(el, *paths) -> t.List[gws.XmlElement]:
    if not paths:
        return el.children
    ls = []
    for path in paths:
        ls.extend(_all(el, path))
    return ls


def first(el, *paths) -> t.Optional[gws.XmlElement]:
    if not paths:
        return el.children[0] if el.children else None
    for path in paths:
        a = _all(el, path)
        if a:
            return a[0]


def text(el, path=None) -> str:
    if not path:
        return el.text
    ls = _all(el, path)
    return ls[0].text if ls else ''


def text_list(el, path=None, deep=False) -> t.List[str]:
    def walk(e):
        if e.text:
            buf.append(e.text)
        if deep:
            for c in e.children:
                walk(c)

    buf = []
    rst = all(el, path) if path else el.children
    for e in rst:
        walk(e)
    return buf


def text_dict(el, path=None, deep=False) -> t.Dict[str, str]:
    def walk(e):
        if e.text:
            buf[e.name] = e.text
        if deep:
            for c in e.children:
                walk(c)

    buf = {}
    rst = all(el, path) if path else el.children
    for e in rst:
        walk(e)
    return buf


def iter_all(el: gws.XmlElement):
    yield el
    for c in el.children:
        yield from iter_all(c)


##


def _all(el, path):
    if isinstance(path, str):
        path = path.split('.')
    ls = [el]
    try:
        for p in path:
            if '[' in p:
                p, _, index = p[:-1].partition('[')
                ls = _filter(p, els=ls[0].children)
                ls = [ls[int(index)]]
            else:
                ls = _filter(p, els=ls[0].children)
        return ls
    except (KeyError, IndexError, AttributeError):
        return []


def _filter(name, atts=None, els=None):
    """Filter attributes or a list of elements by name."""

    name = name.lower()
    has_ns = _NSDELIM in name

    # namespace given, full match

    if has_ns and atts is not None:
        for k, v in atts.items():
            if k.lower() == name:
                return v
        return

    if has_ns and els is not None:
        return [el for el in els if el.name.lower() == name]

    # name only, partial match

    if not has_ns and atts is not None:
        nc = ':' + name
        for k, v in atts.items():
            k = k.lower()
            if k == name or k.endswith(nc):
                return v
        return

    if not has_ns and els is not None:
        ls = []
        nc = ':' + name
        for el in els:
            k = el.name.lower()
            if k == name or k.endswith(nc):
                ls.append(el)
        return ls

    raise ValueError('invalid params to _filter')


##

class Namespaces:
    def __init__(self):
        self.pfx_to_ns = {}
        self.ns_to_pfx = {}
        self.schema_location = {}
        self.adhoc_ns_count = 0

        for pfx, ns, sl in known_namespaces.ALL:
            self.add(pfx, ns, sl)

    def add(self, pfx, ns, schema_location=''):
        self.pfx_to_ns[pfx] = ns
        self.ns_to_pfx[ns] = pfx
        if schema_location:
            self.schema_location[ns] = schema_location

    def prefix(self, ns, generate_missing=False):
        pfx = self.ns_to_pfx.get(ns)
        if pfx or not generate_missing:
            return pfx
        self.adhoc_ns_count += 1
        pfx = 'ns' + str(self.adhoc_ns_count)
        self.add(pfx, ns)
        return pfx

    def ns(self, pfx):
        return self.pfx_to_ns.get(pfx)

    def schema(self, ns):
        return self.schema_location.get(ns)

    def declarations(self, ns=None, default=None, with_schemas=True, for_element=None):
        pfxs = set()

        def collect(el):
            pfx, name = _split_ns(el.name)
            if pfx:
                pfxs.add(pfx)
            for k in el.attributes:
                pfx, name = _split_ns(k)
                if pfx:
                    pfxs.add(pfx)
            for c in el.children:
                collect(c)

        if for_element:
            collect(for_element)
        if ns:
            pfxs.update(ns)
        if default:
            pfxs.add(default)

        atts = []
        schemas = []

        for pfx in pfxs:
            ns = self.pfx_to_ns.get(pfx)
            if not ns and pfx in self.ns_to_pfx:
                # ns URI given instead of a prefix?
                ns = pfx
            if not ns:
                raise Error(f'unknown namespace {pfx!r}')
            atts.append((_XMLNS if pfx == default else _XMLNS + ':' + pfx, ns))
            if with_schemas:
                sch = self.schema(ns)
                if sch:
                    schemas.append(ns)
                    schemas.append(sch)

        if schemas:
            atts.append((_XMLNS + ':' + _XSI, _XSI_URL))
            atts.append((_XSI + ':schemaLocation', ' '.join(schemas)))

        return dict(sorted(atts))


namespaces = Namespaces()


##

class _Parser:
    def __init__(
            self,
            compact_ws,
            keep_ws,
            sort_atts,
            strip_ns,
            to_lower,

    ):
        self.p = None

        self.el_stack = []
        self.ns_stack = []

        self.compact_ws = compact_ws
        self.keep_ws = keep_ws
        self.sort_atts = sort_atts
        self.strip_ns = strip_ns
        self.to_lower = to_lower

    def parse(self, xmlstr):
        self.el_stack = [element(children=[])]
        self.ns_stack = [{'': ''}]

        self.p = xml.parsers.expat.ParserCreate()
        self.p.buffer_text = True

        self.p.StartElementHandler = self.StartElementHandler
        self.p.EndElementHandler = self.EndElementHandler
        self.p.CharacterDataHandler = self.CharacterDataHandler

        try:
            self.p.Parse(xmlstr.strip(), True)
        except xml.parsers.expat.ExpatError as e:
            raise Error('parse error: ' + _XML_Error[getattr(e, 'code')], self.p.ErrorLineNumber, self.p.ErrorColumnNumber)

        for el in self.el_stack[-1].children:
            return el

        raise Error('parse error: no element found')

    def _qname(self, pfx, name):
        if self.strip_ns or not pfx:
            return name
        ns = self.ns_stack[-1].get(pfx)
        if ns:
            pfx = namespaces.prefix(ns) or pfx
        return pfx + ':' + name

    def StartElementHandler(self, tag_name, attributes):

        ns_cur = self.ns_stack[-1]
        ns_new = {}

        atts = {}
        unresolved_atts = []

        for attr_name, val in attributes.items():
            if self.to_lower:
                attr_name = attr_name.lower()
            pfx, name = _split_ns(attr_name)
            if not pfx and name == _XMLNS:
                ns_new[''] = val
            elif pfx == _XMLNS:
                ns_new[name] = val
            elif self.strip_ns or not pfx:
                atts[name] = val
            else:
                unresolved_atts.append((pfx, name, val))

        for pfx, name, val in unresolved_atts:
            atts[self._qname(pfx, name)] = val

        if self.sort_atts:
            atts = dict(sorted(atts.items()))

        if ns_new:
            ns_cur = dict(ns_cur)
            ns_cur.update(ns_new)
        self.ns_stack.append(ns_cur)

        if self.to_lower:
            tag_name = tag_name.lower()

        pfx, name = _split_ns(tag_name)
        el = element(name=self._qname(pfx, name), attributes=atts)
        el.pos = [self.p.CurrentLineNumber - 1, self.p.CurrentColumnNumber]

        self.el_stack[-1].children.append(el)
        self.el_stack.append(el)

    def EndElementHandler(self, tag_name):
        self.el_stack.pop()
        self.ns_stack.pop()

    def CharacterDataHandler(self, data):
        stripped = data.strip()
        if not stripped:
            return

        if not self.keep_ws:
            data = stripped
        if self.compact_ws:
            data = _compact_ws(data)

        top = self.el_stack[-1]
        if not top.children:
            top.text += data
        else:
            top.children[-1].tail += data


_NSDELIM = ':'


def _split_ns(s):
    if _NSDELIM not in s:
        return '', s
    a, _, b = s.partition(':')
    return a, b


def _strip_ns(s):
    if _NSDELIM not in s:
        return s
    a, _, b = s.partition(':')
    return b


def _uname(ns, name):
    if not ns:
        return name
    return '{' + ns + '}' + name


def _encode(v) -> str:
    s = str(v)
    s = s.replace("&", "&amp;")
    s = s.replace(">", "&gt;")
    s = s.replace("<", "&lt;")
    s = s.replace('"', "&quot;")
    return s


def _compact_ws(s: str) -> str:
    if not s:
        return s
    lspace = s[0].isspace()
    rspace = s[-1].isspace()
    s = ' '.join(s.strip().split())
    if lspace:
        s = ' ' + s
    if rspace:
        s = s + ' '
    return s


# https://github.com/python/cpython/blob/main/Modules/expat/expat.h

_XML_Error = [
    'XML_ERROR_NONE',
    'XML_ERROR_NO_MEMORY',
    'XML_ERROR_SYNTAX',
    'XML_ERROR_NO_ELEMENTS',
    'XML_ERROR_INVALID_TOKEN',
    'XML_ERROR_UNCLOSED_TOKEN',
    'XML_ERROR_PARTIAL_CHAR',
    'XML_ERROR_TAG_MISMATCH',
    'XML_ERROR_DUPLICATE_ATTRIBUTE',
    'XML_ERROR_JUNK_AFTER_DOC_ELEMENT',
    'XML_ERROR_PARAM_ENTITY_REF',
    'XML_ERROR_UNDEFINED_ENTITY',
    'XML_ERROR_RECURSIVE_ENTITY_REF',
    'XML_ERROR_ASYNC_ENTITY',
    'XML_ERROR_BAD_CHAR_REF',
    'XML_ERROR_BINARY_ENTITY_REF',
    'XML_ERROR_ATTRIBUTE_EXTERNAL_ENTITY_REF',
    'XML_ERROR_MISPLACED_XML_PI',
    'XML_ERROR_UNKNOWN_ENCODING',
    'XML_ERROR_INCORRECT_ENCODING',
    'XML_ERROR_UNCLOSED_CDATA_SECTION',
    'XML_ERROR_EXTERNAL_ENTITY_HANDLING',
    'XML_ERROR_NOT_STANDALONE',
    'XML_ERROR_UNEXPECTED_STATE',
    'XML_ERROR_ENTITY_DECLARED_IN_PE',
    'XML_ERROR_FEATURE_REQUIRES_XML_DTD',
    'XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING',
    'XML_ERROR_UNBOUND_PREFIX',
    'XML_ERROR_UNDECLARING_PREFIX',
    'XML_ERROR_INCOMPLETE_PE',
    'XML_ERROR_XML_DECL',
    'XML_ERROR_TEXT_DECL',
    'XML_ERROR_PUBLICID',
    'XML_ERROR_SUSPENDED',
    'XML_ERROR_NOT_SUSPENDED',
    'XML_ERROR_ABORTED',
    'XML_ERROR_FINISHED',
    'XML_ERROR_SUSPEND_PE',
    'XML_ERROR_RESERVED_PREFIX_XML',
    'XML_ERROR_RESERVED_PREFIX_XMLNS',
    'XML_ERROR_RESERVED_NAMESPACE_URI',
    'XML_ERROR_INVALID_ARGUMENT',
    'XML_ERROR_NO_BUFFER',
    'XML_ERROR_AMPLIFICATION_LIMIT_BREACH',
]

_XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>'
_XMLNS = 'xmlns'
_XSI = 'xsi'
_XSI_URL = 'http://www.w3.org/2001/XMLSchema-instance'
