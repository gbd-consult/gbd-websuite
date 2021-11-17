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
        _tag_add(el, arg)

    _tag_add(el, kwargs)

    return root


def _tag_add(el, arg):
    if arg is None:
        return

    if isinstance(arg, gws.XmlElement):
        el.children.append(arg)
        return

    if isinstance(arg, str):
        if arg:
            _tag_add_text(el, arg)
        return

    if isinstance(arg, (int, float, bool)):
        _tag_add_text(el, str(arg).lower())
        return

    if isinstance(arg, dict):
        for k, v in arg.items():
            if v is not None:
                el.attributes[k] = v
        return

    try:
        ls = arg if isinstance(arg, (list, tuple)) else list(arg)
    except TypeError as exc:
        raise Error('invalid argument for tag', arg) from exc

    if ls and isinstance(ls[0], str):
        el.children.append(tag(*ls))
        return

    for arg2 in ls:
        _tag_add(el, arg2)


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
        return encode(s)

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
        self._pfx_to_uri = {}
        self._uri_to_pfx = {}
        self._schema = {}
        self._adhoc_ns_count = 0

        for pfx, uri, schema in known_namespaces.ALL:
            self.add(pfx, uri, schema)

    def add(self, pfx, uri, schema=''):
        self._pfx_to_uri[pfx] = uri
        self._uri_to_pfx[uri] = pfx
        if schema:
            self._schema[uri] = self._schema[pfx] = schema

    def prefix(self, uri, generate_missing=False):
        pfx = self._uri_to_pfx.get(uri)
        if pfx or not generate_missing:
            return pfx
        self._adhoc_ns_count += 1
        pfx = 'ns' + str(self._adhoc_ns_count)
        self.add(pfx, uri)
        return pfx

    def uri(self, pfx):
        return self._pfx_to_uri.get(pfx)

    def schema(self, uri_or_pfx):
        return self._schema.get(uri_or_pfx)

    def declarations(self, namespaces=None, default=None, with_schemas=True, for_element=None):
        pset = set()

        if for_element:
            _collect_ns_prefixes(for_element, pset)
        if namespaces:
            pset.update(namespaces)
        if default:
            pset.add(default)

        atts = []
        schemas = []

        for pfx in pset:
            uri = self._pfx_to_uri.get(pfx)
            if not uri and pfx in self._uri_to_pfx:
                # ns URI given instead of a prefix?
                uri = pfx
            if not uri:
                raise Error(f'unknown namespace {pfx!r}')
            atts.append((_XMLNS if pfx == default else _XMLNS + ':' + pfx, uri))
            if with_schemas:
                sch = self._schema.get(uri)
                if sch:
                    schemas.append(uri)
                    schemas.append(sch)

        if schemas:
            atts.append((_XMLNS + ':' + _XSI, _XSI_URL))
            atts.append((_XSI + ':schemaLocation', ' '.join(schemas)))

        return dict(sorted(atts))


namespaces = Namespaces()


def _collect_ns_prefixes(el, prefixes):
    pfx, name = split_name(el.name)
    if pfx:
        prefixes.add(pfx)
    for k in el.attributes:
        pfx, name = split_name(k)
        if pfx:
            prefixes.add(name if pfx == _XMLNS else pfx)
    for c in el.children:
        _collect_ns_prefixes(c, prefixes)


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
            pfx, name = split_name(attr_name)
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

        pfx, name = split_name(tag_name)
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


##


_NSDELIM = ':'


def split_name(s):
    if _NSDELIM not in s:
        return '', s
    a, _, b = s.partition(':')
    return a, b


def unqualify_name(s):
    if _NSDELIM not in s:
        return s
    _, _, b = s.partition(':')
    return b


def qualify_name(s, prefix):
    if _NSDELIM not in s:
        return prefix + _NSDELIM + s
    return s


def requalify_name(s, prefix):
    if _NSDELIM not in s:
        return prefix + _NSDELIM + s
    _, _, b = s.partition(':')
    return prefix + _NSDELIM + s


##

def encode(v) -> str:
    s = str(v)
    s = s.replace("&", "&amp;")
    s = s.replace(">", "&gt;")
    s = s.replace("<", "&lt;")
    s = s.replace('"', "&quot;")
    return s


##

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


##

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
