"""XML parser and generator.

The parser is expat-based. It returns `Element` instances.

The generator (`as_string`) accepts a Tag, which is simply a tuple of tag name, attributes (dict) and subtags (Tags).
"""

import xml.parsers.expat

import gws
import gws.types as t

from . import helper
from .types import Namespace


class Error(Exception):
    pass


def tag(*args) -> gws.Tag:
    return args


def to_string(p) -> str:
    if _is_list(p):
        return _string(p)
    if _is_elem(p):
        return _string(p.to_tag())
    return str(p)


def qname(namespace, tag):
    if namespace:
        return namespace + ':' + tag
    return tag


def encode(v: str) -> str:
    v = str(v).strip()
    v = v.replace("&", "&amp;")
    v = v.replace(">", "&gt;")
    v = v.replace("<", "&lt;")
    v = v.replace('"', "&quot;")
    return v


def _string(tag: gws.Tag):
    if not tag:
        return ''

    name = ''

    tag_iter = iter(tag)
    for e in tag_iter:
        name = str(e)
        break

    sub = []
    atts: dict = {}

    for e in tag_iter:
        if _is_list(e):
            sub.append(_string(e))
            continue
        if isinstance(e, dict):
            atts.update(e)
            continue
        if e is not None:
            e = str(e)
            if e:
                sub.append(encode(e))

    res = name
    atts_str = ''

    if atts:
        atts_str = ' '.join(f'{k}="{encode(v)}"' for k, v in atts.items() if v is not None)
    if atts_str:
        res += ' ' + atts_str
    if sub:
        return '<' + res + '>' + ''.join(sub) + '</' + name + '>'
    return '<' + res + '/>'


_NS_PREFIX = 'xmlns'


class Attribute:
    def __init__(self, qname, name, value):
        self.qname = qname
        self.name = name
        self.value = value


class Element:
    def __init__(self):
        self.attributes: t.List['Attribute'] = []
        self.children: t.List['Element'] = []
        self.name = ''
        self.namespaces = {}
        self.default_namespace = ''
        self.ns = ''
        self.qname = ''
        self.text = ''
        self.pos = [0, 0]

    def to_tag(self):
        atts = {a.qname: a.value for a in self.attributes}
        if self.default_namespace:
            atts[_NS_PREFIX] = self.default_namespace
        for k, v in self.namespaces.items():
            atts[_NS_PREFIX + ':' + k] = v
        tag = [self.qname]
        if atts:
            tag.append(atts)
        tag.extend(c.to_tag() for c in self.children)
        if self.text:
            tag.append(self.text)
        return tuple(tag)

    def strip_ns(self):
        self.ns = ''
        for a in self.attributes:
            a.qname = a.name
        self.namespaces = {}
        self.default_namespace = ''
        for c in self.children:
            c.strip_ns()
        return self

    def attr(self, key, default=None):
        key = key.lower()
        p = 'qname' if ':' in key else 'name'
        for a in self.attributes:
            if getattr(a, p).lower() == key:
                return a.value
        return default

    @property
    def attr_dict(self):
        return {a.name: a.value for a in self.attributes}

    def get(self, path, default=None):
        return self._get(path) or default

    def all(self, path=None) -> t.List['Element']:
        if not path:
            return self.children

        e = self.get(path)
        if _is_list(e):
            return e
        if _is_elem(e):
            return [e]

        return []

    def first(self, path=None, default=None) -> 'Element':
        if not path:
            if self.children:
                return self.children[0]
            return default

        e = self.get(path)
        if _is_list(e) and len(e) > 0:
            return e[0]

        return default

    def get_text(self, path) -> str:
        e = self.get(path)
        if not e:
            return ''
        if _is_str(e):
            return str(e)
        if _is_list(e):
            e = e[0]
        if _is_elem(e):
            return e.text
        return ''

    def _get(self, path):
        if isinstance(path, str):
            path = path.split('.')
        try:
            return self._get2(path)
        except (KeyError, IndexError, AttributeError):
            pass

    def _get2(self, path):
        cur = self

        for p in path:
            if p.isdigit():
                p = int(p)
            if isinstance(p, int):
                cur = cur[p]
            elif p.startswith('@'):
                cur = cur.attr(p[1:])
            else:
                if isinstance(cur, list):
                    cur = cur[0]
                p = p.lower()
                if ':' in p:
                    cur = [c for c in cur.children if c.qname.lower() == p]
                else:
                    cur = [c for c in cur.children if c.name.lower() == p]

        return cur


def from_path(path) -> Element:
    with open(path) as fp:
        return from_string(fp.read())


def from_string(src) -> Element:
    return _Handler().parse(src)


class _StopPeekParser(Exception):
    pass


def peek(src):
    h = _PeekHandler()
    try:
        h.parse(src)
    except _StopPeekParser:
        return h.stack[0]
    except xml.parsers.expat.ExpatError:
        return


def strip_before(text, el):
    p = _lc2pos(text, el.pos[0], el.pos[1])
    if p < 0:
        return ''
    if p == 0:
        return text
    return text[p:]


# Expat API
#
# AttlistDeclHandler(elname, attname, type, default, required)
# CharacterDataHandler(data)
# CommentHandler(data)
# DefaultHandler(data)
# DefaultHandlerExpand(data)
# ElementDeclHandler(name, model)
# EndCdataSectionHandler()
# EndDoctypeDeclHandler()
# EndElementHandler(name)
# EndNamespaceDeclHandler(prefix)
# EntityDeclHandler(entityName, is_parameter_entity, value, base, systemId, publicId, notationName)
# ExternalEntityRefHandler(context, base, systemId, publicId)
# NotationDeclHandler(notationName, base, systemId, publicId)
# NotStandaloneHandler()
# ProcessingInstructionHandler(target, data)
# StartCdataSectionHandler()
# StartDoctypeDeclHandler(doctypeName, systemId, publicId, has_internal_subset)
# StartElementHandler(name, attributes)
# StartNamespaceDeclHandler(prefix, uri)
# UnparsedEntityDeclHandler(entityName, base, systemId, publicId, notationName)
# XmlDeclHandler(version, encoding, standalone)


class _BaseHandler:
    def __init__(self):
        self.p = None
        self.stack = []

    def StartElementHandler(self, name, attributes):
        el = Element()

        for key, val in attributes.items():
            if key == 'xmlns':
                el.default_namespace = val.strip()
                continue

            ns, n = _ns_tag(key)

            if ns == 'xmlns':
                el.namespaces[n] = val.strip()
                continue

            el.attributes.append(Attribute(key, n, val.strip()))

        el.ns, el.name = _ns_tag(name)
        el.qname = name
        el.text = ''

        # line numbers appear to be 1-based
        # column numbers appear to be 0-based
        el.pos = [
            self.p.CurrentLineNumber - 1,
            self.p.CurrentColumnNumber
        ]

        self.stack.append(el)


class _Handler(_BaseHandler):
    def parse(self, src):
        self.stack = [Element()]

        # our inputs are always unicode
        self.p = xml.parsers.expat.ParserCreate(encoding='UTF-8')
        self.p.buffer_text = True

        self.p.StartElementHandler = self.StartElementHandler
        self.p.EndElementHandler = self.EndElementHandler
        self.p.CharacterDataHandler = self.CharacterDataHandler

        try:
            self.p.Parse(src, True)
        except xml.parsers.expat.ExpatError as e:
            raise Error(_errors[getattr(e, 'code')], self.p.ErrorLineNumber, self.p.ErrorColumnNumber)

        for root in self.stack[0].children:
            return root

    def EndElementHandler(self, name):
        el = self.stack.pop()
        el.text = el.text.strip()
        parent = self.stack[-1]
        parent.children.append(el)

    def CharacterDataHandler(self, data):
        self.stack[-1].text += data


class _PeekHandler(_BaseHandler):
    def parse(self, s):
        self.p = xml.parsers.expat.ParserCreate(encoding='UTF-8')
        self.p.StartElementHandler = self.StartElementHandler
        self.p.Parse(s, True)

    def StartElementHandler(self, name, attributes):
        super().StartElementHandler(name, attributes)
        raise _StopPeekParser()


def _ns_tag(s):
    if ':' in s:
        return s.split(':')
    return '', s


def _lc2pos(text, line, col):
    pos = 0
    while line > 0:
        pos = text.find('\n', pos)
        if pos < 0:
            return pos
        pos += 1
        line -= 1
    return pos + col


def _is_list(e):
    return isinstance(e, (list, tuple))


def _is_elem(e):
    return isinstance(e, Element)


def _is_str(e):
    return isinstance(e, str)


_errors = {
    1: 'XML_ERROR_NO_MEMORY',
    2: 'XML_ERROR_SYNTAX',
    3: 'XML_ERROR_NO_ELEMENTS',
    4: 'XML_ERROR_INVALID_TOKEN',
    5: 'XML_ERROR_UNCLOSED_TOKEN',
    6: 'XML_ERROR_PARTIAL_CHAR',
    7: 'XML_ERROR_TAG_MISMATCH',
    8: 'XML_ERROR_DUPLICATE_ATTRIBUTE',
    9: 'XML_ERROR_JUNK_AFTER_DOC_ELEMENT',
    10: 'XML_ERROR_PARAM_ENTITY_REF',
    11: 'XML_ERROR_UNDEFINED_ENTITY',
    12: 'XML_ERROR_RECURSIVE_ENTITY_REF',
    13: 'XML_ERROR_ASYNC_ENTITY',
    14: 'XML_ERROR_BAD_CHAR_REF',
    15: 'XML_ERROR_BINARY_ENTITY_REF',
    16: 'XML_ERROR_ATTRIBUTE_EXTERNAL_ENTITY_REF',
    17: 'XML_ERROR_MISPLACED_XML_PI',
    18: 'XML_ERROR_UNKNOWN_ENCODING',
    19: 'XML_ERROR_INCORRECT_ENCODING',
    20: 'XML_ERROR_UNCLOSED_CDATA_SECTION',
    21: 'XML_ERROR_EXTERNAL_ENTITY_HANDLING',
    22: 'XML_ERROR_NOT_STANDALONE',
    23: 'XML_ERROR_UNEXPECTED_STATE',
    24: 'XML_ERROR_ENTITY_DECLARED_IN_PE',
    25: 'XML_ERROR_FEATURE_REQUIRES_XML_DTD',
    26: 'XML_ERROR_CANT_CHANGE_FEATURE_ONCE_PARSING',
    27: 'XML_ERROR_UNBOUND_PREFIX',
    28: 'XML_ERROR_UNDECLARING_PREFIX',
    29: 'XML_ERROR_INCOMPLETE_PE',
    30: 'XML_ERROR_XML_DECL',
    31: 'XML_ERROR_TEXT_DECL',
    32: 'XML_ERROR_PUBLICID',
    33: 'XML_ERROR_SUSPENDED',
    34: 'XML_ERROR_NOT_SUSPENDED',
    35: 'XML_ERROR_ABORTED',
    36: 'XML_ERROR_FINISHED',
    37: 'XML_ERROR_SUSPEND_PE',
}
