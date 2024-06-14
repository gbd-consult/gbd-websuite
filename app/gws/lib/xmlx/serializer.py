from typing import Optional
import gws
from . import namespace, error


class _SerializerState:
    extra_namespaces: list[gws.XmlNamespace]

    compact_whitespace = False
    remove_namespaces = False
    fold_tags = False
    with_namespace_declarations = False
    with_schema_locations = False
    with_xml_declaration = False

    buf: list[str]
    defaultNamespace: Optional[gws.XmlNamespace]


def _state(kwargs):
    ser = _SerializerState()

    ser.extra_namespaces = kwargs.get('extra_namespaces', [])

    ser.compact_whitespace = kwargs.get('compact_whitespace', False)
    ser.fold_tags = kwargs.get('fold_tags', False)
    ser.remove_namespaces = kwargs.get('remove_namespaces', False)
    ser.with_namespace_declarations = kwargs.get('with_namespace_declarations', False)
    ser.with_schema_locations = kwargs.get('with_schema_locations', False)
    ser.with_xml_declaration = kwargs.get('with_xml_declaration', False)

    ser.buf = []
    ser.defaultNamespace = []

    return ser


##


def to_string(el: gws.XmlElement, **kwargs):
    ser = _state(kwargs)

    _set_default_namespace(ser, el)

    extra_atts = None

    if ser.with_namespace_declarations:
        extra_atts = namespace.declarations(
            for_element=el,
            default_ns=ser.defaultNamespace,
            with_schema_locations=ser.with_schema_locations,
            extra_ns=ser.extra_namespaces,
        )

    if ser.with_xml_declaration:
        ser.buf.append(_XML_DECL)

    _to_string(ser, el, extra_atts)

    return ''.join(ser.buf)


def _to_string(ser: _SerializerState, el: gws.XmlElement, extra_atts=None):
    atts = {}

    for key, val in el.attrib.items():
        if val is None:
            continue
        atts[_make_name(ser, key)] = _value_to_string(ser, val)

    if extra_atts:
        atts.update(extra_atts)

    open_pos = len(ser.buf)
    ser.buf.append('')

    s = _text_to_string(ser, el.text)
    if s:
        ser.buf.append(s)

    for c in el:
        _to_string(ser, c)

    open_tag = _make_name(ser, el.tag)
    close_tag = open_tag
    if atts:
        open_tag += ' ' + ' '.join(f'{k}="{v}"' for k, v in atts.items())

    if len(ser.buf) > open_pos + 1:
        ser.buf[open_pos] = f'<{open_tag}>'
        ser.buf.append(f'</{close_tag}>')
    else:
        ser.buf[open_pos] += f'<{open_tag}/>'

    s = _text_to_string(ser, el.tail)
    if s:
        ser.buf.append(s)


##

def to_list(el: gws.XmlElement, **kwargs):
    ser = _state(kwargs)

    _set_default_namespace(ser, el)

    return _to_list(ser, el)


def _to_list(ser, el):
    name = _make_name(ser, el.tag)
    attr = {_make_name(ser, k): v for k, v in el.attrib.items()}
    text = (el.text or '').strip()
    tail = (el.tail or '').strip()

    sub = [_to_list(ser, c) for c in el]

    if ser.fold_tags and len(sub) == 1 and (not attr and not text and not tail):
        # single wrapper tag, create 'tag/subtag
        inner = sub[0]
        inner[0] = name + '/' + inner[0]
        return inner

    res = [name, attr, text, sub, tail]
    return [x for x in res if x]


##

def _set_default_namespace(ser, el):
    xmlns = el.get('xmlns')
    if xmlns:
        ns = namespace.get(xmlns)
        if not ns:
            raise error.NamespaceError(f'unknown namespace {xmlns!r}')
        ser.defaultNamespace = ns


def _text_to_string(ser, s):
    if s is None:
        return ''
    if isinstance(s, (int, float, bool)):
        return str(s).lower()
    if not isinstance(s, str):
        s = str(s)
    if ser.compact_whitespace:
        s = ' '.join(s.strip().split())
    s = s.replace("&", "&amp;")
    s = s.replace(">", "&gt;")
    s = s.replace("<", "&lt;")
    return s


def _value_to_string(ser, s):
    if s is None:
        return ''
    if isinstance(s, (int, float, bool)):
        return str(s).lower()
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("&", "&amp;")
    s = s.replace('"', "&quot;")
    s = s.replace(">", "&gt;")
    s = s.replace("<", "&lt;")
    return s


def _make_name(ser, name):
    if ser.remove_namespaces:
        return namespace.unqualify_name(name)
    ns, pname = namespace.parse_name(name)
    if not ns:
        return name
    if ns and ser.defaultNamespace and ns.uri == ser.defaultNamespace.uri:
        return pname
    return ns.xmlns + ':' + pname


_XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>'
