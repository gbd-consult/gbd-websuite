"""XML serializer."""

from typing import Optional

import re

import gws

from . import error, namespace, util


class Serializer:
    def __init__(self, el: gws.XmlElement, opts: Optional[gws.XmlOptions]):
        self.root = el
        self.buf = []

        self.opts = opts or gws.XmlOptions()
        self.defaultNamespace = self.opts.defaultNamespace

        # uri -> custom prefix
        self.renames = self.opts.customXmlns or {}

        # prefix -> namespace, declared by the caller
        self.rootMap = dict(self.opts.namespaces or {})

        # prefix -> namespace, resolved from the well-known table while serializing
        self.usedMap = {}

        # element-level namespace maps of the enclosing elements
        self.stack = []

        # prefixes used by attributes, these must be declared even if they map to the default namespace
        self.attributePrefixes = set()

    def to_string(self):
        if self.opts.withXmlDeclaration or self.opts.doctype:
            self.buf.append(_XML_DECL)
            if self.opts.doctype:
                self.buf.append(f'<!DOCTYPE {self.opts.doctype}>')

        self._el_to_string(self.root, is_root=True)

        return ''.join(self.buf)

    ##

    def _el_to_string(self, el, is_root=False):
        self.stack.append(el.namespaces)

        open_pos = len(self.buf)
        self.buf.append('')

        tag = self._element_name(el.tag)
        atts = self._process_atts(el.attrib)

        s = self._text_to_string(el.text)
        if s:
            self.buf.append(s)

        for c in el:
            self._el_to_string(c)

        atts.update(self._namespace_declarations(el, is_root))

        open_tag = tag
        if atts:
            open_tag += ' ' + ' '.join(f'{k}="{v}"' for k, v in atts.items())

        if len(self.buf) > open_pos + 1:
            self.buf[open_pos] = f'<{open_tag}>'
            self.buf.append(f'</{tag}>')
        else:
            self.buf[open_pos] = f'<{open_tag}/>'

        s = self._text_to_string(el.tail)
        if s:
            self.buf.append(s)

        self.stack.pop()

    def _process_atts(self, attrib):
        atts = {}

        for key, val in attrib.items():
            if val is None:
                continue
            atts[self._attribute_name(key)] = self._value_to_string(val)

        return atts

    def _namespace_declarations(self, el, is_root):
        decls = {}

        if is_root and self.opts.withNamespaceDeclarations:
            if self.defaultNamespace:
                decls[''] = self.defaultNamespace
            decls.update(self.rootMap)
            decls.update(self.usedMap)

        decls.update(el.namespaces)

        if not decls:
            return {}

        renamed = {}
        for prefix, ns in decls.items():
            if prefix == '':
                renamed[''] = ns
                continue
            p = self._output_prefix(prefix, ns)
            if self.defaultNamespace and ns.uri == self.defaultNamespace.uri and p not in self.attributePrefixes:
                continue
            renamed[p] = ns

        return namespace.declarations(
            renamed,
            with_schema_locations=is_root and self.opts.withSchemaLocations,
        )

    def _element_name(self, name):
        prefix, pname = namespace.split_name(name)
        self._check_name(pname, name)

        if not prefix:
            return pname

        ns = self._resolve(prefix, name)
        if self.defaultNamespace and ns.uri == self.defaultNamespace.uri:
            return pname

        return self._output_prefix(prefix, ns) + ':' + pname

    def _attribute_name(self, name):
        prefix, pname = namespace.split_name(name)
        self._check_name(pname, name)

        if not prefix:
            return pname

        if prefix == namespace.XMLNS or prefix == namespace.XML:
            self._check_name(prefix, name)
            return name

        ns = self._resolve(prefix, name)
        p = self._output_prefix(prefix, ns)
        self.attributePrefixes.add(p)
        return p + ':' + pname

    def _resolve(self, prefix, name):
        ns = self.rootMap.get(prefix)
        if ns:
            return ns

        for m in reversed(self.stack):
            ns = m.get(prefix)
            if ns:
                return ns

        ns = namespace.get(prefix)
        if ns:
            self.usedMap[prefix] = ns
            return ns

        raise error.NamespaceError(f'unknown namespace prefix in {name!r}')

    def _output_prefix(self, prefix, ns):
        p = self.renames.get(ns.uri, prefix)
        self._check_name(p, p)
        return p

    def _check_name(self, s, name):
        if not re.fullmatch(_NAME_RE, s):
            raise error.WriteError(f'invalid XML name {name!r}')

    def _text_to_string(self, arg):
        s, ok = util.atom_to_string(arg)
        if not ok:
            s = str(arg)
        if self.opts.compactWhitespace:
            s = ' '.join(s.strip().split())
        return util.escape_text(s)

    def _value_to_string(self, arg):
        s, ok = util.atom_to_string(arg)
        if not ok:
            s = str(arg)
        return util.escape_attribute(s)


_XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>'

# XML 1.0 (5th ed.) NCName, https://www.w3.org/TR/xml/#NT-Name

_NAME_START = (
    'A-Za-z_'
    '\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D'
    '\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD\U00010000-\U000EFFFF'
)
_NAME_CHAR = _NAME_START + '\\-.0-9\u00B7\u0300-\u036F\u203F-\u2040'
_NAME_RE = f'[{_NAME_START}][{_NAME_CHAR}]*'
