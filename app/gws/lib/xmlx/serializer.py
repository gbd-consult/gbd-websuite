"""XML serializer.

Element and attribute names are local (``Point``) or Clark names (``{uri}Point``); a prefixed name is an error.
An element in the default namespace (the nearest ``xmlns`` declaration, or ``defaultNamespace``) is written unprefixed.
Otherwise the prefix comes from the nearest enclosing declaration (``XmlElement.namespaces``), or from the namespace table,
renamed via ``customXmlns``. With ``withNamespaceDeclarations``, all namespaces used in the tree are declared on the root.
"""

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

        # namespace lists of the enclosing elements
        self.stack = []

    def to_string(self):
        if self.opts.withXmlDeclaration or self.opts.doctype:
            self.buf.append(_XML_DECL)
            if self.opts.doctype:
                self.buf.append(f'<!DOCTYPE {self.opts.doctype}>')

        root_ns = list(self.root.namespaces)
        if self.opts.withNamespaceDeclarations:
            root_ns = self._collect_namespaces(root_ns)

        self._el_to_string(self.root, root_ns, is_root=True)

        return ''.join(self.buf)

    ##

    def _el_to_string(self, el, namespaces, is_root=False):
        self.stack.append(namespaces)

        tag = self._element_name(el.tag)
        atts = self._process_atts(el.attrib)
        atts.update(self._namespace_declarations(namespaces, is_root))

        open_tag = tag
        if atts:
            open_tag += ' ' + ' '.join(f'{k}="{v}"' for k, v in atts.items())

        s = self._text_to_string(el.text)
        if s or len(el) > 0:
            self.buf.append(f'<{open_tag}>')
            self.buf.append(s)
            for c in el:
                self._el_to_string(c, c.namespaces)
            self.buf.append(f'</{tag}>')
        else:
            self.buf.append(f'<{open_tag}/>')

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

    def _namespace_declarations(self, namespaces, is_root):
        if not namespaces:
            return {}
        return namespace.declarations(
            namespaces,
            self.renames,
            with_schema_locations=is_root and self.opts.withSchemaLocations,
        )

    def _element_name(self, name):
        uri, pname = namespace.split_clark_name(name)
        self._check_name(pname, name)

        if not uri:
            return pname

        if uri == self._default_uri():
            return pname

        return self._prefix(uri, name) + ':' + pname

    def _attribute_name(self, name):
        uri, pname = namespace.split_clark_name(name)
        self._check_name(pname, name)

        if not uri:
            return pname

        return self._prefix(uri, name) + ':' + pname

    def _default_uri(self):
        for nss in reversed(self.stack):
            for ns in nss:
                if ns.xmlns == '':
                    return ns.uri
        if self.defaultNamespace:
            return self.defaultNamespace.uri
        return ''

    def _prefix(self, uri, name):
        if uri == namespace.XML_URI:
            return namespace.XML

        for nss in reversed(self.stack):
            for ns in nss:
                if ns.uri == uri and ns.xmlns:
                    return self._output_prefix(ns)

        ns = namespace.find_by_uri(uri)
        if ns:
            return self._output_prefix(ns)

        if uri.startswith(namespace.ADHOC):
            return uri[len(namespace.ADHOC):]

        raise error.NamespaceError(f'unknown namespace in {name!r}')

    def _output_prefix(self, ns):
        p = self.renames.get(ns.uri, ns.xmlns)
        self._check_name(p, p)
        return p

    def _collect_namespaces(self, root_ns):
        # namespaces to declare on the root: its own, the default one and those used in the tree
        # and not declared on an enclosing element

        nss = list(root_ns)

        if self.defaultNamespace and all(ns.xmlns != '' for ns in nss):
            nss.append(namespace.new('', self.defaultNamespace.uri, self.defaultNamespace.schemaLocation))

        def declared(uri, scopes, for_attribute):
            for lst in scopes:
                for ns in lst:
                    if ns.uri == uri and (ns.xmlns or not for_attribute):
                        return True
            return False

        def resolve(uri, name, scopes, for_attribute):
            if not uri or uri == namespace.XML_URI or uri.startswith(namespace.ADHOC):
                return
            if declared(uri, scopes, for_attribute):
                return
            ns = namespace.find_by_uri(uri)
            if not ns:
                raise error.NamespaceError(f'unknown namespace in {name!r}')
            nss.append(ns)

        def walk(el, scopes):
            uri, _ = namespace.split_clark_name(el.tag)
            resolve(uri, el.tag, scopes, False)
            for key in el.attrib:
                uri, _ = namespace.split_clark_name(key)
                resolve(uri, key, scopes, True)
            for c in el:
                walk(c, scopes + [c.namespaces] if c.namespaces else scopes)

        walk(self.root, [nss])

        seen = {}
        for ns in nss:
            p = self.renames.get(ns.uri, ns.xmlns)
            if p in seen and seen[p] != ns.uri:
                raise error.NamespaceError(f'namespace prefix {p!r} is used for {seen[p]!r} and {ns.uri!r}')
            seen[p] = ns.uri

        return nss

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
