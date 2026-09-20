"""XML serializer.

Element and attribute names are local (``Point``) or Clark names (``{uri}Point``); a prefixed name is an error.
An element in the default namespace (the nearest ``xmlns`` declaration, or ``defaultNamespace``) is written unprefixed.
Otherwise the prefix comes from the nearest enclosing declaration (``XmlElement.namespaces``), or from the namespace table,
renamed via ``customNamespacePrefixes``. With ``withNamespaceDeclarations``, all namespaces used in the tree are declared on the root.
"""

from typing import Optional

import re

import gws

from . import error, namespace, util


class Serializer:
    """Serializes an element tree to a string, see ``XmlElement.to_string``."""

    def __init__(self, el: gws.XmlElement, opts: Optional[gws.XmlOptions]):
        self.root = el
        self.buf = []

        self.opts = opts or gws.XmlOptions()
        self.defaultNamespace = self.opts.defaultNamespace

        self.ns_renames = self.opts.customNamespacePrefixes or {}
        self.ns_stack = []

    def to_string(self) -> str:
        """Returns the XML string."""

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
        self.ns_stack.append(namespaces)

        tag = self._element_name(el.tag)
        atts = self._process_atts(el.attrib)
        atts.update(self._namespace_declarations(namespaces, is_root))

        open_tag = tag
        if atts:
            open_tag += ' ' + ' '.join(f'{k}="{v}"' for k, v in atts.items())

        txt = self._text_to_string(el.text)
        if txt or len(el) > 0:
            self.buf.append(f'<{open_tag}>')
            self.buf.append(txt)
            for child in el:
                self._el_to_string(child, child.namespaces)
            self.buf.append(f'</{tag}>')
        else:
            self.buf.append(f'<{open_tag}/>')

        txt = self._text_to_string(el.tail)
        if txt:
            self.buf.append(txt)

        self.ns_stack.pop()

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
            self.ns_renames,
            with_schema_locations=is_root and self.opts.withSchemaLocations,
        )

    def _element_name(self, name):
        uri, pname = self._parse_name(name)

        if not uri:
            return pname

        if uri == self._default_uri():
            return pname

        return self._prefix(uri, name) + ':' + pname

    def _attribute_name(self, name):
        uri, pname = self._parse_name(name)

        if not uri:
            return pname

        return self._prefix(uri, name) + ':' + pname

    def _parse_name(self, name):
        uri, prefix, pname = namespace.parse_name(name)
        if prefix:
            raise error.WriteError(f'prefixed name {name!r}')
        if not re.fullmatch(_NAME_RE, pname):
            raise error.WriteError(f'invalid XML name {name!r}')
        return uri, pname

    def _default_uri(self):
        for nss in reversed(self.ns_stack):
            for ns in nss:
                if ns.prefix == '':
                    return ns.uri
        if self.defaultNamespace:
            return self.defaultNamespace.uri
        return ''

    def _prefix(self, uri, name):
        if uri == namespace.XML_URI:
            return namespace.XML

        for nss in reversed(self.ns_stack):
            for ns in nss:
                if ns.uri == uri and ns.prefix:
                    return self._final_prefix(ns)

        ns = namespace.find_by_uri(uri)
        if ns:
            return self._final_prefix(ns)

        if uri.startswith(namespace.ADHOC):
            return uri.removeprefix(namespace.ADHOC)

        raise error.NamespaceError(f'unknown namespace in {name!r}')

    def _final_prefix(self, ns) -> str:
        pfx = self.ns_renames.get(ns.uri) or ns.prefix
        if not re.fullmatch(_NAME_RE, pfx):
            raise error.WriteError(f'invalid XML prefix {pfx!r}')
        return pfx

    def _collect_namespaces(self, root_ns):
        # namespaces to declare on the root: its own, the default one and those used in the tree
        # and not declared on an enclosing element

        nss = list(root_ns)

        if self.defaultNamespace and all(ns.prefix != '' for ns in nss):
            nss.append(namespace.new('', self.defaultNamespace.uri, self.defaultNamespace.schemaLocation))

        def declared(uri, scopes, for_attribute):
            for lst in scopes:
                for ns in lst:
                    if ns.uri == uri and (ns.prefix or not for_attribute):
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
            uri, _, _ = namespace.parse_name(el.tag)
            resolve(uri, el.tag, scopes, False)
            for key in el.attrib:
                uri, _, _ = namespace.parse_name(key)
                resolve(uri, key, scopes, True)
            for c in el:
                walk(c, scopes + [c.namespaces] if c.namespaces else scopes)

        walk(self.root, [nss])

        seen = {}
        for ns in nss:
            pfx = self.ns_renames.get(ns.uri) or ns.prefix
            if pfx in seen and seen[pfx] != ns.uri:
                raise error.NamespaceError(f'namespace prefix {pfx!r} is used for {seen[pfx]!r} and {ns.uri!r}')
            seen[pfx] = ns.uri

        return nss

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
    '\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02ff\u0370-\u037d\u037f-\u1fff\u200c-\u200d'
    '\u2070-\u218f\u2c00-\u2fef\u3001-\ud7ff\uf900-\ufdcf\ufdf0-\ufffd\U00010000-\U000effff'
)
_NAME_CHAR = _NAME_START + '\\-.0-9\u00b7\u0300-\u036f\u203f-\u2040'
_NAME_RE = f'[{_NAME_START}][{_NAME_CHAR}]*'
