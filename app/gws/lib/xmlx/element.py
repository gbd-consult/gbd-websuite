"""XML Element.

The Element class extends ``xml.etree.Element``
and implements the `gws.core.types.IXmlElement` protocol.
"""

import xml.etree.ElementTree

import gws

from . import namespace, error


class XmlElementImpl(xml.etree.ElementTree.Element, gws.XmlElement):

    def __init__(self, tag, attrib=None, **extra):
        xml.etree.ElementTree.Element.__init__(self, tag, attrib or {}, **extra)

        self.text = self.text or ''
        self.tail = self.tail or ''

        _, _, pname = namespace.split_name(tag)
        self.name = pname
        self.lcName = pname.lower()

        self.caseInsensitive = False

    def __bool__(self):
        return True

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def find(self, path, namespaces=None):
        return super().find(self._convert_path(path), namespaces)

    def findtext(self, path, default=None, namespaces=None):
        return super().findtext(self._convert_path(path), default, namespaces)

    def findall(self, path, namespaces=None):
        return super().findall(self._convert_path(path), namespaces)

    def iterfind(self, path, namespaces=None):
        return super().iterfind(self._convert_path(path), namespaces)

    def get(self, key, default=None):
        if self.caseInsensitive:
            key = key.lower()
        if key in self.attrib:
            return self.attrib[key]
        for k, v in self.attrib.items():
            if namespace.unqualify_name(k) == key:
                return v
        return default

    def iter(self, tag=None):
        return super().iter(self._convert_path(tag))

    ##

    def to_dict(self):
        return {
            'tag': self.tag,
            'attrib': self.attrib,
            'text': self.text,
            'tail': self.tail,
            'children': [c.to_dict() for c in self.children()]

        }

    def to_string(
            self,
            extra_namespaces=None,
            compact_whitespace=False,
            remove_namespaces=False,
            with_namespace_declarations=False,
            with_schema_locations=False,
            with_xml_declaration=False,
    ):

        def make_text(s):
            if s is None:
                return ''
            if isinstance(s, (int, float, bool)):
                return str(s).lower()
            if not isinstance(s, str):
                s = str(s)
            if compact_whitespace:
                s = ' '.join(s.strip().split())
            s = s.replace("&", "&amp;")
            s = s.replace(">", "&gt;")
            s = s.replace("<", "&lt;")
            return s

        def make_attr(s):
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

        def make_name(name):
            if remove_namespaces:
                return namespace.unqualify_name(name)
            ns, pname = namespace.parse_name(name)
            if not ns:
                return name
            if ns and default_ns and ns.uri == default_ns.uri:
                return pname
            return ns.xmlns + ':' + pname

        def to_str(el, extra_atts=None):

            atts = {}

            for key, val in el.attrib.items():
                if val is None:
                    continue
                atts[make_name(key)] = make_attr(val)

            if extra_atts:
                atts.update(extra_atts)

            open_pos = len(buf)
            buf.append('')

            s = make_text(el.text)
            if s:
                buf.append(s)

            for ch in el:
                to_str(ch)

            open_tag = make_name(el.tag)
            close_tag = open_tag
            if atts:
                open_tag += ' ' + ' '.join(f'{k}="{v}"' for k, v in atts.items())

            if len(buf) > open_pos + 1:
                buf[open_pos] = f'<{open_tag}>'
                buf.append(f'</{close_tag}>')
            else:
                buf[open_pos] += f'<{open_tag}/>'

            s = make_text(el.tail)
            if s:
                buf.append(s)

        ##

        buf = ['']

        default_ns = None
        xmlns = self.get('xmlns')
        if xmlns:
            default_ns = namespace.get(xmlns)
            if not default_ns:
                raise error.NamespaceError(f'unknown namespace {xmlns!r}')

        extra_atts = None
        if with_namespace_declarations:
            extra_atts = namespace.declarations(
                for_element=self,
                default_ns=default_ns,
                with_schema_locations=with_schema_locations,
                extra_ns=extra_namespaces,
            )

        to_str(self, extra_atts)

        if with_xml_declaration:
            buf[0] = _XML_DECL

        return ''.join(buf)

    ##

    def add(self, tag, attrib=None, **extra):
        el = self.__class__(tag, attrib, **extra)
        el.caseInsensitive = self.caseInsensitive
        self.append(el)
        return el

    def attr(self, key, default=None):
        return self.get(key, default)

    def children(self):
        return [c for c in self]

    def findfirst(self, *paths):
        if not paths:
            return self[0] if len(self) > 0 else None
        for path in paths:
            el = self.find(path)
            if el is not None:
                return el

    def textof(self, *paths):
        for path in paths:
            el = self.find(path)
            if el is not None and el.text:
                return el.text

    def textlist(self, *paths, deep=False):
        buf = self._collect_text(paths, deep)
        return [text for _, text in buf]

    def textdict(self, *paths, deep=False):
        buf = self._collect_text(paths, deep)
        return dict(buf)

    ##

    def _collect_text(self, paths, deep):
        def walk(el):
            s = (el.text or '').strip()
            if s:
                buf.append((el.tag, s))
            if deep:
                for c in el:
                    walk(c)

        buf = []

        if not paths:
            for el in self:
                walk(el)
        else:
            for path in paths:
                for el in self.findall(path):
                    walk(el)

        return buf

    def _convert_path(self, path):
        if self.caseInsensitive:
            path = path.lower()
        return path

    def _convert_key(self, key):
        if self.caseInsensitive:
            key = key.lower()
        return key


_XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>'
