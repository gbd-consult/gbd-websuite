"""XmlElement implementation."""

import xml.etree.ElementTree

import gws

from . import namespace, serializer


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

    def to_list(self, fold_tags=True, remove_namespaces=False):
        return serializer.to_list(
            self,
            fold_tags=fold_tags,
            remove_namespaces=remove_namespaces,
        )

    def to_string(
            self,
            extra_namespaces=None,
            compact_whitespace=False,
            remove_namespaces=False,
            with_namespace_declarations=False,
            with_schema_locations=False,
            with_xml_declaration=False,
    ):
        return serializer.to_string(
            self,
            extra_namespaces=extra_namespaces,
            compact_whitespace=compact_whitespace,
            remove_namespaces=remove_namespaces,
            with_namespace_declarations=with_namespace_declarations,
            with_schema_locations=with_schema_locations,
            with_xml_declaration=with_xml_declaration,
        )

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

