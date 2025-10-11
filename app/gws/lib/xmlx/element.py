"""XmlElement implementation."""

from typing import Optional
import xml.etree.ElementPath as ElementPath

import gws

from . import namespace, serializer


class XmlElement(gws.XmlElement):
    def __init__(self, tag: str, attrib: Optional[dict] = None, **extra):
        self.tag = tag
        self.text = ''
        self.tail = ''
        self.attrib = {**(attrib or {}), **extra}
        self._children = []

        # extensions

        pname = namespace.unqualify_name(tag)
        self.name = pname
        self.lcName = pname.lower()

    # ElementTree.Element implementations, copied from Python 3.11 ElementTree.py

    def __repr__(self):
        return '<%s %r at %#x>' % (self.__class__.__name__, self.tag, id(self))

    def makeelement(self, tag, attrib):
        return self.__class__(tag, attrib)

    def __copy__(self):
        elem = self.__class__(self.tag, self.attrib)
        elem.text = self.text
        elem.tail = self.tail
        elem._children = list(self._children)
        return elem

    def __len__(self):
        return len(self._children)

    def __getitem__(self, index):
        return self._children[index]

    def __setitem__(self, index, element):
        self._children[index] = element

    def __delitem__(self, index):
        del self._children[index]

    def append(self, subelement):
        self._children.append(subelement)

    def extend(self, elements):
        for element in elements:
            self._children.append(element)

    def insert(self, index, subelement):
        self._children.insert(index, subelement)

    def remove(self, subelement):
        self._children.remove(subelement)

    def find(self, path, namespaces=None):
        return ElementPath.find(self, path, namespaces)

    def findtext(self, path, default=None, namespaces=None):
        return ElementPath.findtext(self, path, default, namespaces)

    def findall(self, path, namespaces=None):
        return ElementPath.findall(self, path, namespaces)

    def iterfind(self, path, namespaces=None):
        return ElementPath.iterfind(self, path, namespaces)

    def clear(self):
        self.attrib = {}
        self._children = []
        self.text = self.tail = ''

    def get(self, key, default=None):
        return self.attrib.get(key, default)

    def set(self, key, value):
        self.attrib[key] = value

    def keys(self):
        return self.attrib.keys()

    def items(self):
        return self.attrib.items()

    def iter(self, tag=None):
        if tag == '*':
            tag = None
        if tag is None or self.tag == tag:
            yield self
        for e in self._children:
            yield from e.iter(tag)

    def itertext(self):
        tag = self.tag
        if not isinstance(tag, str) and tag is not None:
            return
        t = self.text
        if t:
            yield t
        for e in self:
            yield from e.itertext()
            t = e.tail
            if t:
                yield t

    ## extensions

    def __bool__(self):
        return True

    def __iter__(self):
        for c in self._children:
            yield c

    def children(self):
        return self._children

    def has(self, key):
        return key in self.attrib

    def to_dict(self):
        return {
            'tag': self.tag,
            'attrib': self.attrib,
            'text': self.text,
            'tail': self.tail,
            'children': [c.to_dict() for c in self.children()],
        }

    def to_list(self, opts=None):
        ser = serializer.Serializer(self, opts=opts)
        return ser.to_list()

    def to_string(self, opts=None):
        ser = serializer.Serializer(self, opts=opts)
        return ser.to_string()

    def to_str(self, opts=None):
        ser = serializer.Serializer(self, opts=opts)
        return ser.to_string()

    ##

    def add(self, tag, attrib=None, **extra):
        el = self.__class__(tag, attrib or {}, **extra)
        self.append(el)
        return el

    def attr(self, key, default=''):
        return self.get(key, default)

    def findfirst(self, *paths):
        if not paths:
            return self._children[0] if len(self._children) > 0 else None
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
        ls = self._collect_tags_and_text(paths, deep)
        return [text for _, text in ls]

    def textdict(self, *paths, deep=False):
        ls = self._collect_tags_and_text(paths, deep)
        return dict(ls)


    def _collect_tags_and_text(self, paths, deep):
        def walk(el):
            s = (el.text or '').strip()
            if s:
                ls.append((el.tag, s))
            if deep:
                for c in el:
                    walk(c)

        ls = []

        if not paths:
            for el in self:
                walk(el)
        else:
            for path in paths:
                for el in self.findall(path):
                    walk(el)

        return ls
