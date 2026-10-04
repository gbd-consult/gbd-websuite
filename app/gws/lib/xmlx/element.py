"""XmlElement implementation."""

from typing import Iterable, Optional, cast
import xml.etree.ElementPath as ElementPath

import gws

from . import error, namespace, serializer


class XmlElement(gws.XmlElement):
    """XML element."""

    def __init__(self, tag: str, attrib: Optional[dict] = None, **extra):
        """Create an element.

        Names are taken as they are, no namespace resolution is done.

        Args:
            tag: Tag name, local or Clark.
            attrib: Attributes.
            **extra: Additional attributes.
        """

        self.tag = tag
        self.name = namespace.plain_name(tag)
        self.text = ''
        self.tail = ''
        self.namespaces = []
        self._children = []

        self.attrib = {}
        if attrib:
            self.attrib.update(attrib)
        if extra:
            self.attrib.update(extra)

    # ElementTree.Element implementations, copied from ElementTree.py

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.tag!r} at {id(self):#x}>'

    def makeelement(self, tag, attrib):
        """Create a new element of the same class, as in ``ElementTree``.

        The new element is not added to this element.

        Args:
            tag: Tag name, local or Clark.
            attrib: Attributes.

        Returns:
            A new element.
        """

        return self.__class__(tag, attrib)

    def __copy__(self):
        """Shallow copy: the children are shared with the original."""

        elem = self.__class__(self.tag, self.attrib)
        elem.text = self.text
        elem.tail = self.tail
        elem.namespaces = list(self.namespaces)
        elem._children = list(self._children)
        return elem

    def __len__(self):
        return len(self._children)

    def __getitem__(self, index):
        return self._children[index]

    def __setitem__(self, index, element):
        """Replace the child at the given index."""

        self._children[index] = element

    def __delitem__(self, index):
        """Remove the child at the given index."""

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

    def find(self, path):
        return cast(Optional[gws.XmlElement], ElementPath.find(self, path))

    def findtext(self, path, default=''):
        return cast(str, ElementPath.findtext(self, path, default))

    def findall(self, path):
        return cast(list[gws.XmlElement], ElementPath.findall(self, path))

    def iterfind(self, path):
        return cast(Iterable[gws.XmlElement], ElementPath.iterfind(self, path))

    def clear(self):
        self.attrib = {}
        self.namespaces = []
        self._children = []
        self.text = self.tail = ''

    def get(self, key, default=''):
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
        """An element is always true, also when it has no children."""

        return True

    def __iter__(self):
        return iter(self._children)

    def require(self, path):
        el = self.find(path)
        if el is None:
            raise error.Error(f'XmlElement: required element not found: {path!r}')
        return el

    def children(self):
        return self._children

    def hasattr(self, key):
        return key in self.attrib

    def isa(self, *names):
        n = self.name.lower()
        return any(n == s.lower() for s in names)

    def to_dict(self):
        return {
            'tag': self.tag,
            'attrib': dict(self.attrib),
            'text': self.text,
            'tail': self.tail,
            'children': [c.to_dict() for c in self._children],
        }

    def to_string(self, opts=None):
        ser = serializer.Serializer(self, opts=opts)
        return ser.to_string()

    ##

    def add(self, tag, attrib=None, **extra):
        el = self.__class__(tag, attrib or {}, **extra)
        self.append(el)
        return el

    def declare(self, *namespaces):
        for ns in namespaces:
            if not any(n.uri == ns.uri and n.prefix == ns.prefix for n in self.namespaces):
                self.namespaces.append(ns)

    def findfirst(self, *paths):
        if not paths:
            return self._children[0] if len(self._children) > 0 else None
        for path in paths:
            el = self.find(path)
            if el is not None:
                return el
        return None

    def textof(self, *paths):
        for path in paths:
            el = self.find(path)
            if el is not None and el.text:
                return el.text
        return ''

    def textlist(self, *paths, deep=False):
        ls = self._collect_tags_and_text(paths, deep)
        return [text for _, text in ls]

    def textdict(self, *paths, deep=False):
        ls = self._collect_tags_and_text(paths, deep)
        return dict(ls)

    def _collect_tags_and_text(self, paths, deep):
        """Collect ``(tag, stripped text)`` pairs for ``textlist`` and ``textdict``."""

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
