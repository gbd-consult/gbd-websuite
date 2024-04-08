"""XML Element.

The Element class extends ``xml.etree.Element``
and implements the `gws.core.types.IXmlElement` protocol.
"""

import xml.etree.ElementTree
from xml.etree.ElementTree import Element

import gws
import gws.types as t

from . import namespace, error


class XElement(xml.etree.ElementTree.Element):
    """Class represents XML elements."""
    caseInsensitive: bool
    """Flag for case sensitivity."""
    name: str
    """Name of the element."""
    lname: str
    """Lowercase name."""

    def __init__(self, tag, attrib=None, **extra):
        super().__init__(tag, attrib or {}, **extra)

        self.text = self.text or ''
        self.tail = self.tail or ''

        _, _, pname = namespace.split_name(tag)
        self.name = pname
        self.lname = pname.lower()

        self.caseInsensitive = False

    def __bool__(self):
        return True

    def __iter__(self) -> t.Iterator['XElement']:
        # need this for typing of `for sub in elem` loops
        for i in range(len(self)):
            yield self[i]

    def find(self, path, namespaces=None):
        """Finds first matching element by tag name or path."""
        return super().find(self._convert_path(path), namespaces)

    def findtext(self, path, default=None, namespaces=None):
        """Finds text for first matching element by name or path."""
        return super().findtext(self._convert_path(path), default, namespaces)

    def findall(self, path, namespaces=None):
        """Finds all matching subelements by name or path."""
        return super().findall(self._convert_path(path), namespaces)

    def iterfind(self, path, namespaces=None):
        """Returns an iterable of all matching subelements by name or path."""
        return super().iterfind(self._convert_path(path), namespaces)

    def get(self, key, default=None):
        """Returns the value to a given key."""
        if self.caseInsensitive:
            key = key.lower()
        if key in self.attrib:
            return self.attrib[key]
        for k, v in self.attrib.items():
            if namespace.unqualify_name(k) == key:
                return v
        return default

    def iter(self, tag=None):
        """Creates a tree iterator."""
        return super().iter(self._convert_path(tag))

    ##

    def to_dict(self) -> dict:
        """Creates a dictionary from an XElement object.

        Returns:
            A dict with the keys ``tag``, ``attrib``, ``text``, ``tail``, ``tail``, ``children``."""
        return {
            'tag': self.tag,
            'attrib': self.attrib,
            'text': self.text,
            'tail': self.tail,
            'children': [c.to_dict() for c in self.children()]

        }

    def to_string(
            self,
            compact_whitespace: bool = False,
            remove_namespaces: bool = False,
            with_namespace_declarations: bool = False,
            with_schema_locations: bool = False,
            with_xml_declaration: bool = False,
    ) -> str:
        """Converts the XElement object to a string.

        Args:
            compact_whitespace: String will not contain any whitespaces outside of tags and elements.
            remove_namespaces: String will not contain namespaces.
            with_namespace_declarations: String will keep the namespace declarations.
            with_schema_locations: String will keep the schema locations.
            with_xml_declaration: String will keep the xml ddeclaration.

        Returns:
            A String containing the xml structure.
        """

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

        def to_str(el, with_xmlns):

            atts = {}

            for key, val in el.attrib.items():
                if val is None:
                    continue
                atts[make_name(key)] = make_attr(val)

            if with_xmlns:
                atts.update(
                    namespace.declarations(
                        for_element=el,
                        default_ns=default_ns,
                        with_schema_locations=with_schema_locations))

            open_pos = len(buf)
            buf.append('')

            s = make_text(el.text)
            if s:
                buf.append(s)

            for ch in el:
                to_str(ch, False)

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

        to_str(self, with_namespace_declarations)

        if with_xml_declaration:
            buf[0] = _XML_DECL

        return ''.join(buf)

    ##

    def add(self, tag: str, attrib: dict = None, **extra) -> 'XElement':
        """Creates a new ``XElement`` and adds it as a child.

        Args:
            tag: XML tag.
            attrib: XML attributes ``{key, value}``.

        Returns:
            A XElement.
        """
        el = self.__class__(tag, attrib, **extra)
        el.caseInsensitive = self.caseInsensitive
        self.append(el)
        return el

    def attr(self, key: str, default=None) -> str:
        """Finds the value for a given key in the ``XElement``.

        Args:
            key: Key of the attribute.
            default: The default return.

        Returns:
            The vale of the key, If the key is not found the default is returned.
        """
        return self.get(key, default)

    def children(self) -> ['XElement']:
        """Returns the children of the current ``XElement``."""
        return [c for c in self]

    def findfirst(self, *paths:str) -> Element:
        """Returns the first element in the current element.

        Args:
            paths: Path as ``tag/tag2/tag3`` to the Element to search in.

        Returns:
            Returns the first found element.
            """
        if not paths:
            return self[0] if len(self) > 0 else None
        for path in paths:
            el = self.find(path)
            if el is not None:
                return el

    def textof(self, *paths:str) -> str:
        """Returns the text of a given child-element.

        Args:
            paths: Path as ``tag/tag2/tag3`` to the Element.

        Returns:
            The text of the element.

        """
        for path in paths:
            el = self.find(path)
            if el is not None and el.text:
                return el.text

    def textlist(self, *paths:str, deep: bool = False) -> ['XElement']:
        """Collects texts from child-elements.

        Args:
            paths: Path as ``tag/tag2/tag3`` to the Element to collect texts from.
            deep: If ``False`` it only looks into direct children, otherwise it searches for texts in the complete children-tree.

        Returns:
            A list containing all the text from the child-elements.
        """
        buf = self._collect_text(paths, deep)
        return [text for _, text in buf]

    def textdict(self, *paths:str, deep: bool = False) -> dict:
        """Collects texts from child-elements.

        Args:
            paths: Path as ``tag/tag2/tag3`` to the Element to collect texts from.
            deep: If ``False`` it only looks into direct children, otherwise it searches for texts in the complete children-tree.

        Returns:
            A dict containing all the text from the child-elements.
        """
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
