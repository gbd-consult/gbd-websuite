"""Expat-based XML parser."""

from typing import Optional

import re
import pyexpat

import gws

from . import error, element, namespace


def from_path(path: str, opts: Optional[gws.XmlOptions] = None) -> gws.XmlElement:
    """Parse an XML file.

    Args:
        path: Path to the file.
        opts: Parsing options (``removeNamespaces``, ``compactWhitespace``).

    Returns:
        The root element.

    Raises:
        ParseError: If the document is malformed, contains entity declarations or cannot be decoded.
    """

    with open(path, 'rb') as fp:
        inp = fp.read()
    return _parse(inp, opts)


def from_string(inp: str | bytes, opts: Optional[gws.XmlOptions] = None) -> gws.XmlElement:
    """Parse an XML document from a string or bytes.

    Bytes are decoded as UTF-8, then with the declared encoding, then as Latin-1.

    Args:
        inp: The document as a string or bytes.
        opts: Parsing options (``removeNamespaces``, ``compactWhitespace``).

    Returns:
        The root element.

    Raises:
        ParseError: If the document is malformed, contains entity declarations or cannot be decoded.
    """

    return _parse(inp, opts)


##


def _parse(inp, opts: Optional[gws.XmlOptions] = None) -> gws.XmlElement:
    """Decode the input and run expat on it."""

    inp = _decode_input(inp)
    target = _ParserTarget(opts or gws.XmlOptions())

    parser = pyexpat.ParserCreate()
    parser.buffer_text = True
    parser.StartElementHandler = target.start
    parser.EndElementHandler = target.end
    parser.CharacterDataHandler = target.data
    parser.EntityDeclHandler = target.entity_decl

    try:
        parser.Parse(inp, True)
    except pyexpat.ExpatError as exc:
        raise error.ParseError(exc.args[0]) from exc

    if target.root is None:
        raise error.ParseError('no root element')

    return target.root


class _ParserTarget:
    """Expat event handler that builds the element tree and tracks namespace scopes."""

    def __init__(self, opts: gws.XmlOptions):
        """Create the handler.

        Args:
            opts: Parsing options.
        """

        self.stack = []
        self.scopes = []
        self.root = None
        self.opts = opts

    def make(self, tag: str, attrib: dict) -> element.XmlElement:
        """Create an element from a raw tag and raw attributes.

        Without ``removeNamespaces``, ``xmlns`` declarations are stored in ``XmlElement.namespaces``
        and pushed as a new scope, and prefixed names are resolved to Clark names.

        Args:
            tag: Tag name as in the document.
            attrib: Attributes as in the document.

        Returns:
            The new element.
        """

        if self.opts.removeNamespaces:
            el = element.XmlElement(namespace.plain_name(tag))
            for key, val in attrib.items():
                _, prefix, pname = namespace.parse_name(key)
                if key == namespace.XMLNS or prefix == namespace.XMLNS:
                    continue
                el.attrib[pname] = val
            return el

        namespaces = []
        atts = {}
        to_resolve = []

        for key, val in attrib.items():
            _, prefix, pname = namespace.parse_name(key)
            if key == namespace.XMLNS:
                namespaces.append(namespace.new('', val))
            elif prefix == namespace.XMLNS:
                namespaces.append(namespace.new(pname, val))
            elif prefix == '':
                atts[pname] = val
            else:
                to_resolve.append((prefix, pname, val))

        if namespaces:
            self.scopes.append(namespaces)

        for prefix, pname, val in to_resolve:
            atts[namespace.full_name(pname, self.uri_for(prefix))] = val

        _, prefix, pname = namespace.parse_name(tag)
        el = element.XmlElement(namespace.full_name(pname, self.uri_for(prefix)), atts)
        el.namespaces = namespaces

        return el

    def uri_for(self, prefix: str) -> str:
        """Resolve a prefix against the current namespace scopes.

        Args:
            prefix: Namespace prefix, empty for the default namespace.

        Returns:
            The namespace URI, an empty string for an undeclared default namespace,
            or ``adhoc:<prefix>`` for an undeclared prefix.
        """

        if prefix == namespace.XML:
            return namespace.XML_URI
        for scope in reversed(self.scopes):
            for ns in scope:
                if ns.prefix == prefix:
                    return ns.uri
        return '' if prefix == '' else namespace.ADHOC + prefix

    def start(self, tag: str, attrib: dict):
        """Handle a start tag.

        Args:
            tag: Tag name as in the document.
            attrib: Attributes as in the document.
        """

        el = self.make(tag, attrib)
        if self.stack:
            self.stack[-1].append(el)
        else:
            self.root = el
        self.stack.append(el)

    def end(self, tag):
        """Handle an end tag.

        Args:
            tag: Tag name as in the document.
        """

        el = self.stack.pop()
        if el.namespaces and not self.opts.removeNamespaces:
            self.scopes.pop()

    def data(self, text):
        """Handle character data, appending it to the text or the tail of the last child.

        Args:
            text: Character data.
        """

        if not self.stack:
            return
        if self.opts.compactWhitespace:
            text = ' '.join(text.strip().split())
        if not text:
            return
        top = self.stack[-1]
        if len(top) > 0:
            top[-1].tail += text
        else:
            top.text += text

    def entity_decl(self, *args):
        """Handle an entity declaration.

        Raises:
            ParseError: Always, entity declarations are not allowed.
        """

        raise error.ParseError('entity declarations are not allowed')


def _decode_input(inp) -> str:
    """Decode the input to a string without the XML declaration."""

    # A document can be declared ISO-8859-1, but actually be UTF-8 and vice versa.
    # Therefore, don't let expat do the decoding, always give it a `str`
    # and remove the xml declaration with the (possibly incorrect) encoding.

    if isinstance(inp, bytes):
        return _decode_bytes_input(inp)
    if isinstance(inp, str):
        return _decode_str_input(inp)
    raise error.ParseError(f'invalid input type {type(inp)}')


def _decode_bytes_input(inp: bytes) -> str:
    """Decode bytes as UTF-8, then with the declared encoding, then as Latin-1."""

    inp = inp.removeprefix(_BOM).strip()

    declared = ''

    if inp.startswith(b'<?xml'):
        try:
            end = inp.index(b'?>')
        except ValueError:
            raise error.ParseError('invalid XML declaration')
        head = inp[:end].decode('ascii', errors='replace').lower()
        m = re.search(r'encoding\s*=\s*(\S+)', head)
        if m:
            declared = m.group(1).strip('\'"')
        inp = inp[end + 2 :]

    # UTF-8 is strict and fails on non-UTF-8 input, Latin-1 never fails.

    encodings = ['utf-8']
    if declared and declared not in encodings:
        encodings.append(declared)
    if 'iso-8859-1' not in encodings:
        encodings.append('iso-8859-1')

    for enc in encodings:
        try:
            return inp.decode(encoding=enc, errors='strict')
        except (LookupError, UnicodeDecodeError):
            pass

    raise error.ParseError(f'invalid document encoding, tried {",".join(encodings)}')


def _decode_str_input(inp: str) -> str:
    """Strip a BOM and the XML declaration from a string."""

    inp = inp.lstrip('\ufeff').strip()
    if inp.startswith('<?xml'):
        try:
            end = inp.index('?>')
        except ValueError:
            raise error.ParseError('invalid XML declaration')
        return inp[end + 2 :]

    return inp


_BOM = b'\xef\xbb\xbf'
