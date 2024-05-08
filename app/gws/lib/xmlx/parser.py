"""XML parser."""

import re
import xml.etree.ElementTree

import gws

from . import error, element, namespace


def from_path(
        path: str,
        case_insensitive: bool = False,
        compact_whitespace: bool = False,
        normalize_namespaces: bool = False,
        remove_namespaces: bool = False,
) -> gws.XmlElement:
    """Creates an ``IXmlElement`` object from a .xlm file.

    Args:
        path: Path to the .xml file.
        case_insensitive: If true tags will be written in lowercase into the IXmlElement object.
        compact_whitespace: If true all whitespaces and newlines are omitted.
        normalize_namespaces:
        remove_namespaces: Removes all occurrences of namespaces.

    Returns:
        The ``IXmlElement`` object.
    """
    with open(path, 'rb') as fp:
        inp = fp.read()
    return _parse(inp, case_insensitive, compact_whitespace, normalize_namespaces, remove_namespaces)


def from_string(
        inp: str | bytes,
        case_insensitive: bool = False,
        compact_whitespace: bool = False,
        remove_namespaces: bool = False,
        normalize_namespaces: bool = False,
) -> gws.XmlElement:
    """Creates an ``IXmlElement`` from a string or bytes.

    Args:
        inp: .xml file as a string or bytes.
        case_insensitive: If true tags will be written in lowercase into the IXmlElement object.
        compact_whitespace: If true all whitespaces and newlines are omitted.
        normalize_namespaces:
        remove_namespaces: Removes all occurrences of namespaces.

    Returns:
        The ``IXmlElement`` object.
    """
    return _parse(inp, case_insensitive, compact_whitespace, normalize_namespaces, remove_namespaces)


##


def _parse(inp, case_insensitive, compact_whitespace, normalize_namespaces, remove_namespaces):
    inp2 = _decode_input(inp)
    parser = xml.etree.ElementTree.XMLParser(
        target=_ParserTarget(case_insensitive, compact_whitespace, normalize_namespaces, remove_namespaces))
    try:
        parser.feed(inp2)
        return parser.close()
    except xml.etree.ElementTree.ParseError as exc:
        raise error.ParseError(exc.args[0]) from exc


class _ParserTarget:
    def __init__(self, case_insensitive, compact_whitespace, normalize_namespaces, remove_namespaces):
        self.stack = []
        self.root = None
        self.buf = []
        self.case_insensitive = case_insensitive
        self.compact_whitespace = compact_whitespace
        self.remove_namespaces = remove_namespaces
        self.normalize_namespaces = normalize_namespaces

    def convert_name(self, s: str) -> str:
        """"Converts a given XML-namespace or URI to a proper name.

        Args:
            s: XML-namespace or URI.

        Returns:
            ``{URI}properName``, if ``normalize_namespaces`` flag is True  ``{non-versionalized-URL}properName`` is returned, if ``remove_namespaces`` flag is True ``properName`` is returned.
        """
        xmlns, uri, pname = namespace.split_name(s)
        pname = pname.lower() if self.case_insensitive else pname
        if not xmlns and not uri:
            return pname
        if self.remove_namespaces:
            return pname
        if self.normalize_namespaces:
            ns = namespace.find_by_uri(uri)
            if ns:
                uri = ns.uri
        return '{' + uri + '}' + pname

    def make(self, tag: str, attrib: dict) -> gws.XmlElement:
        """Creates an ``IXmlElement``.

        Args:
            tag: The tag.
            attrib: ``{key:value}``

        Returns:
            A ``IXmlElement.``
        """
        attrib2 = {}

        if attrib:
            for name, val in attrib.items():
                attrib2[self.convert_name(name)] = val

        el = element.XmlElementImpl(self.convert_name(tag), attrib2)
        el.caseInsensitive = self.case_insensitive

        return el

    def flush(self):
        """Loads the buffer into the stack and clears the stack."""
        if not self.buf:
            return

        text = ''.join(self.buf)
        self.buf = []

        if self.compact_whitespace:
            text = ' '.join(text.strip().split())

        if text:
            top = self.stack[-1]
            if len(top) > 0:
                top[-1].tail = text
            else:
                top.text = text

    ##

    def start(self, tag: str, attrib: dict):
        """Flushes the buffer and appends an element to the stack.

        Args:
            tag: Tag of the XML-element.
            attrib: Attribute of the XML-element.
        """
        self.flush()
        el = self.make(tag, attrib)
        if self.stack:
            self.stack[-1].append(el)
        else:
            self.root = el
        self.stack.append(el)

    def end(self, tag):
        """Flushes the buffer and pops the stack."""
        self.flush()
        self.stack.pop()

    def data(self, data):
        """Adds data to the buffer.

        Args:
            data: data to add."""
        self.buf.append(data)

    def close(self):
        """Returns root."""
        return self.root


def _decode_input(inp) -> str:
    # the problem is, we can receive a document
    # that is declared ISO-8859-1, but actually is UTF and vice versa.
    # therefore, don't let expat do the decoding, always give it a `str`
    # and remove the xml decl with the (possibly incorrect) encoding

    if isinstance(inp, bytes):
        inp = inp.strip()

        encodings = []

        if inp.startswith(b'<?xml'):
            try:
                end = inp.index(b'?>')
            except ValueError:
                raise error.ParseError('invalid XML declaration')

            head = inp[:end].decode('ascii').lower()
            m = re.search(r'encoding\s*=\s*(\S+)', head)
            if m:
                encodings.append(m.group(1).strip('\'\"'))
            inp = inp[end + 2:]

        # try the declared encoding, if any, then utf8, then latin

        if 'utf8' not in encodings:
            encodings.append('utf8')
        if 'iso-8859-1' not in encodings:
            encodings.append('iso-8859-1')

        for enc in encodings:
            try:
                return inp.decode(encoding=enc, errors='strict')
            except (LookupError, UnicodeDecodeError):
                pass

        raise error.ParseError(f'invalid document encoding, tried {",".join(encodings)}')

    if isinstance(inp, str):
        inp = inp.strip()

        if inp.startswith('<?xml'):
            try:
                end = inp.index('?>')
            except ValueError:
                raise error.ParseError('invalid XML declaration')
            return inp[end + 2:]

        return inp

    raise error.ParseError(f'invalid input')
