"""XML parser."""

import re
import xml.etree.ElementTree

import gws
import gws.types as t

from . import error, element, namespace


def from_path(
        path: str,
        case_insensitive=False,
        compact_whitespace=False,
        normalize_namespaces=False,
        remove_namespaces=False,
) -> gws.IXmlElement:
    with open(path, 'rb') as fp:
        inp = fp.read()
    return _parse(inp, case_insensitive, compact_whitespace, normalize_namespaces, remove_namespaces)


def from_string(
        inp: str | bytes,
        case_insensitive=False,
        compact_whitespace=False,
        remove_namespaces=False,
        normalize_namespaces=False,
) -> gws.IXmlElement:
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

    def convert_name(self, s):
        if s[0] != '{':
            return s.lower() if self.case_insensitive else s
        uri, name = s[1:].split('}')
        if self.case_insensitive:
            name = name.lower()
        if self.remove_namespaces:
            return name
        if self.normalize_namespaces:
            pfx = namespace.prefix_for_uri(uri)
            if pfx:
                uri = namespace.uri_for_prefix(pfx)
        return '{' + uri + '}' + name

    def make(self, tag, attrib):
        attrib2 = {}

        if attrib:
            for name, val in attrib.items():
                attrib2[self.convert_name(name)] = val

        el = element.XElement(self.convert_name(tag), attrib2)
        el.caseInsensitive = self.case_insensitive

        return el

    def flush(self):
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

    def start(self, tag, attrib):
        self.flush()
        el = self.make(tag, attrib)
        if self.stack:
            self.stack[-1].append(el)
        else:
            self.root = el
        self.stack.append(el)

    def end(self, tag):
        self.flush()
        self.stack.pop()

    def data(self, data):
        self.buf.append(data)

    def close(self):
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
