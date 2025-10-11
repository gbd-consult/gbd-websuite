"""XML parser."""

from typing import Optional, cast

import re
import xml.etree.ElementTree

import gws

from . import error, element, namespace


def from_path(path: str, opts: Optional[gws.XmlOptions] = None) -> gws.XmlElement:
    """Creates an ``XmlElement`` object from a .xml file.

    Args:
        path: Path to the .xml file.
        opts: XML options.
    """

    with open(path, 'rb') as fp:
        inp = fp.read()
    return _parse(inp, opts)


def from_string(inp: str | bytes, opts: Optional[gws.XmlOptions] = None) -> gws.XmlElement:
    """Creates an ``XmlElement`` from a string or bytes.

    Args:
        inp: .xml file as a string or bytes.
        opts: XML options.
    """

    return _parse(inp, opts)


##


def _parse(inp, opts: Optional[gws.XmlOptions] = None) -> gws.XmlElement:
    inp2 = _decode_input(inp)
    parser = xml.etree.ElementTree.XMLParser(target=_ParserTarget(opts or gws.XmlOptions()))
    try:
        parser.feed(inp2)
        return cast(gws.XmlElement, parser.close())
    except xml.etree.ElementTree.ParseError as exc:
        raise error.ParseError(exc.args[0]) from exc


class _ParserTarget:
    def __init__(self, opts: gws.XmlOptions):
        self.stack = []
        self.root = None
        self.buf = []
        self.opts = opts

    def convert_name(self, s: str) -> str:
        xmlns, uri, pname = namespace.split_name(s)
        pname = pname.lower() if self.opts.caseInsensitive else pname
        if self.opts.removeNamespaces:
            return pname
        if not xmlns and not uri:
            return pname
        if uri:
            return '{' + uri + '}' + pname
        return pname

    def make(self, tag: str, attrib: dict) -> gws.XmlElement:
        attrib2 = {}

        if attrib:
            for name, val in attrib.items():
                attrib2[self.convert_name(name)] = val

        el = element.XmlElement(self.convert_name(tag), attrib2)
        return el

    def flush(self):
        if not self.buf:
            return

        text = ''.join(self.buf)
        self.buf = []

        if self.opts.compactWhitespace:
            text = ' '.join(text.strip().split())

        if text:
            top = self.stack[-1]
            if len(top) > 0:
                top[-1].tail = text
            else:
                top.text = text

    def start(self, tag: str, attrib: dict):
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
        return _decode_bytes_input(inp)
    if isinstance(inp, str):
        return _decode_str_input(inp)
    raise error.ParseError(f'invalid input type {type(inp)}')


def _decode_bytes_input(inp: bytes) -> str:
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
            encodings.append(m.group(1).strip('\'"'))
        inp = inp[end + 2 :]

    # try the declared encoding, if any, then utf8, then latin

    if 'utf-8' not in encodings:
        encodings.append('utf-8')
    if 'iso-8859-1' not in encodings:
        encodings.append('iso-8859-1')

    for enc in encodings:
        try:
            return inp.decode(encoding=enc, errors='strict')
        except (LookupError, UnicodeDecodeError):
            pass

    raise error.ParseError(f'invalid document encoding, tried {",".join(encodings)}')


def _decode_str_input(inp: str) -> str:
    inp = inp.strip()

    if inp.startswith('<?xml'):
        try:
            end = inp.index('?>')
        except ValueError:
            raise error.ParseError('invalid XML declaration')
        return inp[end + 2 :]

    return inp
