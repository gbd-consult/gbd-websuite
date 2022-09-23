import re
import xml.etree.ElementTree as ET

import gws.types as t

from . import error, element


def from_path(
        path: str,
        case_insensitive=False,
        compact_whitespace=False,
        remove_namespaces=False
) -> element.XElement:
    with open(path, 'rb') as fp:
        inp = fp.read()
    return _parse(inp, case_insensitive, compact_whitespace, remove_namespaces)


def from_string(
        inp: t.Union[str, bytes],
        case_insensitive=False,
        compact_whitespace=False,
        remove_namespaces=False
) -> element.XElement:
    return _parse(inp, case_insensitive, compact_whitespace, remove_namespaces)


def _parse(text, case_insensitive, compact_whitespace, remove_namespaces) -> element.XElement:
    inp = _decode_input(text)
    parser = ET.XMLParser(target=_ParserTarget(case_insensitive, compact_whitespace, remove_namespaces))
    try:
        parser.feed(inp)
        return parser.close()
    except ET.ParseError as exc:
        raise error.ParseError(exc.args[0]) from exc


class _ParserTarget:
    def __init__(self, case_insensitive, compact_whitespace, remove_namespaces):
        self.stack = []
        self.root = None
        self.buf = []
        self.remove_namespaces = remove_namespaces
        self.case_insensitive = case_insensitive
        self.compact_whitespace = compact_whitespace

    def make(self, tag, attrib):
        attrib2 = {}

        if attrib:
            for name, val in attrib.items():
                if self.remove_namespaces:
                    _, name = _split_ns(name)
                if self.case_insensitive:
                    name = name.lower()
                attrib2[name] = val

        if self.remove_namespaces:
            _, tag = _split_ns(tag)
        if self.case_insensitive:
            tag = tag.lower()

        el = element.XElement(tag, attrib2)
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


def _split_ns(tag):
    if tag[0] != '{':
        return None, tag
    return tag[1:].split('}')


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
