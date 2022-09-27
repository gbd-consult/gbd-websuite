import xml.etree.ElementTree

import gws
import gws.types as t

from . import namespace


class XElement(xml.etree.ElementTree.Element):
    caseInsensitive: bool

    def __init__(self, tag, attrib=None, **extra):
        super().__init__(tag, attrib or {}, **extra)

    def __bool__(self):
        return True

    def __iter__(self) -> t.Iterator['XElement']:
        # need this for typing of `for sub in elem` loops
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
        return super().get(self._convert_key(key), default)

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
            compact_whitespace=False,
            remove_namespaces=False,
            with_namespace_declarations=False,
            with_schema_locations=False,
            with_xml_declaration=False,
    ):
        def text(s):
            if s is None:
                return ''
            if isinstance(s, (int, float, bool)):
                return str(s).lower()
            if not isinstance(s, str):
                s = str(s)
            if compact_whitespace:
                s = ' '.join(s.strip().split())
            return _encode(s)

        def convert_name(name):
            if remove_namespaces:
                return namespace.unqualify(name)
            return namespace.qualify(name)

        def to_str(el, with_xmlns):

            atts = {}

            for key, val in el.items():
                if val is None:
                    continue
                atts[convert_name(key)] = text(val)

            if with_xmlns:
                atts.update(
                    namespace.declarations(
                        for_element=el,
                        default_prefix=el.get('xmlns'),
                        with_schema_locations=with_schema_locations))

            head_pos = len(buf)
            buf.append('')

            s = text(el.text)
            if s:
                buf.append(s)

            for ch in el:
                to_str(ch, False)

            tag = convert_name(el.tag)
            head = tag
            if atts:
                head += ' ' + ' '.join(f'{k}="{v}"' for k, v in atts.items())

            if len(buf) > head_pos + 1:
                buf[head_pos] = '<' + head + '>'
                buf.append('</' + tag + '>')
            else:
                buf[head_pos] += '<' + head + '/>'

            s = text(el.tail)
            if s:
                buf.append(s)

        ##

        buf = ['']
        to_str(self, with_namespace_declarations)
        if with_xml_declaration:
            buf[0] = _XML_DECL
        return ''.join(buf)

    ##

    def children(self):
        for c in self:
            yield c

    def first_of(self, *paths):
        if not paths:
            return self[0] if len(self) > 0 else None
        for path in paths:
            el = self.find(path)
            if el is not None:
                return el

    def text_of(self, *paths):
        for path in paths:
            el = self.find(path)
            if el is not None and el.text:
                return el.text

    def text_list(self, *paths, deep=False):
        buf = self._collect_text(paths, deep)
        return [text for _, text in buf]

    def text_dict(self, *paths, deep=False):
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


##


def _encode(v) -> str:
    s = str(v)
    s = s.replace("&", "&amp;")
    s = s.replace(">", "&gt;")
    s = s.replace("<", "&lt;")
    s = s.replace('"', "&quot;")
    return s


_XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>'
