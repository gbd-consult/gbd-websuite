import gws
from . import namespace, error


class Serializer:
    def __init__(self, el: gws.XmlElement, **kwargs):
        self.el = el

        self.extra_namespaces = kwargs.get('extra_namespaces', [])
        self.xmlns_replacements = kwargs.get('xmlns_replacements', {})
        self.doctype = kwargs.get('doctype')

        self.compact_whitespace = kwargs.get('compact_whitespace', False)
        self.fold_tags = kwargs.get('fold_tags', False)
        self.remove_namespaces = kwargs.get('remove_namespaces', False)
        self.with_namespace_declarations = kwargs.get('with_namespace_declarations', False)
        self.with_schema_locations = kwargs.get('with_schema_locations', False)
        self.with_xml_declaration = kwargs.get('with_xml_declaration', False)

        self.default_namespace = None

        xmlns = el.get('xmlns')
        if xmlns:
            ns = namespace.get(xmlns)
            if not ns:
                raise error.NamespaceError(f'unknown namespace {xmlns!r}')
            self.default_namespace = ns

        self.buf = []

    def to_string(self):
        extra_atts = None

        if self.with_namespace_declarations:
            extra_atts = namespace.declarations(
                for_element=self.el,
                default_namespace=self.default_namespace,
                xmlns_replacements=self.xmlns_replacements,
                extra_namespaces=self.extra_namespaces,
                with_schema_locations=self.with_schema_locations,
            )

        if self.with_xml_declaration:
            self.buf.append(_XML_DECL)
            if self.doctype:
                self.buf.append(f'<!DOCTYPE {self.doctype}>')

        self._el_to_string(self.el, extra_atts)

        return ''.join(self.buf)

    def to_list(self):
        return self._el_to_list(self.el)

    ##

    def _el_to_list(self, el):
        name = self._make_name(el.tag)
        attr = {self._make_name(k): v for k, v in el.attrib.items()}
        text = (el.text or '').strip()
        tail = (el.tail or '').strip()

        sub = [self._el_to_list(c) for c in el]

        if self.fold_tags and len(sub) == 1 and (not attr and not text and not tail):
            # single wrapper tag, create 'tag/subtag
            inner = sub[0]
            inner[0] = name + '/' + inner[0]
            return inner

        res = [name, attr, text, sub, tail]
        return [x for x in res if x]

    def _el_to_string(self, el, extra_atts=None):
        atts = {}

        for key, val in el.attrib.items():
            if val is None:
                continue
            atts[self._make_name(key)] = self._value_to_string(val)

        if extra_atts:
            atts.update(extra_atts)

        open_pos = len(self.buf)
        self.buf.append('')

        s = self._text_to_string(el.text)
        if s:
            self.buf.append(s)

        for c in el:
            self._el_to_string(c)

        open_tag = self._make_name(el.tag)
        close_tag = open_tag
        if atts:
            open_tag += ' ' + ' '.join(f'{k}="{v}"' for k, v in atts.items())

        if len(self.buf) > open_pos + 1:
            self.buf[open_pos] = f'<{open_tag}>'
            self.buf.append(f'</{close_tag}>')
        else:
            self.buf[open_pos] += f'<{open_tag}/>'

        s = self._text_to_string(el.tail)
        if s:
            self.buf.append(s)

    def _text_to_string(self, s):
        if s is None:
            return ''
        if isinstance(s, (int, float, bool)):
            return str(s).lower()
        if not isinstance(s, str):
            s = str(s)
        if self.compact_whitespace:
            s = ' '.join(s.strip().split())
        s = s.replace("&", "&amp;")
        s = s.replace(">", "&gt;")
        s = s.replace("<", "&lt;")
        return s

    def _value_to_string(self, s):
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

    def _make_name(self, name):
        if self.remove_namespaces:
            return namespace.unqualify_name(name)
        ns, pname = namespace.extract(name)
        if not ns:
            return name
        if ns and self.default_namespace and ns.uid == self.default_namespace.uid:
            return pname
        if self.xmlns_replacements and ns.uid in self.xmlns_replacements:
            return self.xmlns_replacements[ns.uid] + ':' + pname
        return ns.xmlns + ':' + pname


_XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>'
