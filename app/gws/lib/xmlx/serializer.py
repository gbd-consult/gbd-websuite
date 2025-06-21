from typing import Optional

import gws

from . import error, namespace, util


class Serializer:
    def __init__(self, el: gws.XmlElement, opts: Optional[gws.XmlOptions]):
        self.el = el
        self.opts = opts or gws.XmlOptions()
        self.buf = []

        self.default_namespace = self.opts.defaultNamespace
        self.namespaces = dict(self.opts.namespaces or {})

        for k, v in el.attrib.items():
            if k == namespace.XMLNS:
                ns = self.namespaces.get(v)
                if not ns:
                    raise error.NamespaceError(f'unknown default namespace {v!r}')
                self.default_namespace = ns
            elif k.startswith(namespace.XMLNS + ':'):
                ns = self.namespaces.get(v)
                if not ns:
                    raise error.NamespaceError(f'unknown namespace {v!r} for {k!r}')
                self.namespaces[k.split(':')[1]] = ns

        self.namespace_declarations = {}
        if self.default_namespace:
            self.namespace_declarations[''] = self.default_namespace
        for xmlns, ns in self.namespaces.items():
            if self.opts.xmlnsReplacements and ns.uid in self.opts.xmlnsReplacements:
                xmlns = self.opts.xmlnsReplacements[ns.uid]
            self.namespace_declarations[xmlns] = ns

        # self.extra_namespaces = kwargs.get('extra_namespaces', [])
        # self.xmlns_replacements = kwargs.get('xmlns_replacements', {})
        # self.doctype = kwargs.get('doctype')

        # self.compact_whitespace = kwargs.get('compact_whitespace', False)
        # self.fold_tags = kwargs.get('fold_tags', False)
        # self.remove_namespaces = kwargs.get('remove_namespaces', False)
        # self.with_namespace_declarations = kwargs.get('with_namespace_declarations', False)
        # self.with_schema_locations = kwargs.get('with_schema_locations', False)
        # self.with_xml_declaration = kwargs.get('with_xml_declaration', False)

    def to_string(self):
        if self.opts.withXmlDeclaration or self.opts.doctype:
            self.buf.append(_XML_DECL)
            if self.opts.doctype:
                self.buf.append(f'<!DOCTYPE {self.opts.doctype}>')

        self._el_to_string(self.el, is_root=True)

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

        if self.opts.foldTags and len(sub) == 1 and (not attr and not text and not tail):
            # single wrapper tag, create 'tag/subtag
            inner = sub[0]
            inner[0] = name + '/' + inner[0]
            return inner

        if len(sub) == 1:
            sub = sub[0]

        res = [name, attr, text, sub, tail]
        return [x for x in res if x]

    def _el_to_string(self, el, is_root=False):
        open_pos = len(self.buf)
        self.buf.append('')

        open_tag = self._make_name(el.tag)
        close_tag = open_tag

        atts = {}
        for key, val in el.attrib.items():
            if val is None:
                continue
            atts[self._make_name(key)] = self._value_to_string(val)

        s = self._text_to_string(el.text)
        if s:
            self.buf.append(s)

        for c in el:
            self._el_to_string(c)

        if is_root and self.opts.withNamespaceDeclarations:
            atts.update(
                namespace.declarations(
                    self.namespace_declarations,
                    with_schema_locations=self.opts.withSchemaLocations,
                )
            )

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

    def _text_to_string(self, arg):
        s, ok = util.atom_to_string(arg)
        if not ok:
            s = str(arg)
        if self.opts.compactWhitespace:
            s = ' '.join(s.strip().split())
        return util.escape_text(s)

    def _value_to_string(self, arg):
        s, ok = util.atom_to_string(arg)
        if not ok:
            s = str(arg)
        return util.escape_attribute(s.strip())

    def _make_name(self, name):
        if self.opts.removeNamespaces:
            return namespace.unqualify_name(name)

        xmlns, _, pname = namespace.split_name(name)
        if not xmlns or xmlns == namespace.XMLNS:
            return name

        ns = self.namespaces.get(xmlns)
        if not ns:
            ns = namespace.find_by_xmlns(xmlns)
        if not ns:
            raise error.NamespaceError(f'unknown namespace for {name!r}')

        if self.default_namespace and ns.uid == self.default_namespace.uid:
            return pname

        if self.opts.xmlnsReplacements and ns.uid in self.opts.xmlnsReplacements:
            xmlns = self.opts.xmlnsReplacements[ns.uid]

        self.namespace_declarations[xmlns] = ns
        return xmlns + ':' + pname


_XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>'
