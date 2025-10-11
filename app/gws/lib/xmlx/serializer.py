from typing import Optional

import gws

from . import error, namespace, util


class Serializer:
    def __init__(self, el: gws.XmlElement, opts: Optional[gws.XmlOptions]):
        self.root = el
        self.buf = []

        self.opts = opts or gws.XmlOptions()
        self.defaultNamespace = self.opts.defaultNamespace

        self.nsIndexXmlns = {}
        self.nsIndexUri = {}
        if self.opts.namespaces:
            for xmlns, ns in self.opts.namespaces.items():
                self.nsIndexXmlns[xmlns] = ns
                self.nsIndexUri[ns.uri] = ns

    def to_string(self):
        if self.opts.withXmlDeclaration or self.opts.doctype:
            self.buf.append(_XML_DECL)
            if self.opts.doctype:
                self.buf.append(f'<!DOCTYPE {self.opts.doctype}>')

        self._el_to_string(self.root, is_root=True)

        return ''.join(self.buf)

    def to_list(self):
        return self._el_to_list(self.root)

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

        if el.attrib:
            atts = self._process_atts(el.attrib)
        else:
            atts = {}

        s = self._text_to_string(el.text)
        if s:
            self.buf.append(s)

        for c in el:
            self._el_to_string(c)

        if is_root and self.opts.withNamespaceDeclarations:
            atts.update(self._namespace_declarations())

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

    # def _process_root_atts(self, attrib):
    #     atts = {}

    #     for key, val in attrib.items():
    #         if key == namespace.XMLNS:
    #             if self.opts.removeNamespaces:
    #                 continue
    #             ns = namespace.find_by_uri(val)
    #             if not ns:
    #                 raise error.NamespaceError(f'unknown default namespace {val!r}')
    #             self.defaultNamespace = ns
    #             continue

    #         if key.startswith(namespace.XMLNS + ':'):
    #             if self.opts.removeNamespaces:
    #                 continue
    #             ns = namespace.find_by_uri(val)
    #             if not ns:
    #                 raise error.NamespaceError(f'unknown namespace {val!r} for {key!r}')
    #             self.namespaceMap[key.split(':')[1]] = ns
    #             continue

    #         if val is None:
    #             continue
    #         n = self._make_name(key)
    #         if n:
    #             atts[n] = self._value_to_string(val)

    #     return atts

    def _process_atts(self, attrib):
        atts = {}

        for key, val in attrib.items():
            if val is None:
                continue
            n = self._make_name(key)
            if n:
                atts[n] = self._value_to_string(val)

        return atts

    def _namespace_declarations(self):
        xmlns_to_ns = {}

        if self.defaultNamespace:
            xmlns_to_ns[''] = self.defaultNamespace

        for xmlns, ns in self.nsIndexXmlns.items():
            if self.defaultNamespace and ns.uid == self.defaultNamespace.uid:
                continue
            if self.opts.customXmlns and ns.uid in self.opts.customXmlns:
                xmlns = self.opts.customXmlns[ns.uid]
            xmlns_to_ns[xmlns] = ns

        return namespace.declarations(
            xmlns_to_ns,
            with_schema_locations=self.opts.withSchemaLocations,
        )

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

        xmlns, uri, pname = namespace.split_name(name)
        if not xmlns and not uri:
            return pname

        ns = None
        if xmlns:
            if xmlns == namespace.XMLNS:
                return name
            ns = self.nsIndexXmlns.get(xmlns) or namespace.find_by_xmlns(xmlns)
        else:
            ns = self.nsIndexUri.get(uri) or namespace.find_by_uri(uri)

        if not ns:
            raise error.NamespaceError(f'unknown namespace for {name!r}')

        if self.defaultNamespace and ns.uid == self.defaultNamespace.uid:
            return pname

        xmlns = ns.xmlns or ns.uid
        if self.opts.customXmlns and ns.uid in self.opts.customXmlns:
            xmlns = self.opts.customXmlns[ns.uid]

        self.nsIndexXmlns[xmlns] = ns
        if uri:
            self.nsIndexUri[uri] = ns
        
        return xmlns + ':' + pname


_XML_DECL = '<?xml version="1.0" encoding="UTF-8"?>'
