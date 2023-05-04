"""Generate configuration references."""

import re
from . import base

STRINGS = {}

STRINGS['en'] = {
    'head_property': 'property',
    'head_type': 'type',
    'head_default': 'default',
    'head_value': 'value',

    'tag_variant': 'variant',
    'tag_struct': 'struct',
    'tag_enum': 'enumeration',
    'tag_type': 'type',
}

STRINGS['de'] = {
    'head_property': 'property',
    'head_type': 'type',
    'head_default': 'default',
    'head_value': 'value',

    'tag_variant': 'variant',
    'tag_struct': 'struct',
    'tag_enum': 'enumeration',
    'tag_type': 'type',
}

LIST_FORMAT = '<b>[</b>{}<b>]</b>'


def create(gen: base.Generator, lang: str):
    return _Creator(gen, lang).run()


##

class _Creator:
    def __init__(self, gen: base.Generator, lang: str):
        self.gen = gen
        self.lang = lang
        self.strings = STRINGS[lang]
        self.queue = []
        self.html = {}

    def run(self):
        start_tid = 'gws.base.application.Config'

        self.queue = [start_tid]
        self.html = {}

        done = set()

        while self.queue:
            tid = self.queue.pop(0)
            if tid in done:
                continue
            done.add(tid)
            self.process(tid)

        res = self.html.pop((0, start_tid))
        res += nl(v for _, v in sorted(self.html.items()))
        return res

    def process(self, tid):
        typ = self.gen.types[tid]

        if typ.c == base.C.CLASS:
            self.html[0, tid] = nl(self.process_class(tid))

        if typ.c == base.C.ENUM:
            self.html[1, tid] = nl(self.process_enum(tid))

        if typ.c == base.C.TYPE:
            target = self.gen.types[typ.tTarget]
            if target.c == base.C.VARIANT:
                self.html[2, tid] = nl(self.process_variant(tid))
            else:
                self.html[3, tid] = nl(self.process_type(tid))

        if typ.c == base.C.LIST:
            self.process(typ.tItem)

    def process_class(self, tid):
        typ = self.gen.types[tid]

        yield header(tid, self.strings['tag_struct'])
        yield f'<p>{self.docstring(tid)}</p>\n'

        rows = {False: [], True: []}

        for prop_name, prop_tid in sorted(typ.tProperties.items()):
            prop_typ = self.gen.types[prop_tid]
            self.queue.append(prop_typ.tValue)
            rows[prop_typ.hasDefault].append([
                as_propname(prop_name) + (as_required('*') if not prop_typ.hasDefault else ''),
                self.type_string(prop_typ.tValue),
                self.default_string(prop_tid),
                self.docstring(prop_tid),
            ])

        yield table(
            [self.strings['head_property'], self.strings['head_type'], self.strings['head_default'], ''],
            rows[False] + rows[True],
        )

    def process_enum(self, tid):
        typ = self.gen.types[tid]

        yield header(tid, self.strings['tag_enum'])
        yield f'<p>{self.docstring(tid)}</p>\n'
        yield table(
            [],
            [
                [as_literal(key), self.docstring(tid, key)]
                for key in typ.enumValues
            ]
        )

    def process_variant(self, tid):
        typ = self.gen.types[tid]
        target = self.gen.types[typ.tTarget]

        yield header(tid, self.strings['tag_variant'])

        rows = []
        for member_name, member_tid in sorted(target.tMembers.items()):
            self.queue.append(member_tid)
            rows.append([as_literal(member_name), self.type_string(member_tid)])

        yield table(
            [self.strings['head_type'], ''],
            rows
        )

    def process_type(self, tid):
        yield header(tid, self.strings['tag_type'])
        yield f'<p>{self.docstring(tid)}</p>\n'

    def type_string(self, tid):
        typ = self.gen.types[tid]

        if typ.c in {base.C.CLASS, base.C.TYPE, base.C.ENUM}:
            return f"<a href='#{tid}'>{as_typename(tid)}</a>"

        if typ.c == base.C.DICT:
            return as_code('dict')

        if typ.c == base.C.LIST:
            return LIST_FORMAT.format(self.type_string(typ.tItem))

        if typ.c == base.C.ATOM:
            return as_typename(tid)

        if typ.c == base.C.LITERAL:
            return ' | '.join(as_literal(s) for s in typ.literalValues)

        return ''

    def default_string(self, tid):
        typ = self.gen.types[tid]
        val = typ.tValue

        if val in self.gen.types and self.gen.types[val].c == base.C.LITERAL:
            return ''
        if not typ.hasDefault:
            return ''
        v = typ.default
        if v is None or v == '':
            return ''
        return as_literal(str(v))

    def raw_docstring(self, tid, enum_value=None):
        typ = self.gen.types[tid]
        if enum_value:
            return typ.enumDocs[enum_value]
        return typ.doc

    def docstring(self, tid, enum_value=None):
        ds = self.raw_docstring(tid, enum_value)
        ds = ds.strip().split('\n')[0]
        ds = escape(ds)
        ds = re.sub(r'`+(.+?)`+', r'<code>\1</code>', ds)
        return ds


def as_literal(s):
    return f'<code class="configref_literal">{escape(s)}</code>'


def as_typename(s):
    return f'<code class="configref_typename">{escape(s)}</code>'


def as_category(s):
    return f'<code class="configref_category">{escape(s)}</code>'


def as_propname(s):
    return f'<code class="configref_propname">{escape(s)}</code>'


def as_required(s):
    return f'<code class="configref_required">{escape(s)}</code>'


def as_code(s):
    return f'<code>{escape(s)}</code>'


def header(tid, category):
    return f'<h5 data-url="#{tid}" id="{tid}">{tid} {as_category(category)}</h5>\n'


def table(heads, rows):
    h = ''
    if heads:
        h = '<thead><tr>' + nl(f'<th>{h}</th>' for h in heads) + '</tr></thead>\n'
    b = nl(
        '<tr>' + nl(f'<td>{c}</td>' for c in row) + '</tr>'
        for row in rows
    )
    return f'<table>\n{h}<tbody>{b}</tbody></table>\n'


def escape(s, quote=True):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    if quote:
        s = s.replace('"', "&quot;")
    return s


nl = '\n'.join
