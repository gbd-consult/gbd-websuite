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

    'label_added': 'added',
    'label_deprecated': 'deprecated',
    'label_changed': 'changed',

}

STRINGS['de'] = {
    'head_property': 'Eigenschaft',
    'head_type': 'Typ',
    'head_default': 'Default',
    'head_value': 'Wert',

    'tag_variant': 'variant',
    'tag_struct': 'struct',
    'tag_enum': 'enumeration',
    'tag_type': 'type',

    'label_added': 'neu',
    'label_deprecated': 'veraltet',
    'label_changed': 'geändert',
}

LIST_FORMAT = '**[**{}**]**'

LABELS = 'added|deprecated|changed'


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
        start_tid = 'gws.base.application.core.Config'

        self.queue = [start_tid]
        self.html = {}

        done = set()

        while self.queue:
            tid = self.queue.pop(0)
            if tid in done:
                continue
            done.add(tid)
            self.process(tid)

        res = self.html.pop(start_tid.lower())
        res += nl(v for _, v in sorted(self.html.items()))
        return res

    def process(self, tid):
        typ = self.gen.types[tid]

        if typ.c == base.C.CLASS:
            self.html[tid.lower()] = nl(self.process_class(tid))

        if typ.c == base.C.ENUM:
            self.html[tid.lower()] = nl(self.process_enum(tid))

        if typ.c == base.C.TYPE:
            target = self.gen.types[typ.tTarget]
            if target.c == base.C.VARIANT:
                self.html[tid.lower()] = nl(self.process_variant(tid))
            else:
                self.html[tid.lower()] = nl(self.process_type(tid))

        if typ.c == base.C.LIST:
            self.process(typ.tItem)

    def process_class(self, tid):
        typ = self.gen.types[tid]

        yield header(tid)
        yield subhead(self.strings['tag_struct'], self.docstring(tid))

        rows = {False: [], True: []}

        for prop_name, prop_tid in sorted(typ.tProperties.items()):
            prop_typ = self.gen.types[prop_tid]
            self.queue.append(prop_typ.tValue)
            rows[prop_typ.hasDefault].append([
                as_propname(prop_name) if prop_typ.hasDefault else as_required(prop_name),
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

        yield header(tid)
        yield subhead(self.strings['tag_enum'], self.docstring(tid))
        yield table(
            ['', ''],
            [
                [as_literal(key), self.docstring(tid, key)]
                for key in typ.enumValues
            ]
        )

    def process_variant(self, tid):
        typ = self.gen.types[tid]
        target = self.gen.types[typ.tTarget]

        yield header(tid)
        yield subhead(self.strings['tag_variant'], '')

        rows = []
        for member_name, member_tid in sorted(target.tMembers.items()):
            self.queue.append(member_tid)
            rows.append([as_literal(member_name), self.type_string(member_tid)])

        yield table(
            [self.strings['head_type'], ''],
            rows
        )

    def process_type(self, tid):
        yield header(tid)
        yield subhead(self.strings['tag_type'], self.docstring(tid))

    def type_string(self, tid):
        typ = self.gen.types[tid]

        if typ.c in {base.C.CLASS, base.C.TYPE, base.C.ENUM}:
            return link(tid, as_typename(tid))

        if typ.c == base.C.DICT:
            return as_code('dict')

        if typ.c == base.C.LIST:
            return LIST_FORMAT.format(self.type_string(typ.tItem))

        if typ.c == base.C.ATOM:
            return as_typename(tid)

        if typ.c == base.C.LITERAL:
            return r' \| '.join(as_literal(s) for s in typ.literalValues)

        return ''

    def default_string(self, tid):
        typ = self.gen.types[tid]
        val = typ.tValue

        if val in self.gen.types and self.gen.types[val].c == base.C.LITERAL:
            return ''
        if not typ.hasDefault:
            return ''
        v = typ.defaultValue
        if v is None or v == '':
            return ''
        return as_literal(v)

    def docstring(self, tid, enum_value=None):
        missing_translation = ''

        # get the original (spec) docstring
        typ = self.gen.types[tid]
        spec_text = first_line(typ.enumDocs.get(enum_value) if enum_value else typ.doc)

        # try the translated (from strings) docstring
        key = tid
        if enum_value:
            key += '.' + enum_value
        local_text = self.gen.strings[self.lang].get(key)

        if not local_text and self.lang != 'en':
            # translation missing: use the english docstring and warn
            base.log.debug(f'missing {self.lang} translation for {key!r}')
            missing_translation = f'`{key}`{{.configref_missing_translation}}'
            local_text = self.gen.strings['en'].get(key)

        local_text = first_line(local_text) or spec_text

        # process a label, like "foobar (added: 8.1)"
        # it might be missing in a translation, but present in the original (spec) docstring
        text, label = self.extract_label(local_text)
        if not label and spec_text != local_text:
            _, label = self.extract_label(spec_text)

        return text + label + missing_translation

    def extract_label(self, text):
        m = re.match(fr'(.+?)\(({LABELS}):\s*([\d.]+)\)$', text)
        if not m:
            return text, ''
        kind = m.group(2).strip()
        name = self.strings[f'label_{kind}']
        version = m.group(3)
        label = f'`{name}: {version}`{{.configref_label_{kind}}}'
        return m.group(1).strip(), label


def as_literal(s):
    return f'`{s!r}`{{.configref_literal}}'


def as_typename(s):
    return f'`{s}`{{.configref_typename}}'


def as_category(s):
    return f'`{s}`{{.configref_category}}'


def as_propname(s):
    return f'`{s}`{{.configref_propname}}'


def as_required(s):
    return f'`{s}`{{.configref_required}}'


def as_code(s):
    return f'`{s}`'


def header(tid):
    return f'\n## {tid} :{tid}\n'


def subhead(category, text):
    return as_category(category) + ' ' + text + '\n'


def link(target, text):
    return f'[{text}](../{target})'


def first_line(s):
    return (s or '').strip().split('\n')[0].strip()


def table(heads, rows):
    widths = [len(h) for h in heads]

    for r in rows:
        widths = [
            max(a, b)
            for a, b in zip(
                widths,
                [len(str(s)) for s in r]
            )
        ]

    def field(n, v):
        return str(v).ljust(widths[n])

    def row(r):
        return ' | '.join(field(n, v) for n, v in enumerate(r))

    out = [row(heads), '', *[row(r) for r in rows]]
    out[1] = '-' * len(out[0])
    return '\n'.join(f'| {s} |' for s in out) + '\n'


def escape(s, quote=True):
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    if quote:
        s = s.replace('"', "&quot;")
    return s


nl = '\n'.join
