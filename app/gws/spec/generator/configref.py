"""Generate the configuration reference in Markdown."""

import re
import json

from . import base

STRINGS = {}

STRINGS['en'] = {
    'head_property': 'property',
    'head_variant': 'one of the following objects:',
    'head_type': 'type',
    'head_default': 'default',
    'head_value': 'value',
    'head_member': 'class',
    'category_variant': 'variant',
    'category_object': 'obj',
    'category_enum': 'enum',
    'category_type': 'type',
    'label_added': 'added',
    'label_deprecated': 'deprecated',
    'label_changed': 'changed',
}

STRINGS['de'] = {
    'head_property': 'Eigenschaft',
    'head_variant': 'Eines der folgenden Objekte:',
    'head_type': 'Typ',
    'head_default': 'Default',
    'head_value': 'Wert',
    'head_member': 'Objekt',
    'category_variant': 'variant',
    'category_object': 'obj',
    'category_enum': 'enum',
    'category_type': 'type',
    'label_added': 'neu',
    'label_deprecated': 'veraltet',
    'label_changed': 'geändert',
}

LIST_FORMAT = '<nobr>{}**[ ]**</nobr>'
DEFAULT_FORMAT = ' _{}:_ {}.'

LABELS = 'added|deprecated|changed'


def create(gen: base.Generator, lang: str):
    """Create the configuration reference.

    The reference starts with the application ``Config`` and contains a
    section for each reachable class, type alias, enum and variant.

    Args:
        gen: Generator state, with strings already collected.
        lang: Language code, ``en`` or ``de``.

    Returns:
        The reference as Markdown text.
    """

    return _Creator(gen, lang).run()


##


class _Creator:
    """Builds the reference by walking the types from the application config."""

    start_tid = 'gws.base.application.core.Config'
    exclude_props = ['uid', 'access', 'type']

    def __init__(self, gen: base.Generator, lang: str):
        self.gen = gen
        self.lang = lang
        self.strings = STRINGS[lang]
        self.queue = []
        self.blocks = []

    def run(self):
        """Create the reference.

        Returns:
            The Markdown text, with sections sorted by kind and uid.
        """

        self.queue = [self.start_tid]
        self.blocks = []
        done = set()

        while self.queue:
            tid = self.queue.pop(0)
            if tid in done:
                continue
            done.add(tid)
            self.process(tid)

        return nl(b[-1] for b in sorted(self.blocks))

    def process(self, tid):
        """Create the section for a type and enqueue the types it refers to.

        Args:
            tid: Type uid.
        """

        typ = self.gen.require_type(tid)

        if typ.c == base.c.CLASS:
            key = 0 if tid == self.start_tid else 1
            self.blocks.append([key, tid.lower(), nl(self.process_class(tid))])

        if typ.c == base.c.TYPE:
            self.blocks.append([2, tid.lower(), nl(self.process_type(tid))])

        if typ.c == base.c.ENUM:
            self.blocks.append([3, tid.lower(), nl(self.process_enum(tid))])

        if typ.c == base.c.VARIANT:
            self.blocks.append([4, tid.lower(), nl(self.process_variant(tid))])

        if typ.c == base.c.LIST:
            self.queue.append(typ.tItem)

    def process_class(self, tid):
        """Create the section for a class, with a table of its properties.

        Args:
            tid: Type uid.

        Yields:
            Markdown blocks.
        """

        typ = self.gen.require_type(tid)

        yield header('object', tid)
        yield subhead(self.strings['category_object'], self.docstring_as_header(tid))

        rows = {False: [], True: []}

        for prop_name, prop_tid in sorted(typ.tProperties.items()):
            if prop_name in self.exclude_props:
                continue
            prop_typ = self.gen.require_type(prop_tid)
            self.queue.append(prop_typ.tValue)
            rows[prop_typ.hasDefault].append(
                [
                    as_propname(prop_name) if prop_typ.hasDefault else as_required(prop_name),
                    self.type_string(prop_typ.tValue),
                    self.docstring_as_cell(prop_tid),
                ]
            )

        yield table(
            [
                self.strings['head_property'],
                self.strings['head_type'],
                '',
            ],
            rows[False] + rows[True],
        )

    def process_enum(self, tid):
        """Create the section for an enum, with a table of its values.

        Args:
            tid: Type uid.

        Yields:
            Markdown blocks.
        """

        typ = self.gen.require_type(tid)

        yield header('enum', tid)
        yield subhead(self.strings['category_enum'], self.docstring_as_header(tid))
        yield table(
            [
                self.strings['head_value'],
                '',
            ],
            [[as_literal(key), self.docstring_as_cell(tid, key)] for key in typ.enumValues],
        )

    def process_variant(self, tid):
        """Create the section for a variant, with a table of its members.

        Args:
            tid: Type uid.

        Yields:
            Markdown blocks.
        """

        typ = self.gen.require_type(tid)

        yield header('variant', tid)
        yield subhead(self.strings['category_variant'], self.strings['head_variant'])

        rows = []
        for member_name, member_tid in sorted(typ.tMembers.items()):
            self.queue.append(member_tid)
            rows.append([as_literal(member_name), self.type_string(member_tid)])

        yield table(
            [
                self.strings['head_type'],
                '',
            ],
            rows,
        )

    def process_type(self, tid):
        """Create the section for a type alias.

        Args:
            tid: Type uid.

        Yields:
            Markdown blocks.
        """

        yield header('type', tid)
        yield subhead(self.strings['category_type'], self.docstring_as_header(tid))

    def type_string(self, tid):
        """Format a type for a table cell.

        Args:
            tid: Type uid.

        Returns:
            A link for named types, a formatted name for other types.
        """

        typ = self.gen.require_type(tid)

        if typ.c in {base.c.CLASS, base.c.TYPE, base.c.ENUM, base.c.VARIANT}:
            return link(tid, as_typename(tid))

        if typ.c == base.c.DICT:
            return as_code('dict')

        if typ.c == base.c.LIST:
            return LIST_FORMAT.format(self.type_string(typ.tItem))

        if typ.c == base.c.ATOM:
            return as_typename(tid)

        if typ.c == base.c.LITERAL:
            return r'  | '.join(as_literal(s) for s in typ.literalValues)

        return typ.c

    def default_string(self, tid):
        """Format the default value of a property.

        Args:
            tid: Property type uid.

        Returns:
            The formatted default, or an empty string if there is no default,
            it is empty, or the property type is a literal.
        """

        typ = self.gen.require_type(tid)
        val = typ.tValue

        if val in self.gen.typeDict and self.gen.typeDict[val].c == base.c.LITERAL:
            return ''
        if not typ.hasDefault:
            return ''
        v = typ.defaultValue
        if v is None or v == '':
            return ''
        return as_literal(v)

    def docstring_as_header(self, tid, enum_value=None):
        """Format a docstring for a section header, keeping line breaks.

        Args:
            tid: Type uid.
            enum_value: Enum member name, to format the docstring of the member.

        Returns:
            The formatted docstring.
        """

        text, label, dev_label = self.docstring_elements(tid, enum_value)
        lines = text.split('\n')
        lines[0] += label + dev_label
        return '\n\n'.join(lines)

    def docstring_as_cell(self, tid, enum_value=None):
        """Format a docstring for a table cell, on a single line.

        Args:
            tid: Type uid.
            enum_value: Enum member name, to format the docstring of the member.

        Returns:
            The formatted docstring.
        """

        text, label, dev_label = self.docstring_elements(tid, enum_value)
        return re.sub(r'\s+', ' ', text) + label + dev_label

    def docstring_elements(self, tid, enum_value=None):
        """Get the parts of a docstring in the reference language.

        Uses the translated string if present, otherwise the English one,
        marked as a missing translation. The default value is appended to the text.

        Args:
            tid: Type uid.
            enum_value: Enum member name, to get the docstring of the member.

        Returns:
            A list ``[text, label, dev_label]``.
        """

        # get the original (spec) docstring
        typ = self.gen.require_type(tid)
        en_text = typ.enumDocs.get(enum_value) if enum_value else typ.doc

        # try the translated (from strings) docstring
        key = tid
        if enum_value:
            key += '.' + enum_value
        local_text = self.gen.strings[self.lang].get(key)

        dev_label = ''

        if en_text and not local_text and self.lang != 'en':
            # translation missing: use the english docstring and warn
            base.log.debug(f'missing {self.lang} translation for {key!r}')
            dev_label = f'`??? {key}`{{.configref_dev_missing_translation}}'
            local_text = self.gen.strings['en'].get(key)
        else:
            dev_label = f'`{key}`{{.configref_dev_uid}}'

        local_text = local_text or en_text

        # process a label, like "foobar"
        # it might be missing in a translation, but present in the original (spec) docstring
        text, label = self.extract_label(local_text)
        if not label and en_text != local_text:
            _, label = self.extract_label(en_text)

        dflt = self.default_string(tid)
        if dflt:
            text += DEFAULT_FORMAT.format(self.strings['head_default'], dflt)

        return [text, label, dev_label]

    def extract_label(self, text):
        """Extract a version label like ``(added in 8.1)`` from the end of a docstring.

        Args:
            text: Docstring.

        Returns:
            A tuple ``(text without the label, formatted label)``; the label is empty if there is none.
        """

        m = re.match(rf'(.+?)\(({LABELS}) in (\d[\d.]+)\)$', text)
        if not m:
            return text, ''
        kind = m.group(2).strip()
        name = self.strings[f'label_{kind}']
        version = m.group(3)
        label = f'`{name}: {version}`{{.configref_label_{kind}}}'
        return m.group(1).strip(), label


def as_literal(s):
    """Format a value as a literal.

    Args:
        s: Value, formatted as JSON.

    Returns:
        Markdown text.
    """

    v = json.dumps(s, ensure_ascii=False)
    return f'`{v}`{{.configref_literal}}'


def as_typename(s):
    """Format a type name.

    Args:
        s: Type name.

    Returns:
        Markdown text.
    """

    return f'`{s}`{{.configref_typename}}'


def as_category(s):
    """Format a category name.

    Args:
        s: Category name.

    Returns:
        Markdown text.
    """

    return f'`{s}`{{.configref_category}}'


def as_propname(s):
    """Format the name of an optional property.

    Args:
        s: Property name.

    Returns:
        Markdown text.
    """

    return f'`{s}`{{.configref_propname}}'


def as_required(s):
    """Format the name of a required property.

    Args:
        s: Property name.

    Returns:
        Markdown text.
    """

    return f'`{s}`{{.configref_required}}'


def as_code(s):
    """Format text as inline code.

    Args:
        s: Text.

    Returns:
        Markdown text.
    """

    return f'`{s}`'


def header(cat, tid):
    """Format a section header.

    Args:
        cat: Category, used in the CSS class of the header.
        tid: Type uid, used as the header text and the anchor.

    Returns:
        Markdown text.
    """

    return f'\n## <span class="configref_category_{cat}"></span>{tid} :{tid}\n'


def subhead(category, text):
    """Format the text below a section header.

    Args:
        category: Category name (not used).
        text: Text.

    Returns:
        Markdown text.
    """

    # return as_category(category) + ' ' + text + '\n'
    return text + '\n'


def link(target, text):
    """Format a link to the section of a type.

    Args:
        target: Type uid.
        text: Link text.

    Returns:
        Markdown text.
    """

    return f'[{text}](../{target})'


def first_line(s):
    """Get the first line of a text.

    Args:
        s: Text or ``None``.

    Returns:
        The first line, stripped.
    """

    return (s or '').strip().split('\n')[0].strip()


def table(heads, rows):
    """Format a Markdown table with padded columns.

    Args:
        heads: Column headers.
        rows: Table rows, lists of cell values.

    Returns:
        Markdown text.
    """

    widths = [len(h) for h in heads]

    for r in rows:
        widths = [max(a, b) for a, b in zip(widths, [len(str(s)) for s in r])]

    def field(n, v):
        return str(v).ljust(widths[n])

    def row(r):
        return ' | '.join(field(n, v) for n, v in enumerate(r))

    out = [row(heads), '', *[row(r) for r in rows]]
    out[1] = '-' * len(out[0])
    return '\n'.join(f'| {s} |' for s in out) + '\n'


def escape(s, quote=True):
    """Escape HTML special characters.

    Args:
        s: Text.
        quote: If True, escape double quotes as well.

    Returns:
        The escaped text.
    """

    s = s.replace('&', '&amp;')
    s = s.replace('<', '&lt;')
    s = s.replace('>', '&gt;')
    if quote:
        s = s.replace('"', '&quot;')
    return s


nl = '\n'.join
