import re

from . import base, util

UX_FIELDS = {
    'label',
    'purpose',
    'whenToUse',
    'complexity',
    'useCases',
    'docsLink',
    'seeAlso',
    'example',
}
"""Allowed field names for ux.ini entries."""

DOCSTRING_MARKER_FIELDS = {
    'complexity',
    'seeAlso',
    'since',
    'deprecated',
}
"""Field-list-style markers recognised inside class docstrings."""

_MARKER_RE = re.compile(r'^[ \t]*:(\w+):[ \t]*(.+?)[ \t]*$', re.MULTILINE)


def collect(gen: base.Generator):
    strings_dct = {}

    for typ in gen.serverTypes:
        _add_string(strings_dct, 'en', typ.name, typ.doc)
        if typ.enumDocs:
            for k, v in typ.enumDocs.items():
                _add_string(strings_dct, 'en', typ.name + '.' + k, v)

    for path in util.find_files(gen.rootDir, pattern=r'/strings(\..+)?\.ini$', deep=True):
        base.log.debug(f'parsing strings from {path!r}')
        d = util.parse_ini(util.read_file(path))
        for lang, strs in d.items():
            for uid, text in strs.items():
                _add_string(strings_dct, lang, uid, text)

    return strings_dct


def collect_docstring_markers(gen: base.Generator):
    """Extract ``:field:``-style markers from class docstrings.

    Looks at every type's ``doc`` and pulls out fields like ``:complexity:
    intermediate`` or ``:seeAlso: gws.x.y``. Result is keyed by language
    ``'en'`` (docstrings are English by convention) and indexed by type
    ``name``.

    Returns a dict in the same shape as ``collect_ux`` — both can be merged.
    """
    ux = {}

    for typ in gen.typeDict.values():
        if not typ.doc or not typ.name:
            continue
        for m in _MARKER_RE.finditer(typ.doc):
            field = m.group(1)
            value = m.group(2).strip()
            if field not in DOCSTRING_MARKER_FIELDS:
                continue
            ux.setdefault('en', {}).setdefault(typ.name, {})[field] = value

    return ux


def collect_ux(gen: base.Generator):
    """Collect structured UX documentation from per-module ``_doc/ux.ini`` files.

    Schlüsselformat: ``<full.uid>.<field>``. Das letzte Punkt-Segment wird als
    Feldname behandelt, der Rest als UID. Unbekannte Felder werden geloggt
    und ignoriert.

    Wird mit ``gen.uxStrings`` (ggf. bereits durch
    :func:`collect_docstring_markers` befüllt) gemerged — ``ux.ini`` hat
    Vorrang vor Docstring-Markern.
    """
    ux = {}
    if gen.uxStrings:
        for lang, entries in gen.uxStrings.items():
            ux[lang] = {uid: dict(fields) for uid, fields in entries.items()}

    for path in util.find_files(gen.rootDir, pattern=r'/ux(\..+)?\.ini$', deep=True):
        base.log.debug(f'parsing ux strings from {path!r}')
        d = util.parse_ini(util.read_file(path))
        for lang, entries in d.items():
            for full_key, text in entries.items():
                uid, _, field = full_key.rpartition('.')
                if not uid or field not in UX_FIELDS:
                    base.log.warning(f'unknown ux field {full_key!r} in {path!r}')
                    continue
                ux.setdefault(lang, {}).setdefault(uid, {})[field] = (text or '').strip()

    return ux


def apply_ux_to_variants(gen: base.Generator):
    """Backfill VARIANT types' ``doc`` from ``uxStrings``.

    Synthesised VARIANT types start without a docstring (today's 0 % coverage
    issue). When a maintainer drops a ``purpose`` into ``_doc/ux.ini`` for the
    variant UID, fold it back into ``typ.doc`` so the regular ``strings.en``
    block picks it up.
    """
    if not gen.uxStrings:
        return

    en = gen.uxStrings.get('en') or {}

    for typ in gen.typeDict.values():
        if typ.c != base.c.VARIANT or typ.doc:
            continue
        entry = en.get(typ.name) or en.get(typ.uid)
        if not entry:
            continue
        purpose = entry.get('purpose') or entry.get('label')
        if purpose:
            typ.doc = purpose


def _add_string(strings_dct, lang, uid, text):
    """Add a docstring to the dict.

    If the string starts with a [xx], override the given language.
    """

    text = (text or '').strip()
    m = re.match(r'^\[(\w\w)](.+)$', text)
    if m:
        lang = m.group(1)
        text = m.group(2).strip()
    strings_dct.setdefault(lang, {})[uid] = text.replace('\\n', '\n')
