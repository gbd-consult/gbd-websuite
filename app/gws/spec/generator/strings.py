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


def collect_ux(gen: base.Generator):
    """Collect structured UX documentation from per-module ``_doc/ux.ini`` files.

    Schlüsselformat: ``<full.uid>.<field>``. Das letzte Punkt-Segment wird als
    Feldname behandelt, der Rest als UID. Unbekannte Felder werden geloggt
    und ignoriert.

    Wird in ``gen.uxStrings`` als bereits gesammelte Marker (aus parser.py)
    eingemischt — ``ux.ini`` hat Vorrang vor Docstring-Markern.
    """
    ux = dict(gen.uxStrings) if gen.uxStrings else {}

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
