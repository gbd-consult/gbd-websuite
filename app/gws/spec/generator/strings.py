import re

from . import base, util


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
