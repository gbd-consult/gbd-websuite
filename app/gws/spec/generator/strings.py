import re

from . import base, util


def collect(gen: base.Generator):
    dct = {}

    for spec in gen.specs:
        typ = gen.types[spec['uid']]
        if not typ.name:
            continue
        _add_string(dct, typ.name, typ.doc)
        if typ.enumDocs:
            for k, v in typ.enumDocs.items():
                _add_string(dct, typ.name + '.' + k, v)

    for path in util.find_files(gen.selfDir + '/..', pattern=r'/strings.+?\.ini$', deep=False):
        base.log.debug(f'parsing strings from {path!r}')
        util.parse_ini(dct, util.read_file(path))

    return dct


def _add_string(dct, uid, doc):
    """Add a docstring to the dict.

    The language is assumed en, unless the string starts with a [xx]
    language code.
    """

    lang = 'en'
    m = re.match(r'^\[(\w\w)](.+)$', doc)
    if m:
        lang = m.group(1)
        doc = m.group(2).strip()
    dct.setdefault(lang, {})[uid] = doc.replace('\\n', '\n')
