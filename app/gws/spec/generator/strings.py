import re

from . import base, util


def generate(gen: base.Generator):
    ini = {}

    for uid, spec in gen.specs.items():
        if spec['c'] not in {base.C.CONFIG, base.C.COMMAND, base.C.CLASS, base.C.PROPERTY}:
            continue
        typ = gen.types[uid]
        _add_string(ini, typ.name, typ.doc)
        if typ.enumDocs:
            for k, v in typ.enumDocs.items():
                _add_string(ini, typ.name + '.' + k, v)

    for path in util.find_files(gen.rootDir + '/gws/spec', pattern=r'/strings.+?\.ini$', deep=False):
        base.log.debug(f'parsing strings from {path!r}')
        _parse_ini(ini, util.read_file(path))

    return _make_ini(ini)


def _add_string(ini, uid, doc):
    lang = 'en'
    m = re.match(r'^\[(\w\w)](.+)$', doc)
    if m:
        lang = m.group(1)
        doc = m.group(2).strip()
    ini.setdefault(lang, {})[uid] = _decode(doc)


def _parse_ini(ini, text):
    section = 'DEFAULT'

    for ln in text.strip().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith((';', '#', '//')):
            continue
        if ln[0] == '[':
            section = ln[1:-1].strip()
            continue
        if '=' not in ln:
            raise ValueError(f'invalid ini string {ln!r}')
        key, _, val = ln.partition('=')
        ini.setdefault(section, {})[key.strip()] = _decode(val.strip())

    return ini


def _make_ini(ini):
    buf = []

    for sec, rows in ini.items():
        buf.append('[' + sec + ']')
        for k, v in sorted(rows.items()):
            buf.append(k + '=' + _encode(v))
        buf.append('')

    return '\n'.join(buf)


def _decode(s):
    s = s.replace('\\n', '\n')
    return s


def _encode(s):
    s = s.replace('\n', '\\n')
    s = s.replace('\t', ' ')
    return s
