import re

from . import base


def generate(state, server_specs, text):
    ini = _parse_ini(text)

    type_names = set()

    for type_name in server_specs:
        t = state.types[type_name]
        if not isinstance(t, base.TNamedType):
            continue
        type_names.add(type_name)
        if not t.doc:
            continue
        lang = 'en'
        doc = t.doc
        m = re.match(r'^\[(\w\w)\](.+)$', doc)
        if m:
            lang = m.group(1)
            doc = m.group(2).strip()
        ini.setdefault(lang, {})[type_name] = _decode(doc)

    # for lang, entries in ini.items():
    #     es = set(entries)
    #     for e in sorted(type_names.difference(es)):
    #         base.log.debug(f'strings: [{lang}] MISSING: {e}')
    #     for e in sorted(es.difference(type_names)):
    #         base.log.debug(f'strings: [{lang}] INVALID: {e}')

    return {
        k: dict(sorted(v.items()))
        for k, v in sorted(ini.items())
    }


# configparser? no, thanks ;)

def _parse_ini(text):
    ini = {}
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
        k, v = ln.split('=', maxsplit=1)
        ini.setdefault(section, {})[k.strip()] = v.strip()

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
