"""Bootstrap a ``_doc/ux.ini`` skeleton for a plugin or base module.

Reads the latest ``app/__build/specs.json``, picks all CLASS types whose
``modName`` lives inside the given source directory (typically a plugin
under ``app/gws/plugin/<name>`` or a base module under ``app/gws/base/...``)
and writes a starter ``ux.ini`` with ``label`` / ``purpose`` /
``complexity`` suggestions per class and per property.

The output is intentionally a *skeleton*: labels are derived from the
identifier with simple heuristics, purpose strings come straight from the
existing docstring's first sentence (or are left blank), complexity is
guessed from a short keyword list. Maintainers are expected to polish it.

Usage::

    python -m gws.spec.generator.bootstrap_ux <plugin-dir> [--apply] [--lang de,en]

Without ``--apply`` the proposal is printed to stdout. With ``--apply``
it is written to ``<plugin-dir>/_doc/ux.ini`` — only if that file does
not yet exist (the bootstrap never overwrites curated content).
"""

import os
import re
import sys
from typing import Optional

from . import util


USAGE = """
GWS UX bootstrap
~~~~~~~~~~~~~~~~

    python -m gws.spec.generator.bootstrap_ux <plugin-dir> [options]

Options:

    --lang <list>        - Comma-separated language codes (default: de,en)
    --apply              - Write to <plugin-dir>/_doc/ux.ini
                           (only if the file does not yet exist)
    --specs <path>       - Path to specs.json
                           (default: app/__build/specs.json)
    -h, --help           - Print this help and exit
"""


DOMAIN_TERMS_DE = {
    'dn': 'DN',
    'url': 'URL',
    'uri': 'URI',
    'tcp': 'TCP',
    'ip': 'IP',
    'dpi': 'DPI',
    'ogc': 'OGC',
    'wms': 'WMS',
    'wmts': 'WMTS',
    'wfs': 'WFS',
    'csw': 'CSW',
    'sql': 'SQL',
    'crs': 'CRS',
    'epsg': 'EPSG',
    'json': 'JSON',
    'xml': 'XML',
    'csv': 'CSV',
    'tc': 'TC',
    'mfa': 'MFA',
    'totp': 'TOTP',
    'ldap': 'LDAP',
    'pdf': 'PDF',
    'png': 'PNG',
    'tiff': 'TIFF',
    'postgis': 'PostGIS',
    'qgis': 'QGIS',
    'alkis': 'ALKIS',
    'osm': 'OSM',
    'cors': 'CORS',
    'http': 'HTTP',
    'https': 'HTTPS',
    'api': 'API',
    'id': 'ID',
}

VERB_PREFIXES = {
    'use': 'verwenden',
    'is': 'ist',
    'has': 'hat',
    'allow': 'erlaubt',
    'enable': 'aktivieren',
    'disable': 'deaktivieren',
    'show': 'anzeigen',
    'hide': 'ausblenden',
}


_CAMEL_RE = re.compile(r'[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+')


def _split_camel(ident: str) -> list[str]:
    return _CAMEL_RE.findall(ident)


def _term(part: str, lang: str) -> str:
    low = part.lower()
    if low in DOMAIN_TERMS_DE:
        return DOMAIN_TERMS_DE[low]
    return part[0].upper() + part[1:].lower() if part else part


def _label_from_ident(ident: str, lang: str) -> str:
    """Derive a human-readable label from a camelCase identifier.

    The heuristic is intentionally simple: split on camel boundaries,
    map a small list of known acronyms to canonical spellings, and if the
    first token is a verb prefix (``use``, ``is``, ``has`` …) move it to
    the end with the German equivalent verb.
    """
    parts = _split_camel(ident)
    if not parts:
        return ident

    first_lower = parts[0].lower()
    if first_lower in VERB_PREFIXES and len(parts) > 1:
        rest = '-'.join(_term(p, lang) for p in parts[1:])
        return f'{rest} {VERB_PREFIXES[first_lower]}'

    return '-'.join(_term(p, lang) for p in parts)


_BASIC_PROPS = {'host', 'port', 'url', 'username', 'user', 'password', 'database', 'name', 'title', 'uid'}
_ADVANCED_KEYWORDS = ('cache', 'internal', 'debug', 'timeout', 'concurrency', 'worker', 'thread', 'pool')


def _guess_complexity(prop_ident: str) -> str:
    low = prop_ident.lower()
    if low in _BASIC_PROPS:
        return 'basic'
    if any(k in low for k in _ADVANCED_KEYWORDS):
        return 'advanced'
    return ''


_SENTENCE_RE = re.compile(r'(.+?[.!?])(?:\s|$)', re.DOTALL)


def _first_sentence(doc: str) -> str:
    """Return the first sentence of ``doc`` (or the whole string)."""
    if not doc:
        return ''
    text = ' '.join(doc.split())
    m = _SENTENCE_RE.match(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _strip_lang_prefix(text: str) -> str:
    """Drop a leading ``[xx]`` language marker, if any."""
    m = re.match(r'^\[\w\w\]\s*(.*)', text or '')
    return m.group(1).strip() if m else (text or '').strip()


def _module_prefix_from_dir(plugin_dir: str) -> str:
    """Translate a source directory like ``app/gws/plugin/postgres`` to ``gws.plugin.postgres``.

    The path may be absolute or relative. We anchor on the first ``gws``
    segment we can find (under either ``app/`` or directly) and join the
    remaining parts with dots.
    """
    p = os.path.abspath(plugin_dir)
    parts = p.split(os.sep)
    if 'gws' not in parts:
        raise ValueError(f'expected a path under .../gws/, got {plugin_dir!r}')
    idx = parts.index('gws')
    return '.'.join(parts[idx:])


def _is_in_module(typ_modname: str, prefix: str) -> bool:
    if not typ_modname:
        return False
    return typ_modname == prefix or typ_modname.startswith(prefix + '.')


def collect_classes(specs: dict, prefix: str) -> list[dict]:
    """Return the CLASS types under ``prefix``, sorted by name."""
    out = []
    for t in specs.get('serverTypes', []):
        if t.get('c') != 'CLASS':
            continue
        if not _is_in_module(t.get('modName', ''), prefix):
            continue
        out.append(t)
    out.sort(key=lambda t: t.get('name', ''))
    return out


def collect_properties_for(specs: dict, class_name: str, prefix: str) -> list[dict]:
    """Return PROPERTY types belonging to ``class_name`` and originating in ``prefix``.

    Inherited properties (whose ``name`` does not start with the class name)
    are skipped — they belong to the parent class and should be documented
    where they are defined.
    """
    out = []
    for t in specs.get('serverTypes', []):
        if t.get('c') != 'PROPERTY':
            continue
        if t.get('tOwner') != class_name:
            continue
        if not (t.get('name') or '').startswith(class_name + '.'):
            continue
        if not _is_in_module(t.get('modName', ''), prefix):
            continue
        out.append(t)
    out.sort(key=lambda t: t.get('name', ''))
    return out


def _format_block(class_typ: dict, props: list[dict], lang: str) -> list[str]:
    """Build the INI lines for one class block."""
    lines: list[str] = []
    cname = class_typ.get('name', '')
    short = cname.rsplit('.', 1)[-1]
    parent = cname.rsplit('.', 1)[0]
    short_module = parent.rsplit('.', 1)[-1] if '.' in parent else parent

    label_seed = _label_from_ident(short_module, lang) if short in ('Config', 'Object', 'Props') else _label_from_ident(short, lang)

    purpose = _strip_lang_prefix(_first_sentence(class_typ.get('doc', '')))

    lines.append(f'# --- {short_module}.{short} {"-" * max(0, 60 - len(short_module) - len(short))}')
    lines.append(f'{cname}.label = {label_seed}')
    if purpose:
        lines.append(f'{cname}.purpose = {purpose}')
    else:
        lines.append(f'{cname}.purpose =')
    lines.append('')

    for p in props:
        pname = p.get('name', '')
        pident = pname.rsplit('.', 1)[-1]
        plabel = _label_from_ident(pident, lang)
        ppurpose = _strip_lang_prefix(_first_sentence(p.get('doc', '')))
        complexity = _guess_complexity(pident)

        lines.append(f'{pname}.label = {plabel}')
        if ppurpose:
            lines.append(f'{pname}.purpose = {ppurpose}')
        if complexity:
            lines.append(f'{pname}.complexity = {complexity}')
        lines.append('')

    return lines


def render_skeleton(specs: dict, prefix: str, langs: list[str]) -> str:
    """Build the complete INI text (header + per-language sections)."""
    classes = collect_classes(specs, prefix)

    out = [
        '# Auto-generated skeleton from bootstrap_ux.py.',
        '# Bitte Texte überprüfen, anpassen, deutsche purpose-Texte ergänzen.',
        '#',
        f'# Modul-Prefix: {prefix}',
        f'# Anzahl Klassen: {len(classes)}',
        '',
    ]

    if not classes:
        out.append(f'# (No CLASS types found under {prefix}. Make sure ./make.sh spec is up to date.)')
        return '\n'.join(out) + '\n'

    for lang in langs:
        out.append(f'[{lang}]')
        out.append('')
        for cls in classes:
            cname = cls.get('name', '')
            props = collect_properties_for(specs, cname, prefix)
            out.extend(_format_block(cls, props, lang))
        out.append('')

    return '\n'.join(out).rstrip() + '\n'


def bootstrap_plugin(plugin_dir: str, langs: list[str], apply: bool, specs_path: str) -> tuple[str, Optional[str]]:
    """Generate (and optionally write) a ``_doc/ux.ini`` skeleton.

    Returns ``(rendered_text, written_path_or_None)``.
    """
    prefix = _module_prefix_from_dir(plugin_dir)
    specs = util.read_json(specs_path)
    text = render_skeleton(specs, prefix, langs)

    if not apply:
        return text, None

    target_dir = os.path.join(plugin_dir, '_doc')
    target = os.path.join(target_dir, 'ux.ini')
    if os.path.exists(target):
        return text, None

    os.makedirs(target_dir, exist_ok=True)
    util.write_file(target, text)
    return text, target


def _parse_args(argv: list[str]) -> dict:
    args: dict = {
        'plugin_dir': '',
        'langs': ['de', 'en'],
        'apply': False,
        'specs': 'app/__build/specs.json',
    }
    it = iter(argv)
    for a in it:
        if a in ('-h', '--help'):
            args['help'] = True
        elif a == '--apply':
            args['apply'] = True
        elif a == '--lang':
            args['langs'] = [s.strip() for s in next(it).split(',') if s.strip()]
        elif a == '--specs':
            args['specs'] = next(it)
        elif a.startswith('-'):
            raise SystemExit(f'unknown argument {a!r}')
        else:
            if args['plugin_dir']:
                raise SystemExit(f'unexpected positional argument {a!r}')
            args['plugin_dir'] = a
    return args


def run(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(list(argv if argv is not None else sys.argv[1:]))
    if args.get('help'):
        sys.stdout.write(USAGE)
        return 0
    if not args['plugin_dir']:
        sys.stdout.write(USAGE)
        return 2

    text, written = bootstrap_plugin(
        args['plugin_dir'],
        args['langs'],
        args['apply'],
        args['specs'],
    )

    if args['apply']:
        if written:
            sys.stdout.write(f'wrote {written}\n')
        else:
            target = os.path.join(args['plugin_dir'], '_doc', 'ux.ini')
            sys.stdout.write(f'skip: {target} already exists (refusing to overwrite)\n')
            sys.stdout.write('--- proposal (not written) ---\n')
            sys.stdout.write(text)
        return 0

    sys.stdout.write(text)
    return 0


if __name__ == '__main__':
    sys.exit(run())
