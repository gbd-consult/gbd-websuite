import os
import re
import json
import sys
import shutil
import bs4


def _read(fname, encoding='utf8'):
    try:
        with open(fname, encoding=encoding) as fp:
            return fp.read()
    except IOError:
        return ''


_nl = '\n'.join

DOC_ROOT = os.path.abspath(os.path.dirname(__file__))
GEN_ROOT = DOC_ROOT + '/ref'

APP_DIR = os.path.abspath(DOC_ROOT + '../../../app')
sys.path.insert(0, APP_DIR)

VERSION = _read(DOC_ROOT + '/../../VERSION').strip()

sys.path.insert(0, os.path.dirname(__file__))
import refgen

with open(DOC_ROOT + '/words.json') as fp:
    WORDS = json.load(fp)

HELP_BASE_URL = 'https://gws.gbd-consult.de/doc/{release}/books/client-user/{lang}/overview'


##


def clear_output():
    shutil.rmtree(DOC_ROOT + '/../_build', ignore_errors=True)
    shutil.rmtree(GEN_ROOT, ignore_errors=True)


def cleanup_rst():
    for p in _find_files(DOC_ROOT + '/books', 'rst$'):
        out = []
        for ln in _read(p).strip().splitlines():
            if not ln.strip():
                if out[-1]:
                    out.append('')
                else:
                    continue
            elif re.match(r'^---|^===|^~~~', ln):
                out.append(ln[0] * len(out[-1]))
            else:
                out.append(ln.rstrip())

        _write_if_changed(p, _nl(out) + '\n')


def format_special(txt, book, lang):
    def _table(m):
        head = m.group(1).strip()
        code = [
            '.. csv-table::',
            '   :delim: |',
            '   :widths: auto',
            '   :align: left',
            ''
        ]
        delim = '|'

        for ln in m.group(2).strip().split('\n'):
            code.append('   ' + delim.join(s.strip() for s in ln.split(delim)))

        return '\n'.join(code)

    def _ref(m):
        return ".. admonition:: %s\n\n   :ref:`%s_configref_%s`" % (
            WORDS[lang]['reference'], lang, m.group(1).strip().replace('.', '_'))

    # some RST shortcuts:

    # ^filename => :doc:`filename`

    txt = re.sub(r'\^([a-z_/]+)', r':doc:`\1`', txt)

    # ^SEE => ..seealso:

    txt = re.sub(r'\^SEE', r'.. seealso::', txt)

    # ^NOTE => ..note:

    txt = re.sub(r'\^NOTE', r'.. note::', txt)

    # ^REF class => config reference link

    txt = re.sub(r'\^REF(.+)', _ref, txt)

    # {TABLE}...{/TABLE} => ..csvtable

    txt = re.sub(r'''(?sx)
        {TABLE (.*?)}
            (.+?)
        {/TABLE}
    ''', _table, txt)

    return txt


def make_config_ref(lang):
    page = 'configref'
    root_type = 'gws.common.application.Config'
    gen = refgen.ConfigRefGenerator(lang, page, _load_spec(lang, APP_DIR), root_type)
    text = gen.run()
    out = GEN_ROOT + '/' + lang + '.' + page + '.txt'
    _write_if_changed(out, text)


def make_cli_ref(lang):
    page = 'cliref'
    gen = refgen.CliRefGenerator(lang, page, _load_spec(lang, APP_DIR))
    text = gen.run()
    out = GEN_ROOT + '/' + lang + '.' + page + '.txt'
    _write_if_changed(out, text)


def make_help(lang):
    release = '.'.join(VERSION.split('.')[:-1])
    base = HELP_BASE_URL.format(lang=lang, release=release)
    html = _read(DOC_ROOT + f'/../_build/books/client-user/{lang}/overview/help.html')

    bs = bs4.BeautifulSoup(html, 'html.parser')

    for n in bs.find_all('link'):
        if n.get('href'):
            n['href'] = _abslink(n['href'], base)
    for n in bs.find_all('a'):
        if n.get('href'):
            n['href'] = _abslink(n['href'], base)
            n['target'] = '_blank'
    for n in bs.find_all('img'):
        if n.get('src'):
            n['src'] = _abslink(n['src'], base)

    for n in bs.find_all(['script', 'nav', 'footer']):
        n.extract()
    for n in bs.find_all(role='navigation'):
        n.extract()

    html = bs.prettify()
    html += """
        <style>
            .wy-nav-content-wrap {
                margin-left: 0 !important;
                background: none !important;
            }
            .wy-body-for-nav {
                background: none !important;
            }
        </style>
    """
    _write_if_changed(DOC_ROOT + f'/../_build/help_{lang}.html', html)


_EXCLUDE = ['vendor', '/t/', '.in.', '.wsgi']


def make_autodoc():
    mods = set()

    for p in _find_files(APP_DIR + '/gws', 'py$'):
        if any(s in p for s in _EXCLUDE):
            continue
        p = re.sub(r'.*?gws/', 'gws/', p)
        p = re.sub(r'__init__.py$', '', p)
        p = re.sub(r'.py$', '', p)
        p = re.sub(r'/$', '', p)
        mods.add(p.replace('/', '.'))

    for mod in sorted(mods):
        txt = [
            mod,
            '=' * len(mod),
            '',
            '.. automodule:: ' + mod,
            '    :members:',
            '    :undoc-members:',
            '',
        ]
        _write_if_changed(GEN_ROOT + '/server/' + mod + '.rst', _nl(txt))

    txt = [
        'Modules',
        '=======',
        '',
        '.. toctree::',
        '    :maxdepth: 1',
        '',
    ]

    for mod in sorted(mods):
        txt.append('    ' + mod)

    _write_if_changed(GEN_ROOT + '/server/index.rst', _nl(txt))


##

def _load_spec(lang, app_dir):
    spec_path = app_dir + '/spec/gen/' + ('' if lang == 'en' else lang + '.') + 'spec.json'

    with open(spec_path) as fp:
        return json.load(fp)


def _write_if_changed(fname, text):
    curr = _read(fname)
    if text != curr:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'wt') as fp:
            fp.write(text)


def _find_files(dirname, pattern):
    for fname in os.listdir(dirname):
        if fname.startswith('.'):
            continue

        path = os.path.join(dirname, fname)

        if os.path.isdir(path):
            yield from _find_files(path, pattern)
            continue

        if re.search(pattern, fname):
            yield path


def _abslink(href, base):
    if href.startswith(('http', '#')):
        return href
    if href.startswith('/'):
        return base + href
    return base + '/' + href
