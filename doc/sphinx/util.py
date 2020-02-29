import os
import re
import json
import sys
import shutil
import bs4

DOC_ROOT = os.path.abspath(os.path.dirname(__file__))
GEN_ROOT = DOC_ROOT + '/ref'

APP_DIR = os.path.abspath(DOC_ROOT + '../../../app')

VERSION = open(DOC_ROOT + '/../../VERSION').read().strip()

sys.path.insert(0, APP_DIR)

with open(os.path.dirname(__file__) + '/words.json') as fp:
    WORDS = json.load(fp)


##


def clear():
    shutil.rmtree(DOC_ROOT + '/../_build', ignore_errors=True)
    shutil.rmtree(GEN_ROOT, ignore_errors=True)


def make_config_ref(lang):
    page = 'configref'
    root_type = 'gws.common.application.Config'
    gen = _ConfigRefGenerator(lang, page, _load_spec(lang, APP_DIR), root_type)
    text = gen.run()
    out = GEN_ROOT + '/' + lang + '.' + page + '.txt'
    _write_if_changed(out, text)


def make_cli_ref(lang):
    page = 'cliref'
    gen = _CliRefGenerator(lang, page, _load_spec(lang, APP_DIR))
    text = gen.run()
    out = GEN_ROOT + '/' + lang + '.' + page + '.txt'
    _write_if_changed(out, text)


_HELP_BASE_URL = 'https://gws.gbd-consult.de/doc/{release}/books/client-user/{lang}/overview'


def make_help(lang):
    release = '.'.join(VERSION.split('.')[:-1])
    base = _HELP_BASE_URL.format(lang=lang, release=release)

    with open(DOC_ROOT + f'/../_build/books/client-user/{lang}/overview/help.html') as fp:
        html = fp.read()

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

    with open(DOC_ROOT + f'/../_build/help_{lang}.html', 'w') as fp:
        fp.write(html)


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

_PRIMITIVES = "bool", "int", "str", "float", "float2", "float4", "dict",


class _ConfigRefGenerator:

    def __init__(self, lang, page, spec, root_type):
        self.book = 'server_admin'
        self.lang = lang
        self.page = page
        self.spec = spec
        self.root_type = root_type
        self.queue = [root_type]
        self.done = set(_PRIMITIVES)
        self.obj_types = {}

    def run(self):
        while self.queue:
            tname = self.queue.pop(0)
            if tname not in self.done:
                self.format_type(self.spec[tname])
                self.done.add(tname)

        return _nl([
            '',
            _nl(s for s in _sorted_values(self.obj_types))
        ])

    def format_type(self, t):
        if t['type'] == 'object':
            self.obj_type(t, self.w('properties') + ':', self.props(t))

        elif t['type'] == 'taggedunion':
            self.obj_type(
                t,
                self.w('one_of') + ':',
                _list(self.ref(b) for b in t['parts'].values())
            )

        elif t['type'] == 'enum':
            self.obj_type(
                t,
                self.w('one_of') + ':',
                _list(_value(v) for v in t['values'])
            )

        elif t['type'] == 'tuple':
            self.obj_type(t)

        elif t['type'] == 'alias':
            self.obj_type(t)

        else:
            raise ValueError('unhandled type %s' % t['type'])

    def obj_type(self, t, text=None, table=None):
        tname = t['name']
        sname = _title(tname)
        s = [
            '',
            '.. _%s:' % self.label(tname),
            '',
            _h2(sname),
            '',
            t.get('doc', ''),
            ''
        ]
        if text:
            s.extend([_i(text), ''])
        if table:
            s.append(table)

        sort_key = 10
        if tname == self.root_type:
            sort_key = 0
        if t['type'] in ('union', 'enum'):
            sort_key = 20

        self.obj_types[(sort_key, sname)] = _nl(s)

    def label(self, tname):
        return '_'.join([self.book, self.lang, self.page, tname.replace('.', '_')])

    def ref(self, tname):
        if tname in _PRIMITIVES:
            return _ee(tname)

        m = re.match(r'(.+?)List$', tname)
        if m:
            return '[%s...]' % self.ref(m.group(1))

        self.queue.append(tname)

        return ':ref:' + _e(self.label(tname))

    def props(self, t):
        return _table(
            [self.w('prop_name'), self.w('prop_type'), '', self.w('prop_required'), self.w('prop_default')],
            [self.prop_row(t, p) for p in t['props']]
        )

    def prop_row(self, t, p):
        return _row(
            _i(p['name']),
            self.ref(p['type']),
            p.get('doc', ''),
            self.w('no') if p['optional'] else self.w('yes'),
            _value(p['default'])
        )

    def w(self, s):
        return WORDS[self.lang][s]


class _CliRefGenerator:
    def __init__(self, lang, page, spec):
        self.book = 'server_admin'
        self.lang = lang
        self.page = page
        self.spec = spec

    def run(self):
        out = {}

        for k, c in self.spec.items():
            if not k.startswith('cli:'):
                continue

            fname = c['command'] + ' ' + c['subcommand'].replace('_', '-')
            t = [
                '',
                '.. _%s:' % self.label(fname),
                '',
                _h3(fname),
                '',
                c.get('doc', '')
            ]
            if c.get('args'):
                t.extend([
                    '',
                    _i(self.w('options') + ':'),
                    '',
                    self.args(c['args'])
                ])
            out[fname] = _nl(t)

        return _nl(_sorted_values(out))

    def args(self, args):
        rows = []

        for a in sorted(args, key=lambda a: a['name']):
            rows.append(_row(
                _ee(a['name']),
                a['doc'].replace('\n', ' ')
            ))

        return _table(None, rows)

    def label(self, fname):
        return '_'.join([self.book, self.lang, self.page, fname.replace(' ', '_')])

    def w(self, s):
        return WORDS[self.lang][s]


##

def _load_spec(lang, app_dir):
    spec_path = app_dir + '/spec/gen/' + ('' if lang == 'en' else lang + '.') + 'spec.json'

    with open(spec_path) as fp:
        return json.load(fp)


def _write_if_changed(fname, text):
    curr = ''
    try:
        with open(fname) as fp:
            curr = fp.read()
    except IOError:
        pass

    if text != curr:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, 'wt') as fp:
            fp.write(text)


_pipe = ' | '.join
_comma = ', '.join
_nl = '\n'.join
_nl2 = '\n\n'.join
___ = '   '


def _q(s):
    return '"%s"' % s


def _e(s):
    return '`%s`' % s


def _value(s):
    if s is None or s == {} or s == '':
        return ''
    return json.dumps(s)


def _b(s):
    return '**%s**' % s


def _i(s):
    return '*%s*' % s


def _ee(s):
    return '``%s``' % s


def hh(s, u):
    return s + '\n' + (str(u) * len(s)) + '\n'


def _h1(s):
    return hh(s, '=')


def _h2(s):
    return hh(s, '-')


def _h3(s):
    return hh(s, '~')


def _h4(s):
    return hh(s, '^')


def _list(ls):
    return _nl(sorted('* ' + x for x in ls))


_exclude_mods = 'gws', 'types', 'ext', 'common', 'gis'


def _title(tname):
    return tname
    # @TODO attempt to make type names "friendlier"...
    # ls = tname.split('.')
    # ls = [s[0].upper() + s[1:] for s in ls if s not in _exclude_mods]
    # return ''.join(ls)


def _row(*values):
    return ___ + _comma('~%s~' % v for v in values)


def _table(headers, rows):
    if not rows:
        return ''
    thead = [
        '.. csv-table::',
        ___ + ':quote: ~',
        ___ + ':widths: auto',
        ___ + ':align: left',
    ]
    if headers:
        thead.append(___ + ':header: ' + _comma(headers))
    return _nl(thead + [''] + sorted(rows))


def _sorted_values(d):
    return [v for _, v in sorted(d.items())]


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
