import re
import json

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


_exclude_mods = 'gws', 'types', 'ext', 'common', 'gis'


def _spec_type(tname):
    # a "spec" type like "filepath"?
    m = re.search(r'\.([a-z]\w*)$', tname)
    return m.group(1) if m else None


def _title(tname):
    ls = tname.split('.')
    if _spec_type(tname):
        return ls[-1]
    ls = [s[0].upper() + s[1:] for s in ls if s not in _exclude_mods]
    return ''.join(ls)


def _row(*values):
    return ___ + _comma('~%s~' % v for v in values)


def _table(headers, rows):
    thead = [
        '.. csv-table::',
        ___ + ':quote: ~',
        ___ + ':widths: auto',
    ]
    if headers:
        thead.append(___ + ':header: ' + _comma(headers))
    return _nl(thead + [''] + sorted(rows))


def _sorted_values(d):
    return [v for _, v in sorted(d.items())]


def _write_if_changed(fname, text):
    curr = ''
    try:
        with open(fname) as fp:
            curr = fp.read()
    except IOError:
        pass

    if text != curr:
        with open(fname, 'wt') as fp:
            fp.write(text)


primitives = "bool", "int", "str", "float", "float2", "float4", "dict",

words = {
    'en': {
        'spec_types': 'Special types',
        'obj_types': 'Object types',
        'list_of': 'list of %s',
        'properties': 'Properties',
        'one_of': 'One of',
        'prop_name': 'name',
        'prop_type': 'type',
        'prop_required': 'required',
        'prop_default': 'default',
        'options': 'Options',
    },

}


class _ConfigRefGenerator:

    def __init__(self, lang, page, types, root_type):
        self.book = 'server_admin'
        self.lang = lang
        self.page = page
        self.types = types
        self.root_type = root_type
        self.queue = [root_type]
        self.done = set(primitives)
        self.spec_types = []
        self.obj_types = {}

    def run(self):
        while self.queue:
            tname = self.queue.pop(0)
            if tname not in self.done:
                self.format_type(self.types[tname])
                self.done.add(tname)

        return _nl([
            '',
            _table(None, self.spec_types),
            '',
            _nl(s for s in _sorted_values(self.obj_types))
        ])

    def format_type(self, t):
        if t['type'] == 'object':
            st = _spec_type(t['name'])
            if st:
                self.spec_types.append(_row(_ee(st), t.get('doc', '')))
            else:
                self.obj_type(t, self.w('properties') + ':', self.props(t))

        elif t['type'] == 'union':
            self.obj_type(
                t,
                self.w('one_of') + ':',
                _nl(sorted('* ' + self.ref(b) for b in t['bases'])))

        elif t['type'] == 'enum':
            self.obj_type(
                t,
                self.w('one_of') + ':',
                _nl(sorted('* ' + _value(v) for v in t['values'])))

        else:
            raise ValueError('unhandled type %s' % t['type'])

    def obj_type(self, t, text, table=None):
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
        if tname in primitives:
            return _ee(tname)

        m = re.match(r'(.+?)List$', tname)
        if m:
            return '[%s...]' % self.ref(m.group(1))

        self.queue.append(tname)
        if _spec_type(tname):
            return _ee(_title(tname))

        return ':ref:' + _e(self.label(tname))

    def prop_row(self, t, p):
        lit = None

        if p['name'] == 'type':
            m = re.search(r'(\w+)\.Config$', t['name'])
            if m:
                lit = _q(m.group(1))

        return _row(
            _i(p['name']),
            self.ref(p['type']),
            lit or p.get('doc', ''),
            'no' if p['optional'] else 'yes',
            _value(p['default'])
        )

    def props(self, t):
        return _table(
            [self.w('prop_name'), self.w('prop_type'), '', self.w('prop_required'), self.w('prop_default')],
            [self.prop_row(t, p) for p in t['props']]
        )

    def w(self, s):
        return words[self.lang][s]


class _CliRefGenerator:
    def __init__(self, lang, page, funcs):
        self.book = 'server_admin'
        self.lang = lang
        self.page = page
        self.funcs = funcs

    def run(self):
        out = {}
        for f in self.funcs:
            fname = f['command'] + ' ' + f['name'].replace('_', '-')
            t = [
                '',
                '.. _%s:' % self.label(fname),
                '',
                _h3(fname),
                '',
                f.get('doc', '')
            ]
            if f['args']:
                t.extend([
                    '',
                    _i(self.w('options') + ':'),
                    '',
                    self.args(f)
                ])
            out[fname] = _nl(t)

        return _nl(_sorted_values(out))

    def args(self, f):
        rows = []
        for a in sorted(f['args'], key=lambda a: a['name']):
            rows.append(_row(
                _ee(a['name']),
                a['doc'].replace('\n', ' ')
            ))
        return _table(None, rows)

    def label(self, fname):
        return '_'.join([self.book, self.lang, self.page, fname.replace(' ', '_')])

    def w(self, s):
        return words[self.lang][s]


def make_config_ref(lang, server_root, doc_root):
    spec = server_root + '/app/spec/en/config.spec.json'
    page = 'configref'
    root_type = 'gws.common.application.Config'

    with open(spec) as fp:
        types = json.load(fp)['types']

    gen = _ConfigRefGenerator(lang, page, types, root_type)
    text = gen.run()

    out = doc_root + '/books/server-admin/' + lang + '/' + page + '.txt'
    _write_if_changed(out, text)


def make_cli_ref(lang, server_root, doc_root):
    spec = server_root + '/app/spec/en/cli.spec.json'
    page = 'cliref'

    with open(spec) as fp:
        funcs = json.load(fp)

    gen = _CliRefGenerator(lang, page, funcs)
    text = gen.run()

    out = doc_root + '/books/server-admin/' + lang + '/' + page + '.txt'
    _write_if_changed(out, text)
