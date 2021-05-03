"""Generate reference pages from spec JSON"""

import re
import os
import json

DOC_ROOT = os.path.abspath(os.path.dirname(__file__))
APP_DIR = os.path.abspath(DOC_ROOT + '../../../app')

PRIMITIVES = "bool", "int", "str", "float", "float2", "float4", "dict",

with open(DOC_ROOT + '/words.json') as fp:
    WORDS = json.load(fp)


class ConfigRefGenerator:

    def __init__(self, lang, page, spec, root_type):
        self.book = 'server_admin'
        self.lang = lang
        self.page = page
        self.spec = spec
        self.root_type = root_type
        self.queue = [root_type]
        self.done = set(PRIMITIVES)
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

        elif t['type'] == 'any':
            pass

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

        self.obj_types[sname] = _nl(s)

    def label(self, tname):
        return '_'.join([self.lang, self.page, tname.replace('.', '_')])

    def ref(self, tname):
        if tname in PRIMITIVES:
            return _i(tname)

        m = re.match(r'(.+?)List$', tname)
        if m:
            base = m.group(1)
            return _b('[') + ' ' + self.ref(base) + ' ' + _b(']')

        self.queue.append(tname)

        return ':ref:' + _e(self.label(tname))

    def props(self, t):
        return _table(
            [self.w('prop_name'), self.w('prop_type'), '', self.w('prop_required'), self.w('prop_default')],
            [self.prop_row(t, p) for p in t['props']]
        )

    def prop_row(self, t, p):
        return _row(
            _ee(p['name']),
            self.ref(p['type']),
            p.get('doc', ''),
            self.w('no') if p['optional'] else self.w('yes'),
            _value(p['default'])
        )

    def w(self, s):
        return WORDS[self.lang][s]


class CliRefGenerator:
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
        return '_'.join([self.lang, self.page, fname.replace(' ', '_')])

    def w(self, s):
        return WORDS[self.lang][s]


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
    return _ee(json.dumps(s))


def _b(s):
    return '**%s**' % s


def _i(s):
    return '*%s*' % s


def _ee(s):
    return '``%s``' % s


def hh(s, u):
    return s + '\n' + (str(u) * len(s))


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


_title_rules = {
    r'^gws.types.(\w+)$': r'\1',
    r'^gws.common.(\w+).types.(\w+)$': r'\1.\2',
    r'^gws.common.(\w+).(\w+)$': r'\1.\2',
    r'^gws.types.ext.(\w+).(\w+)$': r'\1.\2',
    r'^gws.ext.action.(\w+).(\w+)$': r'action.\1.\2',
    r'^gws.(\w+).types.(\w+)$': r'\1.\2',
    r'^gws.ext.': r'',
    r'^gws.common.': r'',
    r'^gws.types.ext.': r'',
    r'^gws.': r'',
}


def _title(tname):
    # for k, v in _title_rules.items():
    #     if re.match(k, tname):
    #         return re.sub(k, v, tname)
    return tname


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
