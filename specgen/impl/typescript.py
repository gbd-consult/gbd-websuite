"""Generate typescript API files from the server spec"""

import json
import re

_decls_template = """
/**
 * Gws Server API.
 * Version @@VERSION@@
 */

type _int = number;
type _float = number;
type _bytes = any;
type _dict = {[k: string]: any};

_decls_
"""

_stub_template = """
/**
 * Gws Server Base Implementation
 * Version @@VERSION@@
 */

import * as gws from './gws-server.api';

export abstract class GwsServer implements gws.GwsServerApi {
\tabstract async _call(cmd, p, options): Promise<any>;

_methods_
} 
"""

_pipe = ' | '.join
_comma = ', '.join
_nl = '\n'.join
_nl2 = '\n\n'.join
_indent = '    '


def _format(template, **kwargs):
    s = re.sub(
        r'_(\w+)_',
        lambda m: kwargs[m.group(1)],
        template).strip()
    return s.replace('\t', _indent)


def _val(s):
    return json.dumps(s)


def _short(s):
    if '.' not in s:
        return s
    s = s.split('.')
    if s[-2] == 'types':
        return s[-1]

    # avoid things like layer.LayerProps -> LayerLayerProps
    a = s[-2].title()
    b = s[-1]
    if b.startswith(a):
        return b
    return a + b

class _Generator:
    def __init__(self, types, methods):
        self.types = types
        self.methods = [m for _, m in sorted(methods.items()) if m['category'] == 'api']
        self.decls = []
        self.impl = []
        self.done = {}

    def run(self):
        for m in self.methods:
            self.gen(m['arg'])
            self.gen(m['return'])

        ms = []
        for m in self.methods:
            ms.append(
                '\t/// %s\n\t%s(p: %s, options?: any): Promise<%s>;'
                % (m['doc'], m['cmd'], _short(m['arg']), _short(m['return'])))
        self.decls.append(
            'export interface GwsServerApi {\n%s\n}'
            % _nl2(ms))

        ms = []
        for m in self.methods:
            ms.append(
                '\tasync %s(p: gws.%s, options?: any): Promise<gws.%s> { return await this._call("%s", p, options) }'
                % (m['cmd'], _short(m['arg']), _short(m['return']), m['cmd']))

        return (
            _format(_decls_template, decls=_nl2(self.decls)),
            _format(_stub_template, methods=_nl(ms))
        )

    def gen(self, tname):
        if tname == 'str':
            return 'string'
        if tname == 'bool':
            return 'boolean'
        if tname not in self.types:
            return '_' + tname
        if tname in self.done:
            return self.done[tname]

        t = self.types[tname]
        name = _short(tname)
        self.done[tname] = name

        if t['type'] == 'object':
            props = []
            ext = ''
            if t.get('extends'):
                ext = ' extends ' + _comma(self.gen(e) for e in t['extends'])
            for p in t['props']:
                pt = self.gen(p['type'])
                props.append(
                    '\t/// %s\n\t%s%s: %s;'
                    % (p['doc'], p['name'], '?' if p['optional'] else '', pt))
            self.decls.append(
                '/// %s\nexport interface %s%s {\n%s\n}'
                % (t['doc'], name, ext, _nl(props)))

        elif t['type'] == 'list':
            b = self.gen(t['base'])
            self.decls.append('export type %s = Array<%s>;' % (name, b))

        elif t['type'] == 'tuple':
            b = _comma(self.gen(b) for b in t['bases'])
            self.decls.append('export type %s = [%s];' % (name, b))

        elif t['type'] == 'enum':
            b = [
                '\t%s = %s,' % (k, _val(v))
                for k, v in sorted(t['values'].items())
            ]
            self.decls.append('export enum %s {\n%s\n};' % (name, _nl(b)))

        elif t['type'] == 'literal':
            b = _val(t['default'])
            self.decls.append('export type %s = %s;' % (name, b))

        elif t['type'] == 'union':
            b = _pipe(sorted(self.gen(b) for b in t['bases']))
            self.decls.append('export type %s = %s;' % (name, b))

        elif t['type'] == 'alias':
            self.decls.append('export type %s = %s;' % (name, t['target']))

        return name


def generate(types, methods):
    return _Generator(types, methods).run()
