"""Generate typescript API files from the server spec"""

import json
import re


def generate(types, methods):
    return _Generator(types, methods).run()


##

class _Generator:
    def __init__(self, types, methods):
        self.types = types
        self.methods = methods
        self.ifaces = []
        self.stub = []
        self.done = {}

    def run(self):
        for m in self.methods:
            self.gen(m.arg)
            self.gen(m.ret)
        return [
            self.write_types(),
            self.write_stub()
        ]

    def write_types(self):
        code = []
        for m in self.methods:
            code.append('')
            code.append('\t/// %s' % m.doc)
            code.append('\t%s(p: %s, options?: any): Promise<%s>;' % (
                m.cmd,
                _short_name(m.arg),
                _short_name(m.ret)))

        return _format(_types_template, ifaces=_nl2(self.ifaces), methods=_nl(code))

    def write_stub(self):
        code = []
        for m in self.methods:
            code.append('')
            code.append('\tasync %s(p: gws.%s, options?: any): Promise<gws.%s> {' % (
                m.cmd,
                _short_name(m.arg),
                _short_name(m.ret)))
            code.append('\t\treturn await this._call("%s", p, options);' % m.cmd)
            code.append('\t}')

        return _format(_stub_template, methods=_nl(code))

    def gen(self, tname):
        if tname == 'str':
            return 'string'

        if tname == 'bool':
            return 'boolean'

        if tname.endswith('.Any'):
            return 'any'

        if tname not in self.types:
            return '_' + tname

        if tname in self.done:
            return self.done[tname]

        t = self.types[tname]
        name = _short_name(tname)
        self.done[tname] = name

        if t.type == 'object':
            ext = _comma(self.gen(e) for e in t.extends)
            props = self.gen_props(t)
            self.ifaces.append(
                '/// %s\nexport interface %s%s {\n%s\n}'
                % (t.doc, name, ' extends ' + ext if ext else '', props))

        elif t.type == 'list':
            b = self.gen(t.bases[0])
            self.ifaces.append('export type %s = Array<%s>;' % (name, b))

        elif t.type == 'dict':
            k = self.gen(t.bases[0])
            v = self.gen(t.bases[1])
            self.ifaces.append('export type %s = {[key: %s]: %s};' % (name, k, v))

        elif t.type == 'tuple':
            b = _comma(self.gen(b) for b in t.bases)
            self.ifaces.append('export type %s = [%s];' % (name, b))

        elif t.type == 'enum':
            b = [
                '\t%s = %s,' % (k, _val(v))
                for k, v in sorted(t.values.items())
            ]
            self.ifaces.append('export enum %s {\n%s\n};' % (name, _nl(b)))

        elif t.type == 'literal':
            b = _val(t.default)
            self.ifaces.append('export type %s = %s;' % (name, b))

        elif t.type == 'taggedunion':
            b = _pipe(sorted(self.gen(b) for b in t.parts.values()))
            self.ifaces.append('export type %s = %s;' % (name, b))

        elif t.type == 'alias':
            self.ifaces.append('export type %s = %s;' % (name, self.gen(t.target)))

        return name

    def gen_props(self, t):
        ps = []
        for p in t.props:
            ps.append('\t/// %s' % p.doc)
            if p.type == 'literal':
                ps.append('\t%s: %r;' % (p.name, p.value))
            else:
                ptype = self.gen(p.type)
                ps.append('\t%s%s: %s;' % (p.name, '?' if p.optional else '', ptype))
        return _nl(ps)


_types_template = """
/**
 * Gws Server API.
 * Version @@VERSION@@
 */

type _int = number;
type _float = number;
type _bytes = any;
type _dict = {[k: string]: any};
type _list = Array<any>;

_ifaces_

export interface GwsServerApi {
_methods_
}


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


def _short_name(s):
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
