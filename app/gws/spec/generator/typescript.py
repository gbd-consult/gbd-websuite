"""Generate typescript API files from the server spec"""

import json
import re
from typing import List

from . import base


def generate(specs, options):
    g = _Generator(specs, options)
    return g.run()


##

class _Generator:
    def __init__(self, specs, options):
        self.specs = specs
        self.options = options
        self.commands = []
        self.decls = []
        self.stub = []
        self.done = {}
        self.rec = {}
        self.rec_cnt = 0
        self.object_names = {}

    def run(self):
        for _, spec in sorted(self.specs.items()):

            # export all API commands
            if spec.abc == base.ABC.command and spec.method == 'api':
                self.commands.append(base.Data(
                    spec=spec,
                    arg_type=self.type_name(spec.arg),
                    ret_type=self.type_name(spec.ret)
                ))

            # export all Props
            if spec.abc == base.ABC.object and self.is_instance(spec, 'gws.Props'):
                self.type_name(spec.name)

        text = _indent(self.write_api()) + '\n\n' + _indent(self.write_stub())
        for k, v in self.rec.items():
            text = text.replace(k, v)
        return text


    def write_api(self):
        api_tpl = """
            /**
             * Gws Server API.
             * Version $VERSION
             *
             */
             
            export const VERSION = '$VERSION'; 
            
            type _int = number;
            type _float = number;
            type _bytes = any;
            type _dict = {[k: string]: any};
            
            $decls
            
            export interface Api {
                $actions
            }
        """

        action_tpl = """
            /// $doc
            $name (p: $arg, options?: any): Promise<$ret>;
        """

        actions = [
            self.format(
                action_tpl,
                name=cc.spec.cmd_name,
                doc=cc.spec.doc,
                arg=cc.arg_type,
                ret=cc.ret_type,
            )
            for cc in self.commands
        ]
        return self.format(
            api_tpl,
            decls=_nl2(self.decls),
            actions=_nl2(actions))

    def write_stub(self):
        stub_tpl = """
            export abstract class BaseServer implements Api {
                abstract _call(cmd, p, options): Promise<any>;
                $actions
            } 
        """
        action_tpl = """
            async $name(p: $arg, options?: any): Promise<$ret> {
                return await this._call("$name", p, options);
            }
        """

        actions = [
            self.format(
                action_tpl,
                name=cc.spec.cmd_name,
                doc=cc.spec.doc,
                arg=cc.arg_type,
                ret=cc.ret_type,
            )
            for cc in self.commands
        ]
        return self.format(
            stub_tpl,
            actions=_nl(actions))

    _builtins_map = {
        base.BUILTIN.any: 'any',
        base.BUILTIN.bool: 'boolean',
        base.BUILTIN.bytes: '_bytes',
        base.BUILTIN.float: '_float',
        base.BUILTIN.int: '_int',
        base.BUILTIN.str: 'string',
    }

    def type_name(self, ref):

        if isinstance(ref, list):
            return self.complex_type_name(ref)

        if ref in self._builtins_map:
            return self._builtins_map[ref]

        if ref in self.done:
            return self.done[ref]

        if ref in self.specs:
            # the rec dict hold circular references
            self.rec_cnt += 1
            rec = f'[__REC__{self.rec_cnt}]'
            self.done[ref] = rec
            name = self.spec_type_name(self.specs[ref])
            self.done[ref] = name
            self.rec[rec] = name
            return name

        raise base.Error(f'unhandled type {ref!r}')

    def complex_type_name(self, ref):
        t, s = ref

        if t == base.T.dict:
            k = self.type_name(s[0])
            v = self.type_name(s[1])
            if k == 'string' and v == 'any':
                return '_dict'
            return '{[key: %s]: %s}' % (k, v)

        if t == base.T.list:
            return f'Array<%s>' % self.type_name(s)

        if t == base.T.literal:
            return _pipe(_val(v) for v in s)

        if t == base.T.tuple:
            return '[%s]' % _comma(self.type_name(e) for e in s)

        if t == base.T.union:
            return _pipe(sorted(self.type_name(e) for e in s))

        if t == base.T.variant:
            return _pipe(sorted(self.type_name(v) for v in s.values()))

    def spec_type_name(self, spec):
        if spec.abc == base.ABC.alias:
            if isinstance(spec.target, str):
                return self.type_name(spec.target)

            tpl = '''
                /// $doc
                export type $name = $target;
            '''
            name = self.object_name(spec.name)
            self.decls.append(self.format(
                tpl,
                name=name,
                doc=spec.doc,
                target=self.type_name(spec.target)
            ))
            return name

        if spec.abc == base.ABC.object:
            tpl = """
                /// $doc
                export interface $name$ext {
                    $props
                }
            """
            name = self.object_name(spec.name)
            self.decls.append(self.format(
                tpl,
                name=name,
                doc=spec.doc,
                ext=' extends ' + self.type_name(spec.super) if spec.super else '',
                props=self.make_props(spec)
            ))
            return name

        if spec.abc == base.ABC.enum:
            tpl = '''
                /// $doc
                export enum $name {
                    $items
                }
            '''
            name = self.object_name(spec.name)
            self.decls.append(self.format(
                tpl,
                name=name,
                doc=spec.doc,
                items=_nl('%s = %s,' % (k, _val(v)) for k, v in sorted(spec.values.items()))
            ))
            return name

        raise base.Error(f'unhandled type {spec.name!r}')

    _REMOVE_NAME_PARTS = ['gws', 'core', 'base', 'lib', 'types', 'action', 'plugins']

    def object_name(self, name):
        name = name.replace('_', '.')
        for g in base.GLOBAL_MODULES:
            if name.startswith(g):
                return name[len(g):]

        # 'gws.base.print.types.Params' => 'BasePrinterTypesParams'
        parts = name.split('.')
        res = ''.join(_ucfirst(s) for s in parts if s not in self._REMOVE_NAME_PARTS)
        if res in self.object_names and self.object_names[res] != name:
            raise base.Error(f'name conflict: {res!r} for {name!r} and {self.object_names[res]!r}')
        self.object_names[res] = name
        return res

    def make_props(self, spec):
        tpl = """
            /// $doc
            $name$opt: $type
        """
        ps = []
        for name, key in spec.props.items():
            p = self.specs[key]
            if p.owner == spec.name:
                ps.append(self.format(
                    tpl,
                    name=name,
                    doc=p.doc,
                    opt='?' if p.has_default else '',
                    type=self.type_name(p.type)))
        return _nl(ps)

    def format(self, template, **kwargs):
        kwargs['VERSION'] = self.options.version
        return re.sub(
            r'\$(\w+)',
            lambda m: kwargs[m.group(1)],
            template
        ).strip()

    def is_instance(self, spec, class_name):
        while True:
            if spec.super == class_name:
                return True
            if not spec.super:
                return False
            spec = self.specs[spec.super]


_pipe = ' | '.join
_comma = ', '.join
_nl = '\n'.join
_nl2 = '\n\n'.join


def _indent(txt):
    r = []

    spaces = ' ' * 4
    indent = 0

    for ln in txt.strip().split('\n'):
        ln = ln.strip()
        if ln == '}':
            indent -= 1
        ln = (spaces * indent) + ln
        if ln.endswith('{'):
            indent += 1
        r.append(ln)

    return _nl(r)


def _val(s):
    return json.dumps(s)


def _ucfirst(s):
    return s[0].upper() + s[1:]
