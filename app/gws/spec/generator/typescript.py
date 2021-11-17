"""Generate typescript API files from the server spec"""

import collections
import json
import re

from . import base


def generate(state: base.ParserState, meta):
    g = _Generator(state, meta)
    return g.run()


##

class _Generator:
    def __init__(self, state, meta):
        self.state = state
        self.meta = meta
        self.commands = {}
        self.declarations = collections.defaultdict(list)
        self.stub = []
        self.done = {}
        self.stack = []
        self.tmp_names = {}
        self.object_names = {}

    def run(self):
        for t in self.state.types.values():
            # export all API commands
            if isinstance(t, base.TCommand) and t.cmd_method == 'api':
                self.commands[t.cmd_name] = base.Data(
                    cmd_name=t.cmd_name,
                    doc=t.doc,
                    arg=self.make(t.arg_t),
                    ret=self.make(t.ret_t)
                )

            # # export all Props
            # if spec.abc == base.ABC.object and self.is_instance(spec, 'gws.Props'):
            #     self.make(spec.name)

        text = _indent(self.write_api()) + '\n\n' + _indent(self.write_stub())
        for tmp, name in self.tmp_names.items():
            text = text.replace(tmp, name)
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

            $declarations

            export interface Api {
                $actions
            }
        """

        action_tpl = """
            /// $doc
            $name (p: $arg, options?: any): Promise<$ret>;
        """

        actions = [
            self.format(action_tpl, name=cc.cmd_name, doc=cc.doc, arg=cc.arg, ret=cc.ret)
            for _, cc in sorted(self.commands.items())
        ]

        namespace_tpl = """
            export namespace $ns {
                $declarations
            }
        """

        decls = []
        d = self.declarations.pop('core')
        decls.append(self.format(namespace_tpl, ns='core', declarations=_nl2(d)))

        for ns, d in sorted(self.declarations.items()):
            decls.append(self.format(namespace_tpl, ns=ns, declarations=_nl2(d)))

        return self.format(api_tpl, declarations=_nl2(decls), actions=_nl2(actions))

    def write_stub(self):
        stub_tpl = """
            export abstract class BaseServer implements Api {
                abstract _call(cmd, p, options): Promise<any>;
                $actions
            }
        """
        action_tpl = """
            $name(p: $arg, options?: any): Promise<$ret> {
                return this._call("$name", p, options);
            }
        """

        actions = [
            self.format(action_tpl, name=cc.cmd_name, doc=cc.doc, arg=cc.arg, ret=cc.ret)
            for _, cc in sorted(self.commands.items())
        ]

        return self.format(stub_tpl, actions=_nl(actions))

    _builtins_map = {
        'any': 'any',
        'bool': 'boolean',
        'bytes': '_bytes',
        'float': '_float',
        'int': '_int',
        'str': 'string',
    }

    def make(self, name):
        if name in self._builtins_map:
            return self._builtins_map[name]
        if name in self.done:
            return self.done[name]

        t = self.state.types[name]

        tmp_name = f'[TMP:%d]' % (len(self.tmp_names) + 1)
        self.done[name] = self.tmp_names[tmp_name] = tmp_name

        self.stack.append(t.name)
        type_name = self.make2(t)
        self.stack.pop()

        self.done[name] = self.tmp_names[tmp_name] = type_name
        return type_name

    def make2(self, t):
        if isinstance(t, base.TDict):
            k = self.make(t.key_t)
            v = self.make(t.value_t)
            if k == 'string' and v == 'any':
                return '_dict'
            return '{[key: %s]: %s}' % (k, v)

        if isinstance(t, base.TList):
            return 'Array<%s>' % self.make(t.item_t)

        if isinstance(t, base.TSet):
            return 'Array<%s>' % self.make(t.item_t)

        if isinstance(t, base.TLiteral):
            return _pipe(_val(v) for v in t.values)

        if isinstance(t, base.TOptional):
            return _pipe([self.make(t.target_t), 'null'])

        if isinstance(t, base.TTuple):
            return '[%s]' % _comma(self.make(it) for it in t.items)

        if isinstance(t, base.TUnion):
            return _pipe(self.make(it) for it in t.items)

        if isinstance(t, base.TVariant):
            return _pipe(self.make(it) for it in t.members.values())

        if isinstance(t, base.TRecord):
            tpl = """
                /// $doc
                export interface $name$ext {
                    $props
                }
            """
            ns, name, full = self.object_name_parts(t.name)
            self.declarations[ns].append(self.format(
                tpl,
                name=name,
                doc=t.doc,
                ext=' extends ' + self.make(t.supers[0]) if t.supers else '',
                props=self.make_props(t)
            ))
            return full

        if isinstance(t, base.TEnum):
            tpl = '''
                /// $doc
                export enum $name {
                    $items
                }
            '''
            ns, name, full = self.object_name_parts(t.name)
            self.declarations[ns].append(self.format(
                tpl,
                name=name,
                doc=t.doc,
                items=_nl('%s = %s,' % (k, _val(v)) for k, v in sorted(t.values.items()))
            ))
            return full

        if isinstance(t, base.TAlias):
            tpl = '''
                /// $doc
                export type $name = $target;
            '''
            ns, name, full = self.object_name_parts(t.name)
            self.declarations[ns].append(self.format(
                tpl,
                name=name,
                doc=t.doc,
                target=self.make(t.target_t)
            ))
            return full

        raise base.Error(f'unhandled type {t.name!r} in {self.stack!r}')

    def make_props(self, t):
        tpl = """
            /// $doc
            $name$opt: $type
        """

        props = []

        for name, key in t.props.items():
            property_type = self.state.types[key]
            if property_type.owner_t == t.name:
                props.append(self.format(
                    tpl,
                    name=name,
                    doc=property_type.doc,
                    opt='?' if property_type.has_default else '',
                    type=self.make(property_type.property_t)))

        return _nl(props)

    def object_name_parts(self, name):
        els = [
            e for e in name.replace('_', '.').split('.')
            if e not in {'gws', 'core', 'types'}
        ]
        if len(els) == 1:
            els.insert(0, 'core')
        if len(els) == 2 and els[0] == 'data':
            els[0] = 'core'
        if els[0] in {'base', 'gis', 'lib'}:
            els = els[1:]

        return '.'.join(els[:-1]), els[-1], '.'.join(els)

    def format(self, template, **kwargs):
        kwargs['VERSION'] = self.meta.version
        return re.sub(
            r'\$(\w+)',
            lambda m: kwargs[m.group(1)],
            template
        ).strip()


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
