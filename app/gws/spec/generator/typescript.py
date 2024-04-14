"""Generate typescript API files from the server spec"""

import json
import re

from . import base


def create(gen: base.Generator):
    return _Creator(gen).run()


##

class _Creator:
    def __init__(self, gen: base.Generator):
        self.gen = gen
        self.commands = {}
        self.namespaces = {}
        self.stub = []
        self.done = {}
        self.stack = []
        self.tmp_names = {}
        self.object_names = {}

    def run(self):
        for typ in self.gen.types.values():
            if typ.extName.startswith(base.EXT_COMMAND_API_PREFIX):
                self.commands[typ.extName] = base.Data(
                    cmdName=typ.extName.replace(base.EXT_COMMAND_API_PREFIX, ''),
                    doc=typ.doc,
                    arg=self.make(typ.tArgs[-1]),
                    ret=self.make(typ.tReturn)
                )

        return self.write()

    _builtins_map = {
        'any': 'any',
        'bool': 'boolean',
        'bytes': '_bytes',
        'float': '_float',
        'int': '_int',
        'str': 'string',
        'dict': '_dict',
    }

    def make(self, uid):
        if uid in self._builtins_map:
            return self._builtins_map[uid]
        if uid in self.done:
            return self.done[uid]

        typ = self.gen.types[uid]

        tmp_name = f'[TMP:%d]' % (len(self.tmp_names) + 1)
        self.done[uid] = self.tmp_names[tmp_name] = tmp_name

        self.stack.append(typ.uid)
        type_name = self.make2(typ)
        self.stack.pop()

        self.done[uid] = self.tmp_names[tmp_name] = type_name
        return type_name

    def make2(self, typ):
        if typ.c == base.C.LITERAL:
            return _pipe(_val(v) for v in typ.literalValues)

        if typ.c in {base.C.LIST, base.C.SET}:
            return 'Array<%s>' % self.make(typ.tItem)

        if typ.c == base.C.OPTIONAL:
            return _pipe([self.make(typ.tTarget), 'null'])

        if typ.c == base.C.TUPLE:
            return '[%s]' % _comma(self.make(t) for t in typ.tItems)

        if typ.c == base.C.UNION:
            return _pipe(self.make(it) for it in typ.tItems)

        if typ.c == base.C.VARIANT:
            return _pipe(self.make(it) for it in typ.tMembers.values())

        if typ.c == base.C.DICT:
            k = self.make(typ.tKey)
            v = self.make(typ.tValue)
            if k == 'string' and v == 'any':
                return '_dict'
            return '{[key: %s]: %s}' % (k, v)

        if typ.c == base.C.CLASS:
            return self.namespace_entry(
                typ,
                template="/// $doc \n  export interface $name$extends { \n $props \n }",
                props=self.make_props(typ),
                extends=' extends ' + self.make(typ.tSupers[0]) if typ.tSupers else '')

        if typ.c == base.C.ENUM:
            return self.namespace_entry(
                typ,
                template="/// $doc \n export enum $name { \n $items \n }",
                items=_nl('%s = %s,' % (k, _val(v)) for k, v in sorted(typ.enumValues.items())))

        if typ.c == base.C.TYPE:
            return self.namespace_entry(
                typ,
                template="/// $doc \n export type $name = $target;",
                target=self.make(typ.tTarget))

        raise base.Error(f'unhandled type {typ.name!r}, stack: {self.stack!r}')

    CORE_NAME = 'core'

    def namespace_entry(self, typ, template, **kwargs):
        ps = typ.name.split(DOT)
        ps.pop(0)
        if len(ps) == 1:
            ns, name, qname = self.CORE_NAME, ps[-1], self.CORE_NAME + DOT + ps[-1]
        else:
            if self.CORE_NAME in ps:
                ps.remove(self.CORE_NAME)
            ns, name, qname = DOT.join(ps[:-1]), ps[-1], DOT.join(ps)
        self.namespaces.setdefault(ns, []).append(self.format(template, name=name, doc=typ.doc, **kwargs))
        return qname

    def make_props(self, typ):
        tpl = "/// $doc \n $name$opt: $type"
        props = []

        for name, uid in typ.tProperties.items():
            property_typ = self.gen.types[uid]
            if property_typ.tOwner == typ.name:
                props.append(self.format(
                    tpl,
                    name=name,
                    doc=property_typ.doc,
                    opt='?' if property_typ.hasDefault else '',
                    type=self.make(property_typ.tValue)))

        return _nl(props)

    ##

    def write(self):
        text = _indent(self.write_api()) + '\n\n' + _indent(self.write_stub())
        for tmp, name in self.tmp_names.items():
            text = text.replace(tmp, name)
        return text

    def write_api(self):

        namespace_tpl = "export namespace $ns { \n $declarations \n }"
        globs = self.format(namespace_tpl, ns=self.CORE_NAME, declarations=_nl2(self.namespaces.pop(self.CORE_NAME, '')))
        # globs = ''
        namespaces = _nl2([
            self.format(namespace_tpl, ns=ns, declarations=_nl2(d))
            for ns, d in sorted(self.namespaces.items())
        ])

        command_tpl = "/// $doc \n $name (p: $arg, options?: any): Promise<$ret>;"
        commands = _nl2([
            self.format(command_tpl, name=cc.cmdName, doc=cc.doc, arg=cc.arg, ret=cc.ret)
            for _, cc in sorted(self.commands.items())
        ])

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

            $globs

            $namespaces

            export interface Api {
                $commands
            }
        """

        return self.format(api_tpl, globs=globs, namespaces=namespaces, commands=commands)

    def write_stub(self):
        command_tpl = """$name(p: $arg, options?: any): Promise<$ret> { \n return this._call("$name", p, options); \n }"""
        commands = [
            self.format(command_tpl, name=cc.cmdName, doc=cc.doc, arg=cc.arg, ret=cc.ret)
            for _, cc in sorted(self.commands.items())
        ]

        stub_tpl = """
            export abstract class BaseServer implements Api {
                abstract _call(cmd, p, options): Promise<any>;
                $commands
            }
        """
        return self.format(stub_tpl, commands=_nl(commands))

    def format(self, template, **kwargs):
        kwargs['VERSION'] = self.gen.meta['version']
        if 'doc' in kwargs:
            kwargs['doc'] = kwargs['doc'].split('\n')[0]
        return re.sub(
            r'\$(\w+)',
            lambda m: kwargs[m.group(1)],
            template
        ).strip()


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


_pipe = ' | '.join
_comma = ', '.join
_nl = '\n'.join
_nl2 = '\n\n'.join

DOT = '.'
