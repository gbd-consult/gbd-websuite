"""Generate the TypeScript API for the client."""

import json
import re

from . import base


def create(gen: base.Generator):
    """Create the TypeScript API.

    The API contains interfaces for ``gws.Request``, ``gws.Response`` and
    ``gws.Props`` and all their subclasses, the types they use, grouped in
    namespaces by module, and the ``Server`` interface with a typed ``call``
    method for each API command.

    Args:
        gen: Generator state.

    Returns:
        The TypeScript source.
    """

    return _Creator(gen).run()


##


class _Creator:
    """Builds the TypeScript declarations."""

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
        """Create the TypeScript API.

        Returns:
            The TypeScript source.
        """

        self.make_client_classes()
        self.make_client_commands()
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

    def make_client_classes(self):
        """Declare ``gws.Request``, ``gws.Response``, ``gws.Props`` and all their subclasses."""

        queue = ['gws.Request', 'gws.Response', 'gws.Props']
        while queue:
            uid = queue.pop(0)
            self.make(uid)
            for typ in self.gen.typeDict.values():
                if typ.c == base.c.CLASS and uid in typ.tSupers:
                    queue.append(typ.uid)

    def make_client_commands(self):
        """Collect the API commands with their argument and return types."""

        for typ in self.gen.typeDict.values():
            if typ.extName.startswith(base.v.EXT_COMMAND_API_PREFIX):
                self.commands[typ.extName] = base.Data(
                    cmdName=typ.extName.replace(base.v.EXT_COMMAND_API_PREFIX, ''),
                    doc=typ.doc,
                    arg=self.make(typ.tArg),
                    ret=self.make(typ.tReturn),
                )

    def make(self, uid):
        """Declare a type and get its TypeScript name.

        A temporary name is used while the type is being declared, so that
        recursive types can refer to it; it is replaced in ``write``.

        Args:
            uid: Type uid.

        Returns:
            The TypeScript type name or expression.
        """

        if uid in self._builtins_map:
            return self._builtins_map[uid]
        if uid in self.done:
            return self.done[uid]

        typ = self.gen.require_type(uid)

        tmp_name = f'[TMP:%d]' % (len(self.tmp_names) + 1)
        self.done[uid] = self.tmp_names[tmp_name] = tmp_name

        self.stack.append(typ.uid)
        type_name = self.make2(typ)
        self.stack.pop()

        self.done[uid] = self.tmp_names[tmp_name] = type_name
        return type_name

    def make2(self, typ):
        """Create the TypeScript type expression or declaration for a type.

        Args:
            typ: Type.

        Returns:
            The TypeScript type name or expression.

        Raises:
            ``Error``: If the type kind is not supported.
        """

        if typ.c == base.c.LITERAL:
            return _pipe(_val(v) for v in typ.literalValues)

        if typ.c in {base.c.LIST, base.c.SET}:
            return 'Array<%s>' % self.make(typ.tItem)

        if typ.c == base.c.OPTIONAL:
            return _pipe([self.make(typ.tTarget), 'null'])

        if typ.c == base.c.TUPLE:
            return '[%s]' % _comma(self.make(t) for t in typ.tItems)

        if typ.c == base.c.UNION:
            return _pipe(self.make(it) for it in typ.tItems)

        if typ.c == base.c.DICT:
            k = self.make(typ.tKey)
            v = self.make(typ.tValue)
            if k == 'string' and v == 'any':
                return '_dict'
            return '{[key: %s]: %s}' % (k, v)

        if typ.c == base.c.CLASS:
            return self.namespace_entry(
                typ,
                template='/// $doc \n  export interface $name$extends { \n $props \n }',
                props=self.make_props(typ),
                extends=' extends ' + self.make(typ.tSupers[0]) if typ.tSupers else '',
            )

        if typ.c == base.c.ENUM:
            return self.namespace_entry(
                typ,
                template='/// $doc \n export enum $name { \n $items \n }',
                items=_nl('%s = %s,' % (k, _val(v)) for k, v in sorted(typ.enumValues.items())),
            )

        if typ.c == base.c.VARIANT:
            target = _pipe(self.make(it) for it in typ.tMembers.values())
            return self.namespace_entry(
                typ,
                template='/// $doc \n export type $name = $target;',
                target=target,
            )

        if typ.c in base.c.TYPE:
            return self.namespace_entry(
                typ,
                template='/// $doc \n export type $name = $target;',
                target=self.make(typ.tTarget),
            )

        raise base.Error(f'unhandled type {typ.name!r}, stack: {self.stack!r}')

    CORE_NAME = 'core'
    """Namespace for top-level names; this part is also removed from qualified names."""

    def namespace_entry(self, typ, template, **kwargs):
        """Add a declaration to the namespace of a type.

        Args:
            typ: Type.
            template: Declaration template with ``$name`` placeholders.
            **kwargs: Template values.

        Returns:
            The qualified TypeScript name of the type.
        """

        ps = typ.name.split(DOT)
        if len(ps) == 1:
            ns, name, qname = self.CORE_NAME, ps[-1], self.CORE_NAME + DOT + ps[0]
        else:
            if self.CORE_NAME in ps:
                ps.remove(self.CORE_NAME)
            ns, name, qname = DOT.join(ps[:-1]), ps[-1], DOT.join(ps)
        self.namespaces.setdefault(ns, []).append(self.format(template, name=name, doc=typ.doc, **kwargs))
        return qname

    def make_props(self, typ):
        """Create the property declarations of a class, without inherited properties.

        Args:
            typ: Class type.

        Returns:
            The property declarations.
        """

        tpl = '/// $doc \n $name$opt: $type'
        props = []

        for name, uid in typ.tProperties.items():
            property_typ = self.gen.require_type(uid)
            if property_typ.tOwner == typ.name:
                props.append(
                    self.format(tpl, name=name, doc=property_typ.doc, opt='?' if property_typ.hasDefault else '', type=self.make(property_typ.tValue))
                )

        return _nl(props)

    ##

    def write(self):
        """Write the API and replace the temporary names.

        Returns:
            The indented TypeScript source.
        """

        text = _indent(self.write_api())
        for tmp, name in self.tmp_names.items():
            text = text.replace(tmp, name)
        return text

    def write_api(self):
        """Fill the API template with the namespaces and commands.

        Returns:
            The TypeScript source.
        """

        api_tpl = """
            /**
             * Gws Server API.
             * Version $VERSION
             *
             */

            export const GWS_VERSION = '$VERSION';

            type _int = number;
            type _float = number;
            type _bytes = any;
            type _dict = {[k: string]: any};

            $globs

            $namespaces

            interface _ServerArgs {
                $server_args
            }
            
            interface _ServerReturns {
                $server_rets
            }
            
            export interface Server {
                call<T extends keyof _ServerArgs>(cmd: T, r: _ServerArgs[T], options?: object): Promise<_ServerReturns[T]>;
                callAny(cmd: string, r: any, options?: object): Promise<any>;
            }
            
            export abstract class BaseServer implements Server {
                abstract execCall(cmd, r, options?): Promise<any>;
            
                call<T extends keyof _ServerArgs>(cmd: T, r: _ServerArgs[T], options?: object): Promise<_ServerReturns[T]> {
                    return this.execCall(cmd, r, options);
                }
                callAny(cmd: string, r: any, options?: object): Promise<any> {
                    return this.execCall(cmd, r, options);
                }
            }
        """

        namespace_tpl = 'export namespace $ns { \n $declarations \n }'

        globs = self.format(
            namespace_tpl,
            ns='gws',
            declarations=_nl2(self.namespaces.pop('gws')),
        )

        namespaces = _nl2(
            [
                self.format(
                    namespace_tpl,
                    ns=ns,
                    declarations=_nl2(d),
                )
                for ns, d in sorted(self.namespaces.items())
            ]
        )

        server_args = _nl(f'"{cc.cmdName}": {cc.arg}' for _, cc in sorted(self.commands.items()))
        server_rets = _nl(f'"{cc.cmdName}": {cc.ret}' for _, cc in sorted(self.commands.items()))

        return self.format(
            api_tpl,
            globs=globs,
            namespaces=namespaces,
            server_args=server_args,
            server_rets=server_rets,
        )

    def format(self, template, **kwargs):
        """Fill a template.

        ``$VERSION`` is set to the application version; of the ``doc`` value
        only the first line is used.

        Args:
            template: Template with ``$name`` placeholders.
            **kwargs: Template values.

        Returns:
            The filled template, stripped.
        """

        kwargs['VERSION'] = self.gen.meta['version']
        if 'doc' in kwargs:
            kwargs['doc'] = kwargs['doc'].split('\n')[0]
        return re.sub(r'\$(\w+)', lambda m: kwargs[m.group(1)], template).strip()


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
