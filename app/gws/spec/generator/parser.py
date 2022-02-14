"""Parse py source files and create a list of units of interest"""

import ast

from typing import Dict, List, cast

from . import base


def parse(state: base.ParserState, meta):
    for b in base.BUILTINS:
        state.types[b] = base.TAtom(name=b)

    state.types['TUncheckedEnum'] = base.TAtom(name='str')

    for chunk in meta.chunks:
        for path in chunk.paths['python']:
            parser = None
            try:
                mod_name = _module_name(chunk, path)
                text = read_file(path)
                parser = _Parser(state, mod_name, path, text, meta)
                parser.run()
            except Exception as e:
                lineno = '?'
                if parser and parser.context:
                    lineno = parser.context[-1].lineno
                msg = repr(e)
                if hasattr(e, 'args'):
                    msg = str(e.args[0])
                raise base.Error(f'{msg} in {path}:{lineno}')


##

def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read().strip()


def write_file(path, text):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


##

DOT = '.'


class _Parser:
    buf: str
    lines: List[str]
    module_node: ast.Module
    module_name: str
    docs: Dict[int, str]
    imports: Dict[str, str]

    def __init__(self, state, module_name: str, path: str, text: str, meta):
        self.state: base.ParserState = state
        self.module_name = module_name
        self.module_path = path
        self.text = text
        self.lines = [''] + self.text.splitlines()
        self.is_init = path.endswith('__init__.py')
        self.context: List = []
        self.meta = meta

    def run(self):
        tree = ast.parse(self.text)

        for node in ast.walk(tree):
            if _cls(node) == 'Module':
                self.module_node = cast(ast.Module, node)
                break
        else:
            raise ValueError('module node not found')

        self.docs = self.prepare_docs()
        self.imports = self.prepare_imports()

        for node in self.nodes('module', 'ClassDef'):
            self.parse_class(node)

        for node in self.nodes('module', 'Assign'):
            self.parse_type_alias(node)

    ##

    def prepare_docs(self):
        # comments can be placed before the prop like "#: blah <nl>foo"
        # or inline like "foo #: blah"

        cmt = '#:'
        docs = {}

        for n, ln in enumerate(self.lines):
            ln = ln.strip()
            if ln.startswith(cmt):
                docs[n + 1] = ln.split(cmt)[1].strip()
            elif cmt in ln:
                docs[n] = ln.split(cmt)[1].strip()

        return docs

    def prepare_imports(self):
        # map import names to module names
        imp = {}

        # "import a.b.c as foo" => {foo: a.b.c}
        for node in self.nodes('module', 'Import'):
            for nn in node.names:
                imp[nn.asname or nn.name] = nn.name

        for node in self.nodes('module', 'ImportFrom'):

            # "from a.b.c import foo" => {foo: a.b.c.foo}
            if node.level == 0:
                for nn in node.names:
                    imp[nn.asname or nn.name] = node.module + DOT + nn.name
                continue

            # "from . import foo" => {foo: "<mod-name>.{
            # "from .. import foo" => "<mod-name-before-dot>.foo"
            # "from ..abc import foo" => "<mod-name-before-dot>.abc.foo"

            m = self.module_name.split(DOT)
            level = node.level - self.is_init
            if level:
                m = m[:-level]
            m = DOT.join(m)
            if node.module:
                m += DOT + node.module
            for nn in node.names:
                imp[nn.asname or nn.name] = m + DOT + nn.name

        # create alias specs for imported types
        for alias, target in imp.items():
            if _is_type_name(alias):
                self.state.aliases[self.module_name + DOT + alias] = target

        return imp

    ##

    def parse_type_alias(self, node):
        """Parse a type alias TypeA = TypeB"""

        name_node = node.targets[0]

        if len(node.targets) > 1 or _cls(name_node) != 'Name' or not _is_type_name(name_node.id):
            return

        # we only accept aliases that have documentation strings '#: ...'
        doc = self.doc_for(node)
        if not doc:
            return

        target_type = self.type_from_node(node.value)

        # mypy doesn't accept aliases to special forms,
        # so we cannot use Variant = Union
        # instead, if the type is Union, look in the comment string for 'Variant'
        if isinstance(target_type, base.TUnion) and 'Variant' in doc:
            target_type = base.TVariantStub(items=target_type.items, pos=self.pos)

        self.add(target_type)

        self.add(base.TAlias(
            doc=doc,
            ident=name_node.id,
            name=self.qname(name_node),
            pos=self.pos,
            target_t=target_type.name,
        ))

    def parse_class(self, node):
        if not _is_type_name(node.name):
            return

        supers = [self.qname(b) for b in node.bases]
        if supers and (supers[0] == 'Enum' or supers[0].endswith('.Enum')):
            return self.parse_enum(node)

        spec = base.TRecord(
            doc=_docstring(node),
            ident=node.name,
            name=self.qname(node),
            pos=self.pos,
            ext_category='',
            ext_kind='',
            ext_type='',
            supers=[self.type_from_name(s).name for s in supers],
        )

        d = self.class_decorator(node)
        if d:
            spec.ext_category = d.category
            spec.ext_kind = d.kind
            spec.ext_type = d.type

        self.add(spec)

        for nn in self.nodes(node.body, 'Assign'):
            self.parse_property(spec, nn, annotated=False)

        for nn in self.nodes(node.body, 'AnnAssign'):
            self.parse_property(spec, nn, annotated=True)

        for nn in self.nodes(node.body, 'FunctionDef'):
            d = self.function_decorator(nn)
            if d and d.kind == 'command':
                self.parse_command(spec, nn, d.name)

    def parse_enum(self, node):
        docs = {}
        values = {}

        for nn in self.nodes(node.body, 'Assign'):
            ident = nn.targets[0].id
            ok, val = self.parse_value(nn.value)
            if not ok or not _is_scalar(val):
                raise ValueError(f'invalid Enum item {ident!r}')
            docs[ident] = self.doc_for(nn)
            values[ident] = val

        self.add(base.TEnum(
            doc=_docstring(node),
            ident=node.name,
            name=self.qname(node),
            pos=self.pos,
            docs=docs,
            values=values,
        ))

    def parse_property(self, owner_type: base.Type, node, annotated: bool):
        ident = node.target.id if annotated else node.targets[0].id
        if ident.startswith('_'):
            return

        has_default, default = self.parse_value(node.value)

        spec = base.TProperty(
            doc=self.doc_for(node),
            ident=ident,
            name=owner_type.name + DOT + ident,
            pos=self.pos,
            default=None,
            has_default=has_default,
            owner_t=owner_type.name,
            property_t='any',
        )

        if has_default:
            spec.default = default

        property_type = None
        if hasattr(node, 'annotation'):
            property_type = self.type_from_node(node.annotation)

        if not property_type:
            typ = 'any'
            if spec.has_default and spec.default is not None:
                typ = type(spec.default).__name__
            property_type = self.type_from_name(typ)

        if property_type:
            if isinstance(property_type, base.TOptional):
                spec.property_t = property_type.target_t
                if not spec.has_default:
                    spec.has_default = True
                    spec.default = None
            else:
                spec.property_t = property_type.name

        self.add(spec)

    def parse_command(self, owner_type: base.Type, node, command_name: str):
        # command names are strictly three parts: method . action . name
        # e.g. 'cli.server.restart
        method, action, cmd = command_name.split(DOT)

        spec = base.TCommand(
            doc=_docstring(node),
            ident=node.name,
            name=method + DOT + action + _ucfirst(cmd),  # cli.serverRestart
            pos=self.pos,
            owner_t=owner_type.name,
            cmd_action=action,  # server
            cmd_command=cmd,  # restart
            cmd_method=method,
            cmd_name=action + _ucfirst(cmd),  # serverRestart
            ext_type=cast(base.TRecord, owner_type).ext_type,
            arg_t='any',
            ret_t='any',
        )

        # action methods have only one spec'able arg (the last one)
        arg_node = node.args.args[-1]
        if arg_node.annotation:
            arg_type = self.type_from_node(arg_node.annotation)
            spec.arg_t = arg_type.name if arg_type else 'any'

        if node.returns:
            ret_type = self.type_from_node(node.returns)
            spec.ret_t = ret_type.name if ret_type else 'any'

        self.add(spec)

    def class_decorator(self, node):
        d = self.gws_decorator(node)
        if not d:
            return

        # e.g. gws.ext.Config('db.provider.foo')
        fn_parts = self.qname(d.func).split(DOT)
        ok, arg = self.parse_value(d.args[0])
        if not ok:
            raise ValueError('invalid argument')
        arg_parts = arg.split(DOT)

        return base.Data(
            category=DOT.join(arg_parts[:-1]),  # 'db.provider'
            type=arg_parts[-1],  # 'foo'
            kind=fn_parts[-1],  # 'Config'
            name=DOT.join(fn_parts[:-1] + arg_parts + fn_parts[-1:]),  # 'gws.ext.db.provider.foo.Config'
        )

    def function_decorator(self, node):
        d = self.gws_decorator(node)
        if not d:
            return

        # e.g. gws.ext.command('api.map.renderXYZ')
        fn_parts = self.qname(d.func).split(DOT)
        kind = fn_parts[-1]

        if kind == 'command':
            if not d.args:
                raise ValueError('invalid argument')
            ok, arg = self.parse_value(d.args[0])
            if not ok:
                raise ValueError('invalid argument')
            return base.Data(kind=kind, name=arg)

        raise ValueError(f'invalid decorator: "{kind}"')

    def gws_decorator(self, node):
        for d in getattr(node, 'decorator_list', []):
            if _cls(d) == 'Call' and self.qname(d.func).startswith(base.GWS_EXT_PREFIX + DOT):
                return d

    ##

    def type_from_node(self, node) -> base.Type:
        # here, node is a type declaration (an alias or an annotation)

        cc = _cls(node)

        # foo: SomeType
        if cc in {'Str', 'Name', 'Attribute', 'Constant'}:
            return self.type_from_name(self.qname(node))

        # foo: Generic[SomeType]
        if cc == 'Subscript':
            # py 3.8 and 3.9 interpret node.slice differently
            return self.type_from_name(
                self.qname(node.value),
                node.slice.value if _cls(node.slice) == 'Index' else node.slice)

        # foo: [SomeType, SomeType]
        if cc in {'List', 'Tuple'}:
            items = [self.type_from_node(e) for e in node.elts]
            return self.add(base.TTuple(items=[it.name for it in items]))

        raise ValueError(f'unsupported type: {cc!r}')

    def type_from_name(self, name: str, param=None) -> base.Type:
        if name in self.state.types:
            return self.state.types[name]

        g = name.split(DOT)[-1].lower()

        if g == 'any':
            return self.state.types['any']

        # literal - 'param' is a value or a tuple of values
        if g == 'literal':
            if not param:
                raise ValueError('invalid literal')
            values = []
            elts = param.elts if _cls(param) == 'Tuple' else [param]
            for elt in elts:
                values.append(self.parse_literal_value(elt))
            return self.add(base.TLiteral(values=values))

        # in other cases, 'param' is a type or  a tuple of types

        param_type = self.type_from_node(param) if param else None
        param_tuple = None
        if isinstance(param_type, base.TTuple):
            param_tuple = param_type.items

        if g == 'optional':
            if not param_type:
                raise ValueError('invalid optional type')
            return self.add(base.TOptional(target_t=param_type.name))

        if g == 'list':
            return self.add(base.TList(item_t=param_type.name if param_type else 'any'))

        if g == 'set':
            return self.add(base.TSet(item_t=param_type.name if param_type else 'any'))

        if g == 'dict':
            if param_tuple:
                if len(param_tuple) != 2:
                    raise ValueError('invalid Dict arguments')
                key, val = param_tuple
            elif param_type:
                key = 'str'
                val = param_type.name
            else:
                key = 'str'
                val = 'any'
            return self.add(base.TDict(key_t=key, value_t=val))

        if g == 'union':
            if not param_tuple:
                raise ValueError('invalid Union')
            return self.add(base.TUnion(items=sorted(param_tuple)))

        if g == 'tuple':
            if not param_type:
                return self.add(base.TTuple(items=[]))
            if not param_tuple:
                raise ValueError('invalid Tuple')
            return self.add(base.TTuple(items=param_tuple))

        if param:
            raise ValueError('invalid generic type')

        return self.add(base.TUnresolvedReference(name=name))

    ##

    @property
    def pos(self):
        return {
            'module_name': self.module_name,
            'module_path': self.module_path,
            'lineno': self.context[-1].lineno if self.context else 0,
        }

    def add(self, t: base.Type) -> base.Type:
        if not hasattr(t, 'pos'):
            setattr(t, 'pos', self.pos)
        self.state.types[t.name] = t
        return t

    def doc_for(self, node):
        if node.lineno in self.docs:
            return self.docs[node.lineno]
        return ''

    def qname(self, node):
        name = self.node_name(node)
        if name in base.BUILTINS:
            return name
        name = self.qualified(name)
        return name

    def qualified(self, name):
        for alias, mod in self.imports.items():
            if name == mod or name.startswith(mod + DOT):
                return name
            if name == alias:
                return mod
            if name.startswith(alias + DOT):
                return mod + DOT + name[(len(alias) + 1):]
        return self.module_name + DOT + name

    def node_name(self, node):
        if _cls(node) == 'Name':
            return node.id

        if _cls(node) == 'Attribute':
            return self.node_name(node.value) + DOT + node.attr

        if _cls(node) == 'Str':
            return node.s

        if _cls(node) == 'Constant':
            v = node.value
            return v if isinstance(v, str) else repr(v)

        if _cls(node) == 'ClassDef':
            return node.name

        raise ValueError('cannot find a node name')

    def nodes(self, where, *cls):
        if where == 'module':
            where = self.module_node.body

        for node in where:
            if not cls or _cls(node) in cls:
                self.context.append(node)
                yield node
                self.context.pop()

    ##

    def parse_value(self, node):
        if node is None:
            return False, None

        cc = _cls(node)

        if cc == 'Num':
            return True, node.n

        if cc in ('Str', 'Bytes'):
            return True, node.s

        if cc in ('Constant', 'NameConstant'):
            return True, node.value

        if cc == 'List':
            vals = []
            for elt in node.elts:
                ok, val = self.parse_value(elt)
                if not ok:
                    raise ValueError(f'invalid list element')
                vals.append(val)
            return True, vals

        if cc == 'Dict':
            dct = {}
            for k, v in zip(node.keys, node.values):
                ok1, key = self.parse_value(k)
                ok2, val = self.parse_value(v)
                if not ok1 or not ok2:
                    raise ValueError(f'invalid dict element')
                dct[key] = val
            return True, dct

        if cc == 'Attribute':
            # Something.someKey - possible enum value
            return True, base.TUncheckedEnum(name=self.qname(node))

        if cc == 'Name':
            # SomeConstant - consider unresolved
            return True, base.TUnresolvedReference(name=node.id)

        base.debug_log(f'unparsed value {cc!r}', base.Data(pos=self.pos))
        return False, None

    def parse_literal_value(self, node):
        cc = _cls(node)
        if cc == 'Num':
            return node.n
        if cc in ('Str', 'Bytes'):
            return node.s
        if cc in ('Constant', 'NameConstant'):
            return node.value
        raise ValueError(f'invalid literal value of type {cc!r}')


##

def _module_name(chunk, path):
    # <chunk.sourceDir>/a/b/c.py => <chunk.name>.a.b.c
    if not path.startswith(chunk.sourceDir):
        raise ValueError(f'invalid path {path!r}')

    p = path[len(chunk.sourceDir):].split('/')
    f = p.pop().split(DOT)[0]
    if f != '__init__':
        p.append(f)
    return chunk.name + DOT.join(p)


def _docstring(node):
    try:
        b = node.body[0]
        if _cls(b) == 'Expr' and _cls(b.value) in ('Constant', 'Str'):
            return b.value.s.strip()
    except:
        pass
    return ''


def _is_scalar(val):
    return isinstance(val, (str, bytes, int, float, bool))


def _is_type_name(name: str) -> bool:
    return (
            bool(name)
            and name[0].isupper()
            and all(s.upper() or s.islower() or s.isdigit() for s in name)
            and any(s.islower() for s in name)
    )


def _is_a(full_name: str, name: str) -> bool:
    # if the name is like 'Object', check if the full name ends with it
    # if the name is like 'some.module', check if the full name starts with it
    if name[0].isupper():
        return full_name == name or full_name.endswith(DOT + name)
    return full_name == name or full_name.startswith(name + DOT)


def _cls(node):
    return node.__class__.__name__


def _camelize(name):
    p = name.split('_')
    return p[0] + ''.join(_ucfirst(s) for s in p[1:])


def _ucfirst(s):
    return s[0].upper() + s[1:]


_comma = ','.join
