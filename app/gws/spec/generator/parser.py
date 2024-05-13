"""Parse py source files and create a list of units of interest"""

import ast
import re

from typing import cast

from . import base, util


def parse(gen: base.Generator, parse_all=False):
    init_parser(gen)
    for chunk in gen.chunks:
        for path in chunk['paths']['python']:
            parse_path(gen, path, chunk['name'], chunk['sourceDir'], parse_all)


def init_parser(gen: base.Generator):
    for b in base.BUILTINS:
        typ = gen.new_type(base.C.ATOM, name=b)
        gen.types[typ.uid] = typ


def parse_path(gen: base.Generator, path: str, base_name: str, base_dir: str, parse_all=True):
    pp = None

    base.log.debug(f'parsing {path=}')

    try:
        # <base_dir>/a/b/__init__.py => <base_name>.a.b
        # <base_dir>/a/b/c.py => <base_name>.a.b.c

        if not path.startswith(base_dir):
            raise ValueError(f'invalid path {path!r}')
        p = path[len(base_dir):].split('/')
        f = p.pop().split(DOT)[0]
        if f != '__init__':
            p.append(f)
        mod_name = base_name + DOT.join(p)

        text = util.read_file(path)
        pp = _PythonParser(gen, mod_name, path, text, parse_all)
        pp.run()

    except Exception as exc:
        lineno = '?'
        if pp and pp.context:
            lineno = pp.context[-1].lineno
        msg = str(exc.args[0]) if hasattr(exc, 'args') else repr(exc)
        raise base.Error(f'{msg} in {path}:{lineno}')


##

class _PythonParser:
    lines: list[str]
    module_node: ast.Module
    module_name: str
    docs: dict[int, str]
    imports: dict[str, str]

    def __init__(self, gen: base.Generator, module_name: str, path: str, text: str, parse_all: bool):
        self.gen = gen
        self.module_name = module_name
        self.module_path = path
        self.tModule = ''
        self.text = text
        self.source_lines = [''] + self.text.splitlines()
        self.is_init = path.endswith('__init__.py')
        self.context: list = []
        self.parse_all = parse_all

    def run(self):
        if any('# gws:nospec' in ln for ln in self.source_lines):
            return

        tree = ast.parse(self.text)

        for node in ast.walk(tree):
            if _cls(node) == 'Module':
                self.module_node = cast(ast.Module, node)
                break
        else:
            raise ValueError('module node not found')

        typ = self.add(base.C.MODULE, name=self.module_name, path=self.module_path, doc=self.inner_doc(self.module_node))
        self.tModule = typ.uid

        self.imports = self.prepare_imports()

        for node in self.nodes(self.module_node.body):
            cc = _cls(node)
            if cc == 'Expr':
                self.parse_ext_declaration(node)
            if cc == 'ClassDef':
                self.parse_class(node)
            elif cc in {'Assign', 'AnnAssign'}:
                self.parse_assign(node, self.outer_doc(node, self.module_node.body), annotated=(cc == 'AnnAssign'))

    def prepare_imports(self):
        # map import names to module names
        imp = {}

        # "import a.b.c as foo" => {foo: a.b.c}
        for node in self.nodes(self.module_node.body, 'Import'):
            for nn in node.names:
                imp[nn.asname or nn.name] = nn.name

        for node in self.nodes(self.module_node.body, 'ImportFrom'):

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

        # create aliases for imported types
        for alias, target in imp.items():
            if _is_type_name(alias) and not _builtin_name(alias):
                self.gen.aliases[self.module_name + DOT + alias] = target

        return imp

    def parse_ext_declaration(self, node):
        if _cls(node.value) != 'Call':
            return
        call = cast(ast.Call, node.value)
        try:
            decl = _name(call.func)
        except ValueError:
            return
        if not decl.startswith(base.EXT_DECL_PREFIX):
            return
        if not call.args:
            raise ValueError('invalid gws.ext declaration')

        args = list(call.args)
        tail = decl.split(DOT).pop() + DOT + _name(args.pop(0))
        self.add(
            base.C.EXT,
            extName=base.EXT_OBJECT_PREFIX + tail,
            tTarget=self.qname(args.pop(0) if args else base.EXT_OBJECT_CLASS)
        )
        self.add(
            base.C.EXT,
            extName=base.EXT_CONFIG_PREFIX + tail,
            tTarget=self.qname(args.pop(0) if args else base.EXT_CONFIG_CLASS)
        )
        self.add(
            base.C.EXT,
            extName=base.EXT_PROPS_PREFIX + tail,
            tTarget=self.qname(args.pop(0) if args else base.EXT_PROPS_CLASS)
        )

    def parse_assign(self, node, doc, annotated):
        """Parse a module level assignment, possibly a type alias or a constant."""

        if annotated:
            name_node = node.target
        else:
            if len(node.targets) > 1:
                return
            name_node = node.targets[0]

        if _cls(name_node) != 'Name' or not _is_type_name(name_node.id):
            return

        typ = None
        if hasattr(node, 'annotation'):
            typ = self.type_from_node(node.annotation)

        if typ and typ.name == 'TypeAlias':
            # type alias
            target_type = self.type_from_node(node.value)
            if doc.startswith(base.VARIANT_COMMENT_PREFIX):
                # variant
                if target_type.c != base.C.UNION:
                    raise ValueError('a Variant must be a Union')
                doc = doc.partition(base.VARIANT_COMMENT_PREFIX)[-1].strip()
                target_type = self.add(base.C.VARIANT, tItems=target_type.tItems)
            self.add(base.C.TYPE, doc=doc, ident=name_node.id, name=self.qname(name_node), tTarget=target_type.uid)

            return

        # possibly, a constant

        c, value = self.parse_const_value(node.value)
        if c == base.C.LITERAL:
            self.add(base.C.CONSTANT, doc=doc, ident=name_node.id, name=self.qname(name_node), value=value)

    def parse_class(self, node):
        if not _is_type_name(node.name):
            return

        supers = [self.qname(b) for b in node.bases]
        if supers and _builtin_name(supers[0]) == 'Enum':
            return self.parse_enum(node)

        typ = self.add(
            base.C.CLASS,
            doc=self.inner_doc(node),
            ident=node.name,
            name=self.qname(node),
            tSupers=[self.type_from_name(s).uid for s in supers if not _builtin_name(s)],
        )

        for nn in self.nodes(node.body):
            cc = _cls(nn)
            if cc in {'Assign', 'AnnAssign'}:
                doc = self.outer_doc(nn, node.body)
                self.parse_property(typ, nn, doc, annotated=(cc == 'AnnAssign'))
            elif cc == 'FunctionDef':
                self.parse_method(typ, nn)

    def parse_enum(self, node):
        docs = {}
        vals = {}

        for nn in self.nodes(node.body):
            if _cls(nn) == 'Assign':
                ident = nn.targets[0].id
                c, value = self.parse_const_value(nn.value)
                if c != base.C.LITERAL:
                    raise ValueError(f'invalid Enum item {ident!r}')
                docs[ident] = self.outer_doc(nn, node.body)
                vals[ident] = value

        self.add(
            base.C.ENUM,
            doc=self.inner_doc(node),
            ident=node.name,
            name=self.qname(node),
            enumDocs=docs,
            enumValues=vals,
        )

    def parse_property(self, owner_typ: base.Type, node, doc: str, annotated: bool):
        ident = node.target.id if annotated else node.targets[0].id
        if ident.startswith('_'):
            return

        typ = self.add(
            base.C.PROPERTY,
            name=owner_typ.name + DOT + ident,
            doc=doc,
            ident=ident,
            tOwner=owner_typ.uid,
            tValue='any',
            defaultValue=None,
            hasDefault=False,
        )

        c, value = self.parse_const_value(node.value)
        if c == base.C.LITERAL:
            typ.defaultValue = value
            typ.hasDefault = True
        if c == base.C.EXPR:
            # see normalizer._evaluate_defaults
            typ.EVAL_DEFAULT = [c, value]

        property_type = None
        if hasattr(node, 'annotation'):
            property_type = self.type_from_node(node.annotation)

        if not property_type:
            t = 'any'
            if typ.hasDefault:
                t = type(typ.defaultValue).__name__
            property_type = self.type_from_name(t)

        if property_type:
            if property_type.c == base.C.OPTIONAL:
                typ.tValue = property_type.tTarget
                if not typ.hasDefault:
                    typ.defaultValue = None
                    typ.hasDefault = True
            else:
                typ.tValue = property_type.uid

    def parse_method(self, owner_typ: base.Type, node):
        ext = self.gws_decorator(node, 'method')

        if not ext and not self.parse_all:
            return

        typ = self.add(
            base.C.METHOD,
            doc=self.inner_doc(node),
            ident=node.name,
            name=owner_typ.name + DOT + node.name,
            tOwner=owner_typ.uid,
            tArgs=[],
            tReturn='any',
            extName=ext,
        )

        args = node.args.args
        if not self.parse_all:
            # ext methods have only one spec'able arg (the last one)
            args = args[-1:]

        for arg_node in args:
            t = 'any'
            if arg_node.annotation:
                arg_type = self.type_from_node(arg_node.annotation)
                if arg_type:
                    t = arg_type.uid
            typ.tArgs.append(t)

        if node.returns:
            ret_type = self.type_from_node(node.returns)
            typ.tReturn = ret_type.uid if ret_type else 'any'

    def gws_decorator(self, node, kind):
        for d in getattr(node, 'decorator_list', []):

            if _cls(d) != 'Call' or len(d.args) != 1:
                continue

            name = _name(d.func)
            if not name.startswith(base.EXT_PREFIX):
                continue

            name = name + DOT + _name(d.args[0])
            ns = name.split(DOT)

            if kind == 'method':
                if len(ns) == 5:
                    # gws.ext.command.api.mapGetBox
                    return name
                raise ValueError(f'invalid function decorator {name!r}')

        return ''

    ##

    def type_from_node(self, node) -> base.Type:
        # here, node is a type declaration (an alias or an annotation)

        cc = _cls(node)

        # foo: SomeType
        if cc in {'Str', 'Name', 'Attribute', 'Constant'}:
            return self.type_from_name(self.qname(node))

        # foo: Generic[SomeType]
        if cc == 'Subscript':
            # Subscript(slice=Index(value=Name... in py3.8
            # Subscript(slice=Name... in py3.9
            return self.type_from_name(
                self.qname(node.value),
                node.slice.value if _cls(node.slice) == 'Index' else node.slice)

        # foo: [SomeType, SomeType]
        if cc in {'List', 'Tuple'}:
            item_types = [self.type_from_node(e) for e in node.elts]
            return self.add(base.C.TUPLE, tItems=[typ.uid for typ in item_types])

        # foo: SomeType | SomeType | ...
        if cc == 'BinOp' and _cls(node.op) == 'BitOr':
            item_types = []
            while _cls(node) == 'BinOp' and _cls(node.op) == 'BitOr':
                item_types.insert(0, self.type_from_node(node.right))
                node = node.left
            item_types.insert(0, self.type_from_node(node))
            return self.add(base.C.UNION, tItems=[typ.uid for typ in item_types])

        raise ValueError(f'unsupported type: {cc!r}')

    def type_from_name(self, name: str, param=None) -> base.Type:
        if not param and name in self.gen.types:
            return self.gen.types[name]

        g = _builtin_name(name)

        if g == 'Any':
            return self.gen.types['any']

        # literal - 'param' is a value or a tuple of values
        if g == 'Literal':
            if not param:
                raise ValueError('invalid literal')
            elts = param.elts if _cls(param) == 'Tuple' else [param]
            vals = [self.parse_literal_value(e) for e in elts]
            return self.add(base.C.LITERAL, literalValues=vals)

        # in other cases, 'param' is a type or a tuple of types

        param_typ = param_items = None
        if param:
            param_typ = self.type_from_node(param)
            if param_typ.c == base.C.TUPLE:
                param_items = param_typ.tItems

        if g == 'Optional':
            if not param_typ:
                raise ValueError('invalid optional type')
            return self.add(base.C.OPTIONAL, tTarget=param_typ.uid)

        if g.lower() == 'list':
            return self.add(base.C.LIST, tItem=param_typ.uid if param_typ else 'any')

        if g.lower() == 'set':
            return self.add(base.C.SET, tItem=param_typ.uid if param_typ else 'any')

        if g.lower() == 'dict':
            if param_items:
                if len(param_items) != 2:
                    raise ValueError('invalid dict arguments')
                key, val = param_items
            elif param_typ:
                key = 'str'
                val = param_typ.uid
            else:
                key = 'str'
                val = 'any'
            return self.add(base.C.DICT, tKey=key, tValue=val)

        if g == 'Union':
            if not param_items:
                raise ValueError('invalid Union')
            return self.add(base.C.UNION, tItems=sorted(param_items))

        if g.lower() == 'tuple':
            if not param_typ:
                return self.add(base.C.TUPLE, tItems=[])
            if not param_items:
                raise ValueError('invalid Tuple')
            return self.add(base.C.TUPLE, tItems=list(param_items))

        if g.lower() == 'callable':
            if not param_typ:
                return self.add(base.C.CALLABLE, tItems=[])
            if not param_items:
                raise ValueError('invalid Callable')
            return self.add(base.C.CALLABLE, tItems=list(param_items))

        if param:
            raise ValueError('invalid generic type')

        if g:
            base.log.debug(f'created ATOM for {name!r}, builtin {g!r}')
            return self.add(base.C.ATOM, name=name)

        return self.add(base.C.UNDEFINED, name=name)

    ##

    @property
    def pos(self):
        return self.module_path + ':' + str(self.context[-1].lineno if self.context else 0)

    def add(self, c: str, **kwargs) -> base.Type:
        kwargs['pos'] = self.pos
        kwargs['tModule'] = self.tModule
        typ = self.gen.new_type(c, **kwargs)
        base.log.debug(f'added {typ.uid=} {typ=}')
        self.gen.types[typ.uid] = typ
        return typ

    def inner_doc(self, node):
        """Returns a normal docstring (first child of the node)."""

        return self.docstring_from(node.body[0]) if node.body else ''

    def outer_doc(self, node, nodes):
        """Returns a docstring which immediately follows this node in a list of nodes."""

        try:
            nxt = nodes[nodes.index(node) + 1]
        except IndexError:
            return ''
        return self.docstring_from(nxt)

    def docstring_from(self, node):
        """If node is a docstring, return its content."""

        if _cls(node) == 'Expr':
            if _cls(node.value) == 'Constant':
                v = node.value.value
                if isinstance(v, str):
                    return v.strip()
            if _cls(node.value) == 'Str':
                return node.value.s.strip()
        return ''

    def qname(self, node):
        name = _name(node)
        b = _builtin_name(name)
        if b:
            return b
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

    def nodes(self, where, *cls):
        for node in where:
            if not cls or _cls(node) in cls:
                self.context.append(node)
                yield node
                self.context.pop()

    ##

    def parse_literal_value(self, node):
        c, value = self.parse_const_value(node)
        if c == base.C.LITERAL and _is_scalar(value):
            return value
        raise ValueError(f'invalid literal value')

    def parse_const_value(self, node):
        if node is None:
            return None, None

        cc = _cls(node)

        if cc == 'Num':
            return base.C.LITERAL, node.n

        if cc in ('Str', 'Bytes'):
            return base.C.LITERAL, node.s

        if cc in ('Constant', 'NameConstant'):
            return base.C.LITERAL, node.value

        if cc in {'Name', 'Attribute'}:
            # SomeConstant or Something.someKey - possible constant/enum value
            return base.C.EXPR, self.qname(node)

        if cc in {'List', 'Tuple'}:
            exprlst, lst = [], []
            for elt in node.elts:
                c, value = self.parse_const_value(elt)
                if not c:
                    return False, None
                if c == base.C.LITERAL:
                    lst.append(value)
                exprlst.append([c, value])
            if len(lst) == len(exprlst):
                return base.C.LITERAL, lst
            return base.C.EXPR, exprlst

        if cc == 'Dict':
            exprdct, dct = {}, {}
            for k, v in zip(node.keys, node.values):
                c, key = self.parse_const_value(k)
                if c != base.C.LITERAL:
                    return False, None
                c, value = self.parse_const_value(v)
                if not c:
                    return False, None
                if c == base.C.LITERAL:
                    dct[key] = value
                exprdct[key] = [c, value]
            if len(dct) == len(exprdct):
                return base.C.LITERAL, dct
            return base.C.EXPR, exprdct

        return None, None


##


def _is_scalar(val):
    return isinstance(val, (str, bytes, int, float, bool))


def _is_type_name(name: str) -> bool:
    return bool(name) and bool(re.match(r'^[A-Z]', name))


def _builtin_name(name: str) -> str:
    if name in base.BUILTINS:
        return name
    if name in base.BUILTIN_TYPES:
        return name
    for b in base.BUILTIN_TYPES:
        if name.endswith(DOT + b):
            return b
        if name.startswith(b + DOT):
            return b
    return ''


def _is_a(full_name: str, name: str) -> bool:
    # if the name is like 'Object', check if the full name ends with it
    # if the name is like 'some.module', check if the full name starts with it
    if name[0].isupper():
        return full_name == name or full_name.endswith(DOT + name)
    return full_name == name or full_name.startswith(name + DOT)


def _cls(node):
    return node.__class__.__name__


def _name(node):
    if isinstance(node, str):
        return node

    cc = _cls(node)

    if cc == 'Name':
        return node.id
    if cc == 'Attribute':
        return _name(node.value) + DOT + node.attr
    if cc == 'Str':
        return node.s
    if cc == 'Constant':
        v = node.value
        return v if isinstance(v, str) else repr(v)
    if cc == 'ClassDef':
        return node.name
    if cc == 'FunctionDef':
        return node.name

    raise ValueError(f'node name missing in {cc!r}')


def _camelize(name):
    p = name.split('_')
    return p[0] + ''.join(_ucfirst(s) for s in p[1:])


def _ucfirst(s):
    return s[0].upper() + s[1:]


DOT = '.'
