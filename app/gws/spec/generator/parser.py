"""Parse py source files and create a list of units of interest"""

import ast
import re

from typing import Dict, List, cast

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

    base.log.debug(f'parsing {path!r}')

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
    lines: List[str]
    module_node: ast.Module
    module_name: str
    docs: Dict[int, str]
    imports: Dict[str, str]

    def __init__(self, gen: base.Generator, module_name: str, path: str, text: str, parse_all: bool):
        self.gen = gen
        self.module_name = module_name
        self.module_path = path
        self.tModule = ''
        self.text = text
        self.source_lines = [''] + self.text.splitlines()
        self.is_init = path.endswith('__init__.py')
        self.context: List = []
        self.parse_all = parse_all

    def run(self):
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
            if _cls(node) == 'ClassDef':
                self.parse_class(node)
            elif _cls(node) == 'Assign':
                self.parse_assign(node, self.outer_doc(node, self.module_node.body))

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

    def parse_assign(self, node, doc):
        """Parse a module level assignment.
        
        It can be a constant or a type alias declaration.
        """

        name_node = node.targets[0]

        if len(node.targets) > 1 or _cls(name_node) != 'Name' or not _is_type_name(name_node.id):
            return

        # type declaration docstring must start with "type:" or "variant:"

        kind = None

        if doc.startswith(base.TYPE_COMMENT_PREFIX):
            kind = 'type'
            doc = doc.partition(base.TYPE_COMMENT_PREFIX)[-1].strip()
        elif doc.startswith(base.VARIANT_COMMENT_PREFIX):
            kind = 'variant'
            doc = doc.partition(base.VARIANT_COMMENT_PREFIX)[-1].strip()

        if kind:
            # type alias or a variant
            target_type = self.type_from_node(node.value)
            if kind == 'variant':
                if target_type.c != base.C.UNION:
                    raise ValueError('a Variant must be a Union')
                target_type = self.add(base.C.VARIANT, tItems=target_type.tItems)
            self.add(base.C.TYPE, doc=doc, ident=name_node.id, name=self.qname(name_node), tTarget=target_type.uid)
        else:
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
            extName=self.gws_decorator(node, 'class'),
        )

        for nn in self.nodes(node.body):
            mc = _cls(nn)
            if mc in {'Assign', 'AnnAssign'}:
                doc = self.outer_doc(nn, node.body)
                self.parse_property(typ, nn, doc, annotated=(mc == 'AnnAssign'))
            elif mc == 'FunctionDef':
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
            default=None,
            hasDefault=False,
        )

        c, value = self.parse_const_value(node.value)
        if c == base.C.LITERAL:
            typ.default = value
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
                t = type(typ.default).__name__
            property_type = self.type_from_name(t)

        if property_type:
            if property_type.c == base.C.OPTIONAL:
                typ.tValue = property_type.tTarget
                if not typ.hasDefault:
                    typ.default = None
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

            name = self.node_name(d.func)
            if not name.startswith(base.EXT_PREFIX):
                continue

            name = name + DOT + self.node_name(d.args[0])
            ns = name.split(DOT)

            if kind == 'class':
                if len(ns) == 4:
                    # gws.ext.config.project
                    return name + '.default'
                if len(ns) == 5:
                    # gws.ext.config.layer.wms
                    return name
                raise ValueError(f'invalid class decorator {name!r}')

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

        if g == 'List':
            return self.add(base.C.LIST, tItem=param_typ.uid if param_typ else 'any')

        if g == 'Set':
            return self.add(base.C.SET, tItem=param_typ.uid if param_typ else 'any')

        if g == 'Dict':
            if param_items:
                if len(param_items) != 2:
                    raise ValueError('invalid Dict arguments')
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

        if g == 'Tuple':
            if not param_typ:
                return self.add(base.C.TUPLE, tItems=[])
            if not param_items:
                raise ValueError('invalid Tuple')
            return self.add(base.C.TUPLE, tItems=list(param_items))

        if g == 'Callable':
            if not param_typ:
                return self.add(base.C.CALLABLE, tItems=[])
            if not param_items:
                raise ValueError('invalid Callable')
            return self.add(base.C.CALLABLE, tItems=list(param_items))

        if param:
            raise ValueError('invalid generic type')

        return self.add(base.C.UNDEFINED, name=name)

    ##

    @property
    def pos(self):
        return self.module_path + ':' + str(self.context[-1].lineno if self.context else 0)

    def add(self, c: str, **kwargs) -> base.Type:
        kwargs['pos'] = self.pos
        kwargs['tModule'] = self.tModule
        typ = self.gen.new_type(c, **kwargs)
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
        name = self.node_name(node)
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

    def node_name(self, node):
        cc = _cls(node)

        if cc == 'Name':
            return node.id
        if cc == 'Attribute':
            return self.node_name(node.value) + DOT + node.attr
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
    return ''


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


DOT = '.'
