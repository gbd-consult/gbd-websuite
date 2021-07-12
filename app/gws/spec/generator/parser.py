"""Parse py source files and create a list of units of interest"""

import ast
import re
from typing import List, Dict, cast

from . import base


def parse(options):
    specs = {}

    for chunk in options.chunks:
        for path in chunk.paths['python']:
            parser = None
            try:
                mod_name = _module_name(chunk, path)
                text = read_file(path)
                parser = _Parser(specs, mod_name, path, text, options)
                parser.run()
            except Exception as e:
                lineno = '?'
                if parser and parser.context:
                    lineno = parser.context[-1].lineno
                msg = repr(e)
                if hasattr(e, 'args'):
                    msg = str(e.args[0])
                raise base.Error(f'{msg} in {path}:{lineno}')

    return specs


##

DOT = '.'


class _Parser:
    buf: str
    lines: List[str]
    module_node: ast.Module
    module_name: str
    docs: Dict[int, str]
    imports: Dict[str, str]

    def __init__(self, specs, module_name: str, path: str, text: str, options):
        self.specs = specs
        self.module_name = module_name
        self.path = path
        self.text = text
        self.lines = [''] + self.text.splitlines()
        self.is_init = path.endswith('__init__.py')
        self.context: List = []
        self.options = options

    def run(self):
        tree = ast.parse(self.text)

        for node in ast.walk(tree):
            if _cls(node) == 'Module':
                self.module_node = cast(ast.Module, node)
                break
        else:
            raise ValueError('module node not found')

        self.add(self.spec(
            abc=base.ABC.module,
            name=self.module_name,
            path=self.path,
        ))

        self.docs = self.prepare_docs()
        self.imports = self.prepare_imports()

        for node in self.nodes('module', 'ClassDef'):
            self.create_class_spec(node)

        for node in self.nodes('module', 'Assign'):
            self.create_alias_spec(node)

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
        for alias, mod in imp.items():
            if _is_type_name(alias):
                self.add(self.spec(
                    abc=base.ABC.alias,
                    ident=alias,
                    name=self.module_name + DOT + alias,
                    target=mod,
                ))

        return imp

    ##

    def create_alias_spec(self, node):
        """Parse a type alias TypeA = TypeB"""

        name_node = node.targets[0]

        if len(node.targets) > 1 or _cls(name_node) != 'Name' or not _is_type_name(name_node.id):
            return

        # we only accept aliases that have documentation strings '#: ...'
        doc = self.doc_for(node)
        if not doc:
            return

        spec = self.spec(
            abc=base.ABC.alias,
            ident=name_node.id,
            name=self.qname(name_node),
            doc=doc,
            target=self.type_ref(node.value),
        )

        # mypy doesn't accept aliases to special forms,
        # so we cannot use Variant = Union
        # instead, if the type is Union, look in the comment string for 'Variant'

        if (
                isinstance(spec.target, list)
                and spec.target[0] == base.T.union
                and 'Variant' in spec.doc):
            spec.target[0] = base.T.unchecked_variant

        self.add(spec)

    def create_class_spec(self, node):
        if not _is_type_name(node.name):
            return

        sup = self.type_ref(node.bases[0]) if len(node.bases) > 0 else ''
        if _is_a(sup, 'Enum'):
            return self.create_enum_spec(node)

        spec = self.spec(
            abc=base.ABC.object,
            ident=node.name,
            name=self.qname(node),
            doc=_docstring(node),
            super=sup,
        )

        d = self.class_decorator(node)
        if d:
            spec.ext_category = d.category
            spec.ext_kind = d.kind
            spec.ext_type = d.type
            spec.name = d.name

        for nn in self.nodes(node.body, 'Assign'):
            self.create_property_spec(spec, nn, annotated=False)

        for nn in self.nodes(node.body, 'AnnAssign'):
            self.create_property_spec(spec, nn, annotated=True)

        for nn in self.nodes(node.body, 'FunctionDef'):
            d = self.function_decorator(nn)
            if not d:
                continue
            if d.kind == 'command':
                self.create_command_spec(spec, nn, d.name)

        self.add(spec)

    def create_enum_spec(self, node):
        spec = self.spec(
            abc=base.ABC.enum,
            ident=node.name,
            name=self.qname(node),
            doc=_docstring(node),
            docs={},
            values={},
        )

        for nn in self.nodes(node.body, 'Assign'):
            ident = nn.targets[0].id
            ok, val = self.parse_value(nn.value)
            if not ok or not _is_scalar(val):
                raise ValueError(f'invalid Enum item {ident!r}')
            spec.docs[ident] = self.doc_for(nn)
            spec.values[ident] = val

        self.add(spec)

    def create_property_spec(self, owner_spec, node, annotated):
        ident = node.target.id if annotated else node.targets[0].id
        if ident.startswith('_'):
            return

        has_default, default = self.parse_value(node.value)

        spec = self.spec(
            abc=base.ABC.property,
            owner=owner_spec.name,
            ident=ident,
            name=owner_spec.name + DOT + ident,
            doc=self.doc_for(node),
            has_default=has_default,
        )

        if has_default:
            spec.default = default

        type_ref = None
        if hasattr(node, 'annotation'):
            type_ref = self.type_ref(node.annotation)

        if not type_ref:
            type_name = 'any'
            if spec.has_default and spec.default is not None:
                type_name = type(spec.default).__name__
            type_ref = self.type_ref_from_name(type_name)

        if isinstance(type_ref, list) and type_ref[0] == base.T.optional:
            spec.type = type_ref[1]
            if not spec.has_default:
                spec.has_default = True
                spec.default = None
        else:
            spec.type = type_ref

        self.add(spec)

    def create_command_spec(self, owner_spec, node, command_name):
        method = command_name.split(DOT)[0].lower()

        # 'api.map.renderBox' => 'mapRenderBox'
        s = command_name.split(DOT)
        cmd_name = s[1] + ''.join(_ucfirst(p) for p in s[2:])

        spec = self.spec(
            abc=base.ABC.command,
            owner=owner_spec.name,
            ident=node.name,
            name=method + DOT + cmd_name,
            cmd_name=cmd_name,
            ext_type=owner_spec.ext_type,
            method=method,
            doc=_docstring(node),
            arg='any',
            ret='any',
        )

        # action methods have only one spec'able arg (the last one)
        arg_node = node.args.args[-1]
        if arg_node.annotation:
            spec.arg = self.type_ref(arg_node.annotation)

        if node.returns:
            spec.ret = self.type_ref(node.returns)

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
            if _cls(d) == 'Call' and self.qname(d.func).startswith(base.GWS_EXT_PREFIX):
                return d

    ##

    def type_ref(self, node):
        return self._type_ref(self.type_spec(node))

    def type_ref_from_name(self, name):
        return self._type_ref(self.type_spec_from_name(name))

    def _type_ref(self, spec):
        if not isinstance(spec, base.Data):
            return spec
        self.add(spec)
        return spec.name

    def type_spec(self, node):
        # here, node is a type declaration (an alias or an annotation)
        if node is None:
            return

        cc = _cls(node)

        # foo: SomeType
        if cc in {'Str', 'Name', 'Attribute', 'Constant'}:
            return self.type_spec_from_name(self.qname(node), None)

        # foo: List[SomeType]
        if cc == 'Subscript':
            return self.type_spec_from_name(self.qname(node.value), node.slice.value)

        # foo: [SomeType, SomeType]
        if cc in {'List', 'Tuple'}:
            elts = [self.type_ref(e) for e in node.elts]
            return ['tuple', elts]

        raise ValueError(f'unsupported type: {cc!r}')

    def type_spec_from_name(self, name, param=None):
        if name in base.BUILTINS:
            return name

        if name in self.specs and self.specs[name].abc != base.ABC.stub:
            return name

        g = name.split(DOT)[-1].lower()

        if g == 'any':
            return 'any'

        # literal - 'param' is a value or a tuple of values
        if g == 'literal':
            values = []
            elts = param.elts if _cls(param) == 'Tuple' else [param]
            for elt in elts:
                values.append(self.parse_literal_value(elt))
            return [base.T.literal, values]

        # in other cases, 'param' is a type or  a tuple of types

        param_ref = self.type_ref(param)
        param_tuple = None
        if isinstance(param_ref, list) and param_ref[0] == 'tuple':
            param_tuple = param_ref[1]

        if g == 'optional':
            if not param_ref:
                raise ValueError('invalid optional type')
            return [base.T.optional, param_ref]

        if g == 'list':
            return [base.T.list, param_ref or 'any']

        if g == 'dict':
            if param_tuple:
                if len(param_tuple) != 2:
                    raise ValueError('invalid Dict arguments')
                key, val = param_tuple
                if key != 'str':
                    raise ValueError('Dict keys must be str')
            elif param_ref:
                key = 'str'
                val = param_ref
            else:
                key = 'str'
                val = 'any'

            return [base.T.dict, [key, val]]

        if g == 'union':
            if not param_tuple:
                raise ValueError('invalid Union')
            return [base.T.union, param_tuple]

        if g == 'tuple':
            if not param_ref:
                return [base.T.tuple, []]
            if not param_tuple:
                raise ValueError('invalid Tuple')
            return param_ref

        if param:
            raise ValueError('invalid generic type')

        return self.spec(abc=base.ABC.stub, name=name)

    ##

    def add(self, spec, key=None):
        if not spec:
            return
        key = key or spec.get('name')
        self.specs[key] = spec
        return key

    def spec(self, **kwargs):
        lineno = 0
        if self.context:
            lineno = self.context[-1].lineno
        return base.Data(module=self.module_name, lineno=lineno, **kwargs)

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
            return True, [base.T.unchecked_enum, self.qname(node)]

        base.warn('unparsed value', cc, self.module_name)
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

def read_file(path):
    with open(path, 'rt', encoding='utf8') as fp:
        return fp.read().strip()


def write_file(path, text):
    with open(path, 'wt', encoding='utf8') as fp:
        fp.write(text)


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
