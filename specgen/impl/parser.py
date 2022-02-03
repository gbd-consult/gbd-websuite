"""Parse py source files and create a list of units of interest"""

import ast
import re


class Error(Exception):
    pass


def parse(paths):
    units = []
    parser = _Parser()

    for path in paths:
        try:
            units.extend(parser.parse(path))
        except Exception as e:
            raise Error(f'parse error in {path}') from e
    return units, parser.stubs


##

_builtins = 'str', 'int', 'float', 'bool', 'list', 'dict', 'bytes'

from .base import Unit, Stub


class _Parser:
    def __init__(self):
        self.stubs = {}

    def parse(self, path):
        self.mod_name = _mod_name(path)
        self.is_init = path.endswith('__init__.py')

        with open(path) as fp:
            self.buf = fp.read()
            self.lines = self.buf.splitlines()
            self.lines.insert(0, '')  # make it 1-based

        self.docs = self.extract_docs()

        self.imports = {}
        self.units = []

        tree = ast.parse(self.buf)

        self.enum_aliases(tree)
        self.enum_imports(tree)
        self.enum_classes(tree)
        self.enum_stubs(tree)
        self.enum_cli_functions(tree)

        return self.units

    def extract_docs(self):
        cmt = '#:'
        ds = {}

        for n, ln in enumerate(self.lines):
            ln = ln.strip()

            # comments can be placed before the prop like "#: blah <nl>foo"
            # or inline like "foo #: blah"

            if ln.startswith(cmt):
                ds[n + 1] = ln[len(cmt):].strip()
            elif cmt in ln:
                ds[n] = ln.split(cmt)[1].strip()

        return ds

    def enum_imports(self, tree):
        for node in _nodes(tree, 'Import'):
            for nn in node.names:
                if nn.asname:
                    self.imports[nn.asname] = nn.name

        for node in _nodes(tree, 'ImportFrom'):
            if not node.level:
                for nn in node.names:
                    self.imports[nn.asname or nn.name] = node.module + '.' + nn.name
            m = self.mod_name.split('.')
            lev = node.level
            if self.is_init:
                lev -= 1
            if lev:
                m = m[:-lev]
            m = '.'.join(m)
            if node.module:
                m += '.' + node.module
            for nn in node.names:
                self.imports[nn.asname or nn.name] = m + '.' + nn.name

    def enum_aliases(self, tree):
        mark = 'alias'
        for node in _nodes(tree, 'Assign'):
            doc = self.doc_for(node)
            if doc.startswith(mark):
                self.add(
                    'alias',
                    name=self.mod_name + '.' + node.targets[0].id,
                    doc=doc[len(mark):].strip(),
                    types=self.type_decl_to_type_list(node.value),
                )

    def enum_classes(self, tree):
        for node in _nodes(tree, 'ClassDef'):
            if 'ignore' in self.doc_for(node):
                continue

            class_uid = self.add(
                'class',
                name=self.mod_name + '.' + node.name,
                doc=_docstring(node),
                supers=[self.qname(b) for b in node.bases],
            )

            for b in node.body:
                self.member(b, class_uid)

    def enum_stubs(self, tree):
        mark = 'export'
        for node in _nodes(tree, 'ClassDef'):
            doc = self.doc_for(node)
            if doc.startswith(mark):
                name = doc[len(mark):].strip()
                self.stub(node, name or node.name)

    def enum_cli_functions(self, tree):
        cli_command = None

        for node in _nodes(tree, 'Module'):
            for b in node.body:
                if _cls(b) == 'Assign':
                    try:
                        if b.targets[0].id == 'COMMAND':
                            cli_command = b.value.s
                    except:
                        pass
                if _cls(b) == 'FunctionDef' and cli_command:
                    self.cli_function(b, cli_command)

    def member(self, node, class_uid):
        if _cls(node) == 'Assign':
            self.add(
                'valueprop',
                name=node.targets[0].id,
                doc=self.doc_for(node),
                parent=class_uid,
                default=_value(node.value, strict=False),
            )
            return

        if _cls(node) == 'AnnAssign':
            self.add(
                'prop',
                name=node.target.id,
                doc=self.doc_for(node),
                parent=class_uid,
                types=self.type_decl_to_type_list(node.annotation),
                default=_value(node.value),
            )
            return

        if _cls(node) == 'FunctionDef':

            method_uid = self.add(
                'method',
                name=node.name,
                doc=_docstring(node),
                parent=class_uid,
            )

            for a in node.args.args:
                # NB it's important to add arguments in order, see spec/_method_spec
                self.add(
                    'argument',
                    name=a.arg,
                    parent=method_uid,
                    types=self.type_decl_to_type_list(a.annotation)
                )

            self.add(
                'return',
                name='',
                parent=method_uid,
                types=self.type_decl_to_type_list(node.returns)
            )

    def cli_function(self, node, cli_command):
        if node.name.startswith('_'):
            return

        args = []

        for d in node.decorator_list:
            if _cls(d) == 'Call':
                name = getattr(d.func, 'id', None) or getattr(d.func, 'attr', None)
                if name == 'arg':
                    a = Unit(name=d.args[0].s)
                    for kw in d.keywords:
                        if kw.arg == 'help':
                            a.doc = kw.value.s
                            break
                    args.append(a)

        self.add(
            'clifunc',
            name=node.name,
            doc=_docstring(node),
            command=cli_command,
            args=args,
        )

    def stub(self, node, name):
        if name in self.stubs:
            raise ValueError('stub %r already declared' % name)
        stub = Stub(name, self.mod_name + '.' + node.name)
        stub.bases = [self.qname(b) for b in node.bases]
        for m in node.body:
            self.stub_member(m, stub)
        self.stubs[stub.name] = stub

    def stub_member(self, node, stub, with_funcs=True):

        def _attr_name(n):
            # expect Name(id='prop') or Attribute(value=Name(id='self'), attr='foo')
            if _cls(n) == 'Name':
                return n.id
            if _cls(n) == 'Attribute' and _cls(n.value) == 'Name' and n.value.id == 'self' and isinstance(n.attr, str):
                return n.attr

        if 'noexport' in self.doc_for(node):
            return

        line = self.lines[node.lineno]

        if _cls(node) == 'Assign':
            name = _attr_name(node.targets[0])
            if name and not name.startswith('_'):
                val = _value(node.value, strict=False)
                if val is not None:
                    stub.members[name] = {
                        'kind': 'prop',
                        'name': name,
                        'type_name': type(val).__name__,
                        'value': val,
                        'line': line.strip(),
                    }
            return

        if _cls(node) == 'AnnAssign':
            name = _attr_name(node.target)
            if name and not name.startswith('_'):
                val = _value(node.value, strict=False)
                stub.members[name] = {
                    'kind': 'prop',
                    'name': name,
                    'type_node': node.annotation,
                    'value': val,
                    'line': line.strip(),
                }
            return

        if with_funcs and _cls(node) == 'FunctionDef':
            if node.name in ('__init__', 'configure'):
                for b in node.body:
                    self.stub_member(b, stub, with_funcs=False)
                return

            if node.name.startswith('_'):
                return

            if node.decorator_list:
                line = self.lines[node.lineno]  # @TODO assuming a single decorator
                if node.decorator_list[0].id in ('property', 'cached_property'):
                    if node.returns:
                        stub.members[node.name] = {
                            'kind': 'prop',
                            'name': node.name,
                            'type_node': node.returns,
                            'line': line.strip(),
                        }
                    return

            stub.members[node.name] = {
                'kind': 'method',
                'name': node.name,
                'args': node.args,
                'returns': node.returns,
                'line': line.strip(),
            }

    def node_name(self, node):
        if _cls(node) == 'Name':
            name = node.id
            if name in self.imports:
                return self.imports[name]
            return name

        if _cls(node) == 'Attribute':
            return self.node_name(node.value) + '.' + node.attr

        if _cls(node) == 'Str':
            return node.s

        if _cls(node) == 'Constant':
            return node.value

        raise Error('unknown name node', node)

    def qname(self, node):
        name = self.node_name(node)
        if name in _builtins:
            return name
        if '.' in name:
            if name.startswith('ext.'):
                name = 'gws.types.' + name
            return name
        return self.mod_name + '.' + name

    def type_decl_to_type_list(self, node):
        # here, node is a type declaration (an alias or an annotation)
        if node is None:
            return ['void']

        # foo: SomeType
        if _cls(node) in ('Str', 'Name', 'Attribute', 'Constant'):
            return [self.qname(node)]

        # foo: List[SomeType] => [List, [SomeType]]
        # foo: Dict[A, B] => [Dict, [A, B]]
        if _cls(node) == 'Subscript':
            if _cls(node.value) == 'Index':
                # 3.8
                val = node.slice.value
            else:
                # 3.9
                val = node.slice
            s = self.type_decl_to_type_list(val)
            if s[0] == 'tuple':
                s = s[1]
            return [self.qname(node.value), s]

        # foo: [SomeType, SomeType]
        if _cls(node) == 'Tuple':
            # @TODO tuples/unions of compound types
            els = []
            for e in node.elts:
                t = self.type_decl_to_type_list(e)
                if len(t) > 1:
                    raise Error('unsupported tuple type', node)
                els.append(t[0])
            return ['tuple', els]

        raise Error('unsupported type', node)

    def add(self, kind, **kwargs):
        kwargs.update({
            'kind': kind,
            'module': self.mod_name
        })
        unit = Unit(**kwargs)
        self.units.append(unit)
        return unit.uid

    def doc_for(self, node):
        if node.lineno in self.docs:
            return self.docs[node.lineno]
        return ''


def _value(node, strict=True):
    if node is None:
        return None
    cc = _cls(node)
    if cc == 'Num':
        return node.n
    if cc == 'Str':
        return node.s
    if cc == 'Bytes':
        return node.s
    if cc == 'NameConstant':
        return node.value
    if cc == 'Constant':
        return node.value
    if cc == 'List':
        return [_value(e) for e in node.elts]
    if cc == 'Dict':
        return {
            _value(k): _value(v)
            for k, v in zip(node.keys, node.values)
        }
    if strict:
        raise Error('unknown value type', node)
    return None


def _mod_name(path):
    # /home/xyz/gbd-websuite/app/gws/common/search/__init__.py => gws.common.search
    m = re.search(r'/app/gws/([^.]+)', path)
    if not m:
        raise ValueError('cannot parse path %r' % path)
    p = 'gws.' + m.group(1).replace('/', '.')
    p = re.sub(r'.__init__$', '', p)
    return p


def _docstring(node):
    try:
        b = node.body[0]
        if _cls(b) == 'Expr' and _cls(b.value) in ('Constant', 'Str'):
            return b.value.s.strip()
    except:
        pass
    return ''


def _nodes(tree, cls):
    for node in ast.walk(tree):
        if _cls(node) == cls:
            yield node


def _cls(node):
    return node.__class__.__name__
