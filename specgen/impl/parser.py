"""Parse py source files and create a list of objects of interest"""

import ast
import re
import os


class Error(Exception):
    pass


def parse(root_dir):
    p = _Parser()
    paths = _find_files(root_dir, 'py$')
    # paths = [root_dir + '/types/__init__.py']

    for path in paths:
        try:
            p.parse(path)
        except Error as e:
            raise Error(e.args[0], path, e.args[1].lineno)
    return p.objects


##

_builtins = 'str', 'int', 'float', 'bool', 'list', 'dict'


class _Parser:

    def __init__(self):
        self.objects = []
        self._uid = 0

    def parse(self, path):
        self.imports = {}
        self.mod_name = _mod_name(path)
        self.path = path
        self.comments = {}

        with open(self.path) as fp:
            buf = fp.read()

        self.extract_comments(buf)

        tree = ast.parse(buf)

        self.enum_aliases(tree)
        self.enum_imports(tree)
        self.enum_classes(tree)
        self.enum_cli_functions(tree)

    def extract_comments(self, buf):
        cmt = '#:'

        for n, ln in enumerate(buf.splitlines(), 1):
            ln = ln.strip()

            # comments can be place before the prop like "#: blah <nl>foo"
            # or inline like "foo #: blah"

            if ln.startswith(cmt):
                self.comments[n + 1] = ln[len(cmt):].strip()
            elif cmt in ln:
                self.comments[n] = ln.split(cmt)[1].strip()

    def enum_imports(self, tree):
        for node in _nodes(tree, 'Import'):
            for nn in node.names:
                if nn.asname:
                    self.imports[nn.asname] = nn.name

    def enum_aliases(self, tree):
        mark = 'alias:'
        for node in _nodes(tree, 'Assign'):
            doc = self.doc_for(node)
            if doc.startswith(mark):
                self.add({
                    'kind': 'alias',
                    'uid': self.uid(),
                    'doc': doc[len(mark):].strip(),
                    'name': self.mod_name + '.' + node.targets[0].id,
                    'type': self.typelist(node.value),
                })

    def enum_classes(self, tree):
        for node in _nodes(tree, 'ClassDef'):
            uid = self.uid()

            for b in node.body:
                self.member(b, uid)

            self.add({
                'kind': 'object',
                'uid': uid,
                'doc': _docstring(node),
                'name': self.mod_name + '.' + node.name,
                'supers': [self.qname(b) for b in node.bases],
            })

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

    def member(self, node, cuid):
        cc = _cls(node)

        if cc == 'Assign':
            self.add({
                'kind': 'assign_prop',
                'uid': self.uid(),
                'parent_uid': cuid,
                'name': node.targets[0].id,
                'type': 'void',
                'default': _value(node.value, strict=False),
                'doc': self.doc_for(node),
            })
            return

        if cc == 'AnnAssign':
            self.add({
                'kind': 'prop',
                'uid': self.uid(),
                'parent_uid': cuid,
                'name': node.target.id,
                'type': self.typelist(node.annotation),
                'default': _value(node.value),
                'doc': self.doc_for(node),
            })
            return

        if cc == 'FunctionDef':

            uid = self.uid()

            for a in node.args.args:
                self.add({
                    'kind': 'arg',
                    'uid': self.uid(),
                    'parent_uid': uid,
                    'name': a.arg,
                    'type': self.typelist(a.annotation)
                })

            self.add({
                'kind': 'return',
                'uid': self.uid(),
                'parent_uid': uid,
                'name': '',
                'type': self.typelist(node.returns)
            })

            self.add({
                'kind': 'method',
                'uid': uid,
                'parent_uid': cuid,
                'name': node.name,
                'doc': _docstring(node),
            })

    def cli_function(self, node, cli_command):
        if node.name.startswith('_'):
            return

        args = []

        for d in node.decorator_list:
            if _cls(d) == 'Call':
                name = getattr(d.func, 'id', None) or getattr(d.func, 'attr', None)
                if name == 'arg':
                    a = {'name': d.args[0].s, 'doc': ''}
                    for kw in d.keywords:
                        if kw.arg == 'help':
                            a['doc'] = kw.value.s
                            break
                    args.append(a)

        self.add({
            'kind': 'clifunc',
            'uid': self.uid(),
            'doc': _docstring(node),
            'name': node.name,
            'command': cli_command,
            'args': args,
        })

    def qname_inner(self, node):
        cc = _cls(node)

        if cc == 'Name':
            name = node.id
            if name in self.imports:
                return self.imports[name]
            return name

        if cc == 'Attribute':
            return self.qname_inner(node.value) + '.' + node.attr

        if cc == 'Str':
            return node.s

        raise Error('unknown name node', node)

    def qname(self, node):
        name = self.qname_inner(node)
        if '.' in name or name in _builtins:
            return name
        return self.mod_name + '.' + name

    def typelist(self, node):
        if node is None:
            return 'void'

        cc = _cls(node)

        # if cc == 'Str':
        #     return [node.s]

        if cc in ('Str', 'Name', 'Attribute'):
            return [self.qname(node)]

        if cc == 'Subscript':
            return [self.qname(node.value)] + self.typelist(node.slice.value)

        if cc == 'Tuple':
            # @TODO tuples/unions of compound types
            els = []
            for e in node.elts:
                t = self.typelist(e)
                if len(t) > 1:
                    raise Error('unsupported tuple type', node)
                els.append(t[0])
            return ['tuple', els]

        raise Error('unsupported type', node)

    def add(self, d):
        d['module'] = self.mod_name
        self.objects.append(d)

    def doc_for(self, node):
        if node.lineno in self.comments:
            return self.comments[node.lineno]
        return ''

    def uid(self):
        self._uid += 1
        return self._uid


def _find_files(dirname, pattern):
    for fname in os.listdir(dirname):
        if fname.startswith('.'):
            continue

        path = os.path.join(dirname, fname)

        if os.path.isdir(path):
            yield from _find_files(path, pattern)
            continue

        if re.search(pattern, fname):
            yield path


def _cls(node):
    return node.__class__.__name__


def _value(node, strict=True):
    if node is None:
        return None
    cc = _cls(node)
    if cc == 'Num':
        return node.n
    if cc == 'Str':
        return node.s
    if cc == 'NameConstant':
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
        if _cls(b) == 'Expr' and _cls(b.value) == 'Str':
            return b.value.s.strip()
    except:
        pass
    return ''


def _nodes(tree, cls):
    for node in ast.walk(tree):
        if _cls(node) == cls:
            yield node
