"""Create stub sources from parsed stubs."""

import re

_indent = ' ' * 4
_comma = ', '.join
_nl = '\n'.join
_nl2 = '\n\n'.join


def run(stubs):
    classes = {s.class_name: s for s in stubs.values()}

    for stub in stubs.values():
        stub.bases = [_format_base(b, classes) for b in stub.bases]

    for stub in stubs.values():
        stub.bases = [b for b in stub.bases if b and b in stubs]

    rs = _topsort(stubs)

    for stub in rs:
        _check_props(stub, stubs)

    return _nl2(_code(stub) for stub in rs)


def _format_base(b, classes):
    if b in classes:
        b = classes[b].name
    if '.' not in b:
        return b
    s = b.split('.')
    if len(s) == 2 and s[0] == 'gws':
        return s[1]
    if len(s) == 3 and s[0] == 'gws' and s[1] == 'types':
        return s[2]
    return b


def _check_props(stub, stubs):
    for b in stub.bases:
        base = stubs.get(b)
        if base:
            stub.members = {k: v for k, v in stub.members.items() if k not in base.members}


def _topsort(stubs):
    ls = list(stubs.values())
    rs = []
    seen = set()

    while ls:
        ls2 = []
        rs2 = []
        for stub in ls:
            if all(b in seen for b in stub.bases):
                rs2.append(stub)
            else:
                ls2.append(stub)
        seen.update(s.name for s in rs2)
        rs.extend(sorted(rs2, key=lambda s: s.name))
        ls = ls2

    return rs


def _code(stub):
    head = 'class %s' % stub.name
    if stub.bases:
        head += '(' + _comma(b for b in stub.bases) + ')'
    head += ':'

    code = []
    props = []
    methods = []

    for name, m in sorted(stub.members.items()):
        kind, args_node, type_node, line = m
        if kind == 'p':
            props.append(_prop_code(name, type_node, line))
        if kind == 'm':
            methods.append(_method_code(name, args_node, type_node, line))

    for s in props:
        code.append(_indent + s)

    for s in methods:
        code.append(_indent + s)

    code = _nl(code) if code else (_indent + 'pass')
    return head + '\n' + code


# @TODO use the ast

def _prop_code(name, type_node, line):
    if isinstance(type_node, str):
        t = type_node
    elif '->' in line:
        t = _extract_return_annotation(line)
    else:
        t = _extract_var_annotation(line)
    return '%s : %s' % (name, t)


def _method_code(name, args_node, type_node, line):
    t = _format_function_decl(line)
    return t + ' pass'


def _extract_var_annotation(line):
    m = re.match(r'^[\w.]+\s*:([^=]+)', line.strip())
    if not m:
        raise ValueError('cannot parse %r' % line)
    return _quote_all(m.group(1).strip())


def _extract_return_annotation(line):
    m = re.search(r'->(.+?):', line.strip())
    if not m:
        raise ValueError('cannot parse %r' % line)
    return _quote_all(m.group(1).strip())


def _format_function_decl(line):
    def _repl(m):
        return m.group(1) + _quote_all(m.group(2))

    line = line.strip()
    line = re.sub(r'(:\s*)([^,=]+)', _repl, line)
    line = re.sub(r'(->\s*)([^:]+)', _repl, line)

    return line


def _quote_all(s):
    def _repl(m):
        return _quote(m.group(0))

    return re.sub(r"""(?x) '[\w.]+' | "[\w.]+" | [\w.]+""", _repl, s)


_no_quote = set('Any Dict List Optional Tuple Union Object Data Config Props Attribute'.split())


def _quote(s):
    if s[0] in ('"', "'"):
        return s
    if s.startswith('t.'):
        s = s.split('.')[1]
    if s[0].islower() or s in _no_quote:
        return s
    s = s.split('.')[-1]
    return repr(s)
