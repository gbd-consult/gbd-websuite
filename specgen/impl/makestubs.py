"""Create stub sources from parsed stubs."""

import re

_indent = ' ' * 4
_comma = ', '.join
_nl = '\n'.join
_nl2 = '\n\n'.join

_known_bases = set('Any Dict List Optional Tuple Union Data Config Props Attribute Enum'.split())


def run(stubs):
    classes = {s.class_name: s for s in stubs.values()}
    classes['gws.Object'] = stubs['IObject']
    classes['Object'] = stubs['IObject']

    for stub in stubs.values():
        stub.bases = [_format_base(b, classes) for b in stub.bases]

    for stub in stubs.values():
        stub.bases = [b for b in stub.bases if _check_base(b, stubs)]

    rs = _topsort(stubs)

    for stub in rs:
        _check_props(stub, stubs)

    return _nl2(_code(stub) for stub in rs)


def _format_base(b, classes):
    if b.startswith('gws.types.I'):
        return None
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


def _check_base(b, stubs):
    if not b:
        return False
    if b in stubs:
        return True
    return b in _known_bases


def _check_props(stub, stubs):
    for b in stub.bases:
        base = stubs.get(b)
        if base:
            stub.members = {k: v for k, v in stub.members.items() if k not in base.members}


def _topsort(stubs):
    ls = list(stubs.values())
    rs = []
    seen = set(_known_bases)

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
        if m['kind'] == 'prop':
            value = m.get('value') if 'Enum' in stub.bases else None
            props.append(_prop_code(m, value))
        if m['kind'] == 'method':
            methods.append(_method_code(m))

    for s in props:
        code.append(_indent + s)

    for s in methods:
        code.append(_indent + s)

    code = _nl(code) if code else (_indent + 'pass')
    return head + '\n' + code


# @TODO use the ast

def _prop_code(m, value):
    if 'type_name' in m:
        t = m['type_name']
    elif '->' in m['line']:
        t = _extract_return_annotation(m['line'])
    else:
        t = _extract_var_annotation(m['line'])
    return '%s: %s = %r' % (m['name'], t, value)


def _method_code(m):
    t = _format_function_decl(m['line'])
    return t + ' pass'


def _extract_var_annotation(line):
    line = line.split('#')[0].strip()
    m = re.match(r'^[\w.]+\s*:([^=]+)', line)
    if not m:
        raise ValueError('cannot parse %r' % line)
    return _quote_all(m.group(1).strip())


def _extract_return_annotation(line):
    line = line.split('#')[0].strip()
    m = re.search(r'->(.+?):', line)
    if not m:
        raise ValueError('cannot parse %r' % line)
    return _quote_all(m.group(1).strip())


def _format_function_decl(line):
    def _repl(m):
        return m.group(1) + _quote_all(m.group(2))

    line = line.split('#')[0].strip()
    line = re.sub(r'(:\s*)([^,=]+)', _repl, line)
    line = re.sub(r'(->\s*)([^:]+)', _repl, line)

    return line


def _quote_all(s):
    def _repl(m):
        return _quote(m.group(0))

    return re.sub(r"""(?x) '[\w.]+' | "[\w.]+" | [\w.]+""", _repl, s)


def _quote(s):
    if s[0] in ('"', "'"):
        return s
    if s.startswith('t.'):
        s = s.split('.')[1]
    if s[0].islower() or s in _known_bases:
        return s
    s = s.split('.')[-1]
    return repr(s)
