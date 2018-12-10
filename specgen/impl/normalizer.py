"""Perform certain normalizations on the parse's results"""

_builtins = 'str', 'int', 'float', 'bool', 'list', 'dict'


def _simplify(s):
    if '.' not in s:
        return s
    s = [a for a in s.split('.') if a not in ('core',)]
    return '.'.join(s)


def _exts(objects):
    ls = {}

    for c in objects:
        if c['kind'] == 'object' and c['name'].startswith('gws.ext'):
            # gws.ext.gis.layer.vector.Config -> gws.types.ext.gis.layer.Config
            t = c['name'].split('.')
            u = 'gws.types.' + '.'.join(t[1:-2] + [t[-1]])
            if u not in ls:
                ls[u] = set()
            ls[u].add(c['name'])

    for u, m in ls.items():
        objects.append({
            'kind': 'union',
            'name': u,
            'bases': sorted(list(m))
        })


def _optionals(objects):
    for c in objects:
        if 'type' in c:
            if c['type'][0].endswith('Optional'):
                c['optional'] = True
                c['type'] = c['type'][1:]
            else:
                c['optional'] = c.get('default') is not None


def _lists(objects):
    ls = {}

    for c in objects:
        if 'type' in c:
            if c['type'][0].endswith('List'):
                # @TODO list of lists
                _, base = c['type']
                tname = base + 'List'
                ls[tname] = {
                    'kind': 'list',
                    'name': tname,
                    'bases': [base]
                }
                c['type'] = tname

    objects.extend(ls.values())


def _tuple_name(bases):
    if len(set(bases)) == 1:
        return '%s%d' % (bases[0], len(bases))
    return '_'.join(bases).replace('.', '')


def _unions(objects):
    ls = {}

    for c in objects:
        if 'type' in c:
            if c['type'][0].endswith('Union'):
                # [Union, tuple, [base, base...]]
                bases = c['type'][2]
                tname = _tuple_name(bases)
                ls[tname] = {
                    'kind': 'union',
                    'name': tname,
                    'bases': bases
                }
                c['type'] = tname

    objects.extend(ls.values())


def _tuples(objects):
    ls = {}

    for c in objects:
        if 'type' in c:
            if c['type'][0].lower().endswith('tuple'):
                # [tuple, [base, base...]]
                # or ['gws.types.Tuple', 'tuple', [base, base...]]

                bases = c['type'][1]
                if bases == 'tuple':
                    bases = c['type'][2]

                tname = _tuple_name(bases)
                ls[tname] = {
                    'kind': 'tuple',
                    'name': tname,
                    'bases': bases,
                }
                c['type'] = tname

    objects.extend(ls.values())


def _is_enum(c):
    return any(s.endswith('Enum') for s in c.get('supers', []))


def _enums(objects):
    for c in objects:
        if c['kind'] == 'object' and _is_enum(c):
            values = {}
            for p in objects:
                if p['kind'] == 'assign_prop' and p['parent_uid'] == c['uid']:
                    values[p['name']] = p['default']
            c['kind'] = 'enum'
            c['values'] = values


def _types(objects):
    for c in objects:
        if c.get('type'):
            if isinstance(c['type'], list):
                if len(c['type']) == 1:
                    c['type'] = c['type'][0]
                else:
                    raise ValueError('mixed type list', c)


def _is_object(objects, name):
    # return any(p['name'] == name for p in objects)
    return any(_simplify(p['name']) == name for p in objects)


def _supers(objects):
    for c in objects:
        su = []
        for s in c.pop('supers', []):
            if _is_object(objects, s):
                su.append(s)
            if su:
                c['supers'] = su


def normalize(objects):
    _exts(objects)
    _optionals(objects)
    _lists(objects)
    _enums(objects)
    _unions(objects)
    _tuples(objects)
    _supers(objects)
    _types(objects)

    for c in objects:
        c['name'] = _simplify(c['name'])
        if c.get('supers'):
            c['supers'] = [_simplify(s) for s in c['supers']]
        if c.get('type'):
            c['type'] = _simplify(c['type'])
        if c.get('bases'):
            c['bases'] = [_simplify(b) for b in c['bases']]
