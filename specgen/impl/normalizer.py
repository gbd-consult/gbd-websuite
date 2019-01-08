"""Perform certain normalizations on the parse's results"""


def normalize(objects):
    _check_optional(objects)

    _unions_from_exts(objects)

    _list_kinds(objects)
    _enum_kinds(objects)
    _union_kinds(objects)
    _tuple_kinds(objects)

    _clean_types(objects)
    _clean_global_refs(objects)
    _clean_supers(objects)

    return objects


##


_builtins = 'str', 'int', 'float', 'bool', 'list', 'dict'


def _check_optional(objects):
    # eliminate Optional types and set 'optional' flags respectively

    for c in objects:
        if 'type' in c:
            if c['type'][0].endswith('Optional'):
                c['optional'] = True
                c['type'] = c['type'][1:]
            else:
                c['optional'] = c.get('default') is not None


def _unions_from_exts(objects):
    # join different gws.ext types into unions

    ls = {}

    for c in objects:
        if c['kind'] == 'object' and c['name'].startswith('gws.ext'):
            # gws.ext.gis.layer.vector.Config -> gws.types.ext.gis.layer.Config
            t = c['name'].split('.')
            tname = 'gws.types.' + '.'.join(t[1:-2] + [t[-1]])
            if tname not in ls:
                ls[tname] = set()
            ls[tname].add(c['name'])

    for tname, typenames in ls.items():
        objects.append({
            'kind': 'union',
            'name': tname,
            'bases': sorted(list(typenames))
        })


def _list_kinds(objects):
    # create explicit 'list' kinds from List types: List[Foo] => FooList

    ls = {}

    for c in objects:
        if 'type' in c and c['type'][0].endswith('List'):
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


def _enum_kinds(objects):
    # replace 'Enum' descendants with 'enum' kinds

    def is_enum(c):
        return any(s.endswith('Enum') for s in c.get('supers', []))

    def get_values(parent_uid):
        values = {}
        for p in objects:
            if p['kind'] == 'assign_prop' and p['parent_uid'] == parent_uid:
                values[p['name']] = p['default']
        return values

    for c in objects:
        if c['kind'] == 'object' and is_enum(c):
            c['kind'] = 'enum'
            c['values'] = get_values(c['uid'])


def _union_kinds(objects):
    # create explicit 'union' kinds for Union types

    ls = {}

    for c in objects:
        if 'type' in c and c['type'][0].endswith('Union'):
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


def _tuple_kinds(objects):
    # replace Tuple types with 'tuple' kinds

    ls = {}

    for c in objects:
        if 'type' in c and c['type'][0].lower().endswith('tuple'):
            # [tuple, [base, base...]]
            # or ['gws.types.Tuple', 'tuple', [base, base...]]

            bases = c['type'][1]
            if bases == 'tuple':
                bases = c['type'][2]

            c['kind'] = 'tuple'
            c['bases'] = bases
            del c['type']

    objects.extend(ls.values())


def _clean_supers(objects):
    # remove unparsed superclasses

    def is_object(name):
        return any(p['name'] == name for p in objects)

    for c in objects:
        ls = []
        for s in c.pop('supers', []):
            if is_object(s):
                ls.append(s)
        if ls:
            c['supers'] = ls


def _clean_types(objects):
    # ensure there are no multi-types anymore

    for c in objects:
        if 'type' in c and isinstance(c['type'], list):
            if len(c['type']) == 1:
                c['type'] = c['type'][0]
            else:
                raise ValueError('mixed type list', c)


def _clean_global_refs(objects):
    # remove 'core' part from global objects

    def clean(s):
        if '.' not in s:
            return s
        s = [a for a in s.split('.') if a not in ('core',)]
        return '.'.join(s)

    for c in objects:
        c['name'] = clean(c['name'])
        if c.get('supers'):
            c['supers'] = [clean(s) for s in c['supers']]
        if c.get('type'):
            c['type'] = clean(c['type'])
        if c.get('bases'):
            c['bases'] = [clean(b) for b in c['bases']]


def _tuple_name(bases):
    if len(set(bases)) == 1:
        return '%s%d' % (bases[0], len(bases))
    return '_'.join(bases).replace('.', '')
