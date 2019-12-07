"""Perform certain normalizations on the parse's results"""


def normalize(units):
    _check_optional(units)
    _type_unions_from_exts(units)

    _list_kinds(units)
    _enum_kinds(units)
    _union_kinds(units)
    _tuple_kinds(units)

    _clean_types(units)
    _clean_global_refs(units)
    _clean_supers(units)

    return units


##

from .base import Unit


def _check_optional(units):
    # eliminate Optional types and set 'optional' flags respectively

    for u in units:
        if u.types and u.types[0].endswith('Optional'):
            u.optional = True
            u.types = u.types[1:]
        else:
            u.optional = u.default is not None


def _type_unions_from_exts(units):
    # create "type"-based discriminated unions from gws.ext... types

    ls = {}

    for u in units:
        if u.kind == 'class' and u.name.startswith('gws.ext'):
            # gws.ext.layer.vector.Config -> gws.types.ext.layer.Config
            t = u.name.split('.')
            union_name = 'gws.types.' + '.'.join(t[1:-2] + [t[-1]])
            if union_name not in ls:
                ls[union_name] = set()
            ls[union_name].add(u.name)

    for union_name, types in ls.items():
        units.append(Unit(
            kind='typeunion',
            name=union_name,
            bases=sorted(types)
        ))


def _list_kinds(units):
    # create explicit 'list' kinds from List types: List[Foo] => FooList

    ls = {}

    for u in units:
        if u.types and u.types[0].endswith('List'):
            # @TODO list of lists
            _, base = u.types
            tname = base + 'List'
            u.types = [tname]
            ls[tname] = Unit(
                kind='list',
                name=tname,
                bases=[base]
            )

    units.extend(ls.values())


def _enum_kinds(units):
    # replace 'Enum' descendants with 'enum' kinds

    def is_enum(u):
        return any(s.endswith('Enum') for s in u.supers)

    def get_values(parent_uid):
        return {
            u.name: u.default
            for u in units
            if u.kind == 'valueprop' and u.parent == parent_uid
        }

    for u in units:
        if u.kind == 'class' and is_enum(u):
            u.kind = 'enum'
            u.values = get_values(u.uid)


def _union_kinds(units):
    # create explicit 'union' kinds for Union types

    ls = {}

    for u in units:
        if u.types and u.types[0].endswith('Union'):
            # u.types is like [Union, tuple, [base, base...]]
            bases = sorted(u.types[2])
            tname = _tuple_name(bases)
            ls[tname] = Unit(
                kind='union',
                name=tname,
                bases=bases
            )
            u.types = [tname]

    units.extend(ls.values())


def _tuple_kinds(units):
    # replace Tuple types with 'tuple' kinds

    for u in units:
        if u.types and u.types[0].lower().endswith('tuple'):
            # u.types is [tuple, [base, base...]]
            # or ['gws.types.Tuple', 'tuple', [base, base...]]

            bases = u.types[1]
            if bases == 'tuple':
                bases = u.types[2]

            u.kind = 'tuple'
            u.bases = bases
            u.types = []


def _clean_types(units):
    # ensure all u.types are reduced to a single member

    for u in units:
        if len(u.types) > 1:
            raise ValueError('mixed type list', vars(u))
        u.type = u.types[0] if u.types else ''


def _clean_global_refs(units):
    # remove 'core' part from global units

    def clean(s):
        x = s
        if '.' not in s:
            return s
        s = [a for a in s.split('.') if a != 'core']
        return '.'.join(s)

    for u in units:
        u.name = clean(u.name)
        u.supers = [clean(s) for s in u.supers]
        u.bases = [clean(s) for s in u.bases]
        u.type = clean(u.type)


def _clean_supers(units):
    # remove unparsed superclasses

    names = set(u.name for u in units)

    for u in units:
        u.supers = [s for s in u.supers if s in names]


def _tuple_name(bases):
    if len(set(bases)) == 1:
        return '%s%d' % (bases[0], len(bases))
    return '_or_'.join(bases).replace('.', '_')
