"""Extract PROPS for types.py"""

import gid6 as gid

type_map = {
    'CharacterString': 'str',
    'Boolean': 'bool',
    'Integer': 'int',
    'Length': 'int',
    'Area': 'float',
    'Volume': 'float',
    'DateTime': 'str',
    'Date': 'str',
}

md = gid.METADATA

all_titles = {}

def get_props(topics):
    ptypes = {}
    titles = {}

    for v in md.values():
        if v['kind'] == 'object' and any(t in v['key'] for t in topics):
            for a in v['attributes']:
                n = a['name']
                t = a['type']

                if t in md and md[t].get('kind') == 'enum':
                    t = 'EnumPair'
                elif t in type_map:
                    t = type_map[t]
                else:
                    # ignore object types for now
                    # print('ignored type', n, t)
                    t = ''

                if t and a['list'] == 1:
                    t = 'list[' + t + ']'

                ptypes.setdefault(n, []).append(t)
                titles.setdefault(n, []).append(a['title'])

    decls = []

    for n in ptypes:
        ptype = set(ptypes[n])
        title = set(titles[n])

        if len(ptype) > 1 or len(title) > 1 or '' in ptype or '' in title:
            # print('mixed key', k, ptype, title)
            continue

        all_titles[n] = title.pop()
        decls.append(f'    {n}: {ptype.pop()}')

    return sorted(decls)


py = get_props([
    'gebaeude/angaben_zum_gebaeude',
])

print('\n\nclass GebaeudeProps(Object):')
print('\n'.join(py))

py = get_props([
    'tatsaechliche_nutzung',
    'gesetzliche_festlegungen_gebietseinheiten_kataloge/bodenschaetzung_bewertung',
    'gesetzliche_festlegungen_gebietseinheiten_kataloge/oeffentlich_rechtliche_und_sonstige_festlegungen',
])

print('\n\nclass PartProps(Object):')
print('\n'.join(py))

print('\n\nPROPS = {')
for k in sorted(all_titles):
    print(f'    {k!r}: {all_titles[k]!r},')
print('}')
