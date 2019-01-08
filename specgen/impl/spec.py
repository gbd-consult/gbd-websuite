"""Generate spec objects from parse's results"""


def generate(objects, types, flatten=True):
    g = _Generator(objects, flatten)
    for tname in types:
        g.run(tname)
    return g.res



class _Generator:
    def __init__(self, objects, flatten):
        self.open = set()
        self.res = {}
        self.objects = objects
        self.flatten = flatten

    def run(self, tname):
        self.gen(tname)
        return self.res

    def gen(self, tname):
        if tname in self.res or tname in self.open:
            return tname

        obj = self.find(tname)
        if not obj:
            return tname

        self.open.add(tname)
        r = self.item(obj)
        self.open.remove(tname)

        if r:
            if r['type'] == 'alias':
                tname = r['target']
            else:
                r['name'] = tname
                r['doc'] = obj.get('doc', '')
                self.res[tname] = r

        return tname

    def find(self, tname):
        for obj in self.objects:
            if obj['name'] == tname:
                return obj

    def item(self, obj):
        if obj['kind'] == 'object':
            return self.klass(obj)

        if obj['kind'] == 'union':
            return {
                'type': 'union',
                'bases': [self.gen(t) for t in obj['bases']]
            }

        if obj['kind'] == 'list':
            return {
                'type': 'list',
                'base': self.gen(obj['bases'][0])
            }

        if obj['kind'] == 'tuple':
            return {
                'type': 'tuple',
                'bases': obj['bases'],
            }

        if obj['kind'] == 'enum':
            return {
                'type': 'enum',
                'values': obj['values'],
            }

        if obj['kind'] == 'alias':
            return {
                'type': 'alias',
                'target': self.gen(obj['type'])
            }

    def klass(self, obj):
        props = []

        for m in self.objects:
            if m['kind'] == 'prop' and m['parent_uid'] == obj['uid']:
                props.append({
                    'name': m['name'],
                    'doc': m.get('doc', ''),
                    'type': self.gen(m['type']),
                    'optional': m['optional'],
                    'default': m['default'],
                })

        r = {
            'type': 'object',
        }

        extends = []

        for sname in obj.get('supers', []):
            self.gen(sname)
            if sname in self.res:
                if not self.flatten:
                    extends.append(sname)
                else:
                    names = set(p['name'] for p in props)
                    for p in self.res[sname]['props']:
                        if p['name'] not in names:
                            props.append(p)

        if extends:
            r['extends'] = extends

        r['props'] = sorted(props, key=lambda m: m['name'])
        return r


