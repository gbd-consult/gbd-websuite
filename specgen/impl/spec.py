"""Generate type spec objects from parsed units"""


def for_types(units, type_names, flatten=True):
    g = _TypeGenerator(units, flatten)
    for tname in type_names:
        g.run(tname)
    return g.specs


def for_methods(units):
    specs = {}

    for u in units:
        if u.kind == 'method':
            m = _method_spec(units, u)
            if m:
                specs[m.cmd] = m

    return specs


def for_cli_functions(units):
    specs = {}

    for u in units:
        if u.kind == 'clifunc':
            tname = u.command + '.' + u.name
            specs[tname] = CliFunctionSpec(
                args=u.args,
                command=u.command,
                doc=u.doc,
                module=u.module,
                name=tname,
                subcommand=u.name,
            )

    return specs


##

from .base import *


class _TypeGenerator:
    def __init__(self, units, flatten):
        self.open = set()
        self.specs = {}
        self.units = units
        self.flatten = flatten

    def run(self, tname):
        self.gen(tname)
        return self.specs

    def gen(self, tname):
        if tname in self.specs or tname in self.open:
            return tname

        u = self.find(tname)
        if not u:
            return tname

        self.open.add(tname)
        r = self.item(u)
        self.open.remove(tname)

        if not r:
            return tname

        # if r.type == 'alias':
        #     return r.target

        r.name = tname
        r.doc = u.doc
        self.specs[tname] = r

        return tname

    def find(self, tname):
        for u in self.units:
            if u.name == tname:
                return u

    def item(self, u):
        if u.kind == 'class':
            return self.klass(u)

        if u.kind == 'any':
            return TypeSpec(
                type='any',
            )

        if u.kind == 'union':
            return TypeSpec(
                type='union',
                bases=[self.gen(t) for t in u.bases]
            )

        if u.kind == 'typeunion':
            return TypeSpec(
                type='typeunion',
                bases=[self.gen(t) for t in u.bases]
            )

        if u.kind == 'list':
            return TypeSpec(
                type='list',
                bases=[self.gen(u.bases[0])]
            )

        if u.kind == 'dict':
            return TypeSpec(
                type='dict',
                bases=[self.gen(t) for t in u.bases]
            )

        if u.kind == 'tuple':
            return TypeSpec(
                type='tuple',
                bases=[self.gen(t) for t in u.bases]
            )

        if u.kind == 'enum':
            return TypeSpec(
                type='enum',
                values=u.values,
            )

        if u.kind == 'alias':
            return TypeSpec(
                type='alias',
                target=self.gen(u.type)
            )

    def klass(self, u):
        props = []

        for p in self.units:
            if p.kind == 'prop' and p.parent == u.uid:
                props.append(PropertySpec(
                    name=p.name,
                    doc=p.doc,
                    type=self.gen(p.type),
                    optional=p.optional,
                    default=p.default,
                ))

        extends = []

        # generate superclasses
        for sname in u.supers:
            self.gen(sname)

            # no spec generated - skip
            if sname not in self.specs:
                continue

            # don't flatten - add spec to the extends list
            if not self.flatten:
                extends.append(sname)
                continue

            # flatten - add super props to our props
            own_names = set(p.name for p in props)
            sup = self.specs[sname]
            props.extend(p for p in sup.props if p.name not in own_names)

        return ObjectSpec(
            props=sorted(props, key=lambda p: p.name),
            extends=extends
        )


def _method_spec(units, u):
    cat = _method_category(u)
    if not cat:
        return

    m = MethodSpec(
        name=u.name,
        module=u.module,
        doc=u.doc,
        category=cat,
        action=u.module.split('.')[-1],
    )

    args = []

    for c in units:
        if c.kind == 'argument' and c.parent == u.uid:
            args.append(c.type)
        if c.kind == 'return' and c.type != 'void' and c.parent == u.uid:
            m.ret = c.type

    # we only check the _last_ argument to each method, which is the payload
    if args and args[-1] != 'void':
        m.arg = args[-1]

    # convert module/method names to public command identifiers
    # map.api_render_xyz => mapRenderXyz
    # map.http_get_xyz => mapHttpGetXyz

    f = m.name.split('_')
    if cat == 'api':
        f.pop(0)
    m.cmd = m.action + ''.join(s.title() for s in f)

    return m


def _method_category(u):
    if u.name.startswith('api'):
        return 'api'
    if u.name.startswith('http_get'):
        return 'http_get'
    if u.name.startswith('http_post'):
        return 'http_post'
    if u.name.startswith('http'):
        return 'http'
