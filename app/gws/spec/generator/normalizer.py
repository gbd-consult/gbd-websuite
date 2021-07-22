import re
from typing import cast

from . import base


def normalize(state: base.ParserState, meta):
    _add_system_aliases(state)
    _check_enums(state)
    _check_variants(state)
    _synthesize_ext_configs_and_props(state)
    _synthesize_ext_type_props(state)
    _synthesize_ext_variants(state)
    _resolve_aliases(state)
    _make_props(state)


def prepare_for_server(state: base.ParserState, meta):
    objs = {}
    cmds = {}

    deps = {'gws.base.application.Config'}

    for t in state.types.values():
        if isinstance(t, base.TObject) and t.ext_kind == 'Object':
            objs[t.name] = dict(
                _='ExtObject',
                ext_category=t.ext_category,
                ext_type=t.ext_type,
                ident=t.ident,
                module_name=t.pos['module_name'],
                module_path=t.pos['module_path'],
                name=t.name,
            )

        if isinstance(t, base.TCommand):
            owner_type = cast(base.TObject, _get_type(state, t.owner_t))
            cmds[t.name] = dict(
                _='ExtCommand',
                arg=t.arg_t,
                class_name=owner_type.name,
                cmd_action=t.cmd_action,
                cmd_command=t.cmd_command,
                cmd_method=t.cmd_method,
                cmd_name=t.cmd_name,
                function_name=t.ident,
                name=t.name,
            )
            deps.add(t.arg_t)

    specs = _select_dependent(state, deps)
    specs.update(objs)
    specs.update(cmds)
    return specs


def _select_dependent(state, names):
    queue = [['root', n] for n in names]
    selected = {}

    while queue:
        parent, q = queue.pop(0)
        t = _get_type(state, q) if isinstance(q, str) else cast(base.Type, q)

        if not t:
            raise base.Error(f'unresolved reference {q!r} in {parent!r}')

        if t.name in selected:
            continue

        if isinstance(t, base.TAtom):
            continue

        nxt = []

        if isinstance(t, base.TDict):
            nxt = [t.key_t, t.value_t]
        if isinstance(t, base.TList):
            nxt = [t.item_t]
        if isinstance(t, base.TOptional):
            nxt = t.target_t
        if isinstance(t, (base.TTuple, base.TUnion)):
            nxt = t.items
        if isinstance(t, base.TVariant):
            nxt = t.members.values()

        if isinstance(t, base.TObject):
            selected[t.name] = dict(name=t.name, props=t.props)
            nxt = t.props.values()

        if isinstance(t, base.TProperty):
            selected[t.name] = dict(
                default=t.default, has_default=t.has_default, name=t.name, ident=t.ident, property_t=t.property_t)
            nxt = [t.property_t]

        if isinstance(t, base.TAlias):
            selected[t.name] = dict(target_t=t.target_t)
            nxt = [t.target_t]

        if isinstance(t, base.TEnum):
            selected[t.name] = dict(values=t.values)

        for it in nxt:
            queue.append([t.name, it])

        selected.setdefault(t.name, dict(vars(t)))
        selected[t.name]['_'] = type(t).__name__

    return selected


##


def _add_system_aliases(state):
    # alias global short names (e.g. `gws.Crs`) to long names (`gws.core.types.Crs`)

    for t in state.types.values():
        if t.name.startswith('gws.core.'):
            alias = re.sub(r'^[a-z.]+', '', t.name)
            if alias:
                state.aliases['gws.' + alias] = t.name


def _resolve_aliases(state):
    resolved = {}
    newtypes = {}
    stack = []

    def _visit(t):
        if isinstance(t, (base.TAtom, base.TLiteral, base.TEnum)):
            return t

        if isinstance(t, base.TDict):
            return base.TDict(key_t=_resolve(t.key_t), value_t=_resolve(t.value_t))

        if isinstance(t, base.TList):
            return base.TList(item_t=_resolve(t.item_t))

        if isinstance(t, base.TOptional):
            return base.TOptional(target_t=_resolve(t.target_t))

        if isinstance(t, base.TTuple):
            return base.TTuple(items=[_resolve(it) for it in t.items])

        if isinstance(t, base.TUnion):
            return base.TUnion(items=[_resolve(it) for it in t.items])

        if isinstance(t, base.TVariant):
            return base.TVariant(members={k: _resolve(v) for k, v in t.members.items()})

        if isinstance(t, base.TCommand):
            t.arg_t = _resolve(t.arg_t)
            t.ret_t = _resolve(t.ret_t)
            t.owner_t = _resolve(t.owner_t)
            return t

        if isinstance(t, base.TObject):
            if t.supers:
                t.supers = [_resolve(s) for s in t.supers]
            return t

        if isinstance(t, base.TProperty):
            t.owner_t = _resolve(t.owner_t)
            t.property_t = _resolve(t.property_t)
            return t

        if isinstance(t, base.TAlias):
            t.target_t = _resolve(t.target_t)
            return t

    def _resolve(name):
        if not name:
            return name
        if name in resolved:
            return resolved[name]
        if name in stack:
            raise ValueError(f'circular reference {stack!r}->{name!r}')

        t = _get_type(state, name)

        if not t:
            base.log.debug(f'unbound reference {name!r}')
            return name

        if isinstance(t, base.TUnresolvedReference):
            base.log.debug(f'unresolved reference {name!r}')
            return name

        stack.append(t.name)
        t2 = _visit(t)
        stack.pop()

        resolved[name] = t2.name
        newtypes[t2.name] = t2
        return t2.name

    for name in state.types:
        _resolve(name)

    state.types = newtypes


def _check_enums(state):
    for t in state.types.values():
        default = getattr(t, 'default', None)
        if isinstance(default, list) and len(default) == 2 and default[0] == base.UNCHECKED_ENUM:
            name = default[1]
            obj_name, _, item_name = name.rpartition('.')
            enum_spec = _get_type(state, obj_name)
            if enum_spec and isinstance(enum_spec, base.TEnum) and item_name in enum_spec.values:
                t.default = enum_spec.values[item_name]
            else:
                t.default = None
                t.has_default = False
                base.log.debug(f'invalid enum value {name!r}')


def _check_variants(state):
    upd = {}

    for t in state.types.values():
        if not isinstance(t, base.TVariantStub):
            continue

        members = {}

        for item_name in t.items:
            try:
                item_type = _get_type(state, item_name)
                tag_prop = _get_type(state, item_type.name + '.' + base.GWS_TAG_PROPERTY)
                tag_prop_type = _get_type(state, tag_prop.property_t)
                if isinstance(tag_prop_type, base.TLiteral) and len(tag_prop_type.values) == 1:
                    members[tag_prop_type.values[0]] = item_name
                else:
                    raise ValueError()
            except Exception:
                raise base.Error(f'invalid Variant: {t.pos!r}')

        upd[t.name] = base.TVariant(members=members)

    state.types.update(upd)


_synthesized_pos = {
    'lineno': 0,
    'module_name': '<synthesized>',
    'module_path': '',
}


def _synthesize_ext_configs_and_props(state):
    upd = {}

    existing_names = set(t.name for t in state.types.values() if hasattr(t, 'ext_kind'))

    super_types = {
        'Config': _get_type(state, 'gws.WithAccess'),
        'Props': _get_type(state, 'gws.Props'),
    }

    for t in state.types.values():
        if getattr(t, 'ext_kind', None) != 'Object':
            continue

        for kind in ['Config', 'Props']:
            name = _dot(base.GWS_EXT_PREFIX, t.ext_category, t.ext_type, kind)
            if name in existing_names:
                continue
            upd[name] = base.TObject(
                doc=t.doc,
                ident=kind,
                name=name,
                pos=_synthesized_pos,
                supers=[super_types[kind].name],
                ext_category=t.ext_category,
                ext_kind=kind,
                ext_type=t.ext_type,
            )
            # base.log.debug(f'synthesized {name!r}')

    state.types.update(upd)


def _synthesize_ext_type_props(state):
    upd = {}

    for t in state.types.values():
        if getattr(t, 'ext_kind', None) not in ('Config', 'Props'):
            continue

        literal = base.TLiteral(values=[t.ext_type])
        upd[literal.name] = literal

        name = _dot(t.name, base.GWS_TAG_PROPERTY)
        upd[name] = base.TProperty(
            doc='',
            ident=t.ext_type,
            name=name,
            pos=_synthesized_pos,
            default=None,
            has_default=False,
            property_t=literal.name,
            owner_t=t.name,
        )

    state.types.update(upd)


def _synthesize_ext_variants(state):
    variants = {}

    for t in state.types.values():
        if not isinstance(t, base.TObject) or not getattr(t, 'ext_type', None):
            continue
        # gws.ext.db.provider.foo.Object => gws.ext.db.provider.Object
        variant_name = _dot(base.GWS_EXT_PREFIX, t.ext_category, t.ext_kind)
        variants.setdefault(variant_name, {})[t.ext_type] = t.name

    upd = {}

    for name, members in variants.items():
        variant_type = base.TVariant(members=members)
        upd[variant_type.name] = variant_type

        alias_type = base.TAlias(
            doc='',
            ident=name.split('.')[-1],
            name=name,
            pos=_synthesized_pos,
            target_t=variant_type.name,
        )
        upd[alias_type.name] = alias_type

    state.types.update(upd)


def _make_props(state):
    done = {}
    own_props = {}
    stack = []

    for t in state.types.values():
        if isinstance(t, base.TProperty):
            obj_name, _, prop_name = t.name.rpartition('.')
            own_props.setdefault(obj_name, {})[prop_name] = t.name

    def _make(t):
        name = t.name

        if name in done:
            return done[name]
        if name in stack:
            raise base.Error(f'circular inheritance {stack!r}->{name!r}')

        props = {}

        if isinstance(t, base.TObject):
            for sup in t.supers:
                super_type = _get_type(state, sup)
                if super_type:
                    stack.append(name)
                    props.update(_make(super_type))
                    stack.pop()
                else:
                    base.log.debug(f'unknown supertype {sup!r}')
            if name in own_props:
                props.update(own_props[name])
            t.props = props

        done[name] = props
        return props

    for t in state.types.values():
        if isinstance(t, base.TObject):
            _make(t)


def _get_type(state, name):
    while name in state.aliases:
        a = state.aliases.get(name)
        if not a or a == name:
            break
        name = a
    return state.types.get(name)


def _dot(*args):
    return '.'.join(args)
