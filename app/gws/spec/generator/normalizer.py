from . import base


def normalize(specs, options):
    _add_core_aliases(specs)
    _check_enums(specs)
    _check_variants(specs)
    _create_config_and_props(specs)
    _create_ext_variants(specs)
    _create_commands_variant(specs)
    _make_prop_lists(specs)

    return specs


_SERVER_ROOTS = ['gws.base.application.Config', 'gws.ext.Command', 'gws.ext.Object']


def prepare_for_server(specs, options):
    obj_desc = _ext_objects_descriptors(specs)
    cmd_desc = _ext_commands_descriptors(specs)
    specs = _select_dependent(specs, _SERVER_ROOTS)
    specs.update(obj_desc)
    specs.update(cmd_desc)
    return specs


##

def _select_dependent(specs, bases):
    queue = list(bases)
    selected = {}

    while queue:
        s = queue.pop(0)
        if not s:
            continue

        if isinstance(s, list):
            if s[0] == base.T.list:
                queue.append(s[1])
            elif s[0] in {base.T.tuple, base.T.dict, base.T.union}:
                queue.extend(s[1])
            elif s[0] == base.T.variant:
                queue.extend(s[1].values())
            continue

        if s in selected or s in base.BUILTINS:
            continue

        if s not in specs:
            raise base.Error(f'type not found: {s!r}')

        spec = specs[s]
        selected[s] = spec

        queue.append(spec.type)
        queue.append(spec.arg)
        queue.append(spec.ret)
        queue.append(spec.super)
        queue.append(spec.target)

        if spec.props:
            queue.extend(spec.props.values())

    return selected


##


def _get_with_alias(specs, name):
    s = specs.get(name)
    if s and s.abc == base.ABC.alias and isinstance(s.target, str):
        return _get_with_alias(specs, s.target)
    return s




def _add_core_aliases(specs):
    ext = {}
    for s in specs.values():
        for g in base.GLOBAL_MODULES:
            if s.name.startswith(g):
                alias = 'gws.' + s.name[len(g):]
                ext[alias] = base.Data(
                    abc=base.ABC.alias,
                    name=alias,
                    target=s.name,
                )
                break

    specs.update(ext)


def _check_enums(specs):
    for s in specs.values():
        # @TODO enums can appear in other contexts as well
        if s.default and isinstance(s.default, list) and s.default[0] == base.T.unchecked_enum:
            parts = s.default[1].split('.')
            key = parts.pop()
            spec = _get_with_alias(specs, '.'.join(parts))
            if spec and spec.abc == base.ABC.enum and key in spec.values:
                s.default = spec.values[key]
            else:
                base.warn('invalid value', s.default)
                s.default = None
                s.has_default = False


def _check_variant(specs, names):
    variants = {}
    for name in names:
        try:
            obj_spec = specs[name]
            type_prop_spec = specs[obj_spec.name + '.type']
            variants[type_prop_spec.type[1][0]] = name
        except:
            raise base.Error(f'{name!r}: invalid Variant member')
    return [base.T.variant, variants]


def _check_variants(specs):
    for spec in specs.values():
        for p in 'arg', 'ret', 'target', 'type':
            t = getattr(spec, p)
            if isinstance(t, list) and t[0] == base.T.unchecked_variant:
                setattr(spec, p, _check_variant(specs, t[1]))


#


def _create_config_and_props(specs):
    ext = {}

    ext_names = set(s.name for s in specs.values() if s.ext_kind)

    for s in specs.values():
        if s.ext_kind != 'Object':
            continue

        ext_name = base.GWS_EXT_PREFIX + s.ext_category + '.' + s.ext_type + '.Config'
        if ext_name not in ext_names:
            ext[ext_name] = base.Data(
                abc=base.ABC.object,
                doc=s.doc,
                ext_category=s.ext_category,
                ext_kind='Config',
                ext_type=s.ext_type,
                ident='Config',
                name=ext_name,
                super='gws.WithAccess',
            )

        ext_name = base.GWS_EXT_PREFIX + s.ext_category + '.' + s.ext_type + '.Props'
        if ext_name not in ext_names:
            ext[ext_name] = base.Data(
                abc=base.ABC.object,
                doc=s.doc,
                ext_category=s.ext_category,
                ext_kind='Props',
                ext_name=ext_name,
                ext_type=s.ext_type,
                ident='Props',
                name=ext_name,
                super='gws.Props',
            )

    specs.update(ext)


def _create_ext_variants(specs):
    ext = {}

    for s in specs.values():
        if s.abc != base.ABC.object or not s.ext_type:
            continue

        # gws.ext.db.provider.foo.Object => gws.ext.db.provider.Object
        variant_name = base.GWS_EXT_PREFIX + s.ext_category + '.' + s.ext_kind

        if variant_name not in ext:
            ext[variant_name] = base.Data(
                abc=base.ABC.alias,
                name=variant_name,
                target=[base.T.variant, {}]
            )
        ext[variant_name].target[1][s.ext_type] = s.name

        # synthetize the 'type' property
        prop_key = s.name + '.' + 'type'
        ext[prop_key] = base.Data(
            abc=base.ABC.property,
            default=None,
            doc='object type',
            has_default=False,
            ident='type',
            name=prop_key,
            owner=s.name,
            type=[base.T.literal, [s.ext_type]],
        )

        # gws.ext.db.provider.foo.Object => gws.ext.Object
        top_name = base.GWS_EXT_PREFIX + s.ext_kind

        if top_name not in ext:
            ext[top_name] = base.Data(
                abc=base.ABC.alias,
                name=top_name,
                target=[base.T.union, []]
            )
        ext[top_name].target[1].append(variant_name)

    specs.update(ext)


def _create_commands_variant(specs):
    ls = [k for k, s in specs.items() if s.abc == base.ABC.command]
    key = 'gws.ext.Command'
    specs[key] = base.Data(abc=base.ABC.alias, name=key, target=[base.T.union, ls])


def _ext_objects_descriptors(specs):
    desc = {}

    for s in specs.values():
        if s.abc == base.ABC.object and s.ext_kind == 'Object':
            mod = specs[s.module]
            desc[s.name] = base.Data(
                abc=base.ABC.descriptor,
                ext_kind=s.ext_kind,
                ext_type=s.ext_type,
                ident=s.ident,
                name=s.name,
                module_name=mod.name,
                module_path=mod.path,
            )

    return desc


def _ext_commands_descriptors(specs):
    desc = {}

    for s in specs.values():
        if s.abc == base.ABC.command:
            owner = specs[s.owner]
            desc[s.name] = base.Data(
                abc=base.ABC.command,
                action_type=owner.ext_type,
                function_name=s.ident,
                arg=s.arg,
                class_name=s.owner,
            )

    return desc


#

def _own_props(specs, spec):
    prefix = spec.name + '.'
    return [s for s in specs.values() if s.name.startswith(prefix) and s.abc == base.ABC.property]


def _make_prop_lists(specs):
    done = set()

    def _make_props(spec, stack):
        if not spec:
            return {}
        if spec.abc == base.ABC.alias and isinstance(spec.target, str):
            return _make_props(specs.get(spec.target), stack)
        if spec.abc != base.ABC.object:
            return {}
        if spec.name in done:
            return spec.props
        if spec.name in stack:
            raise base.Error(f'inheritance cycle in {spec.name!r}')
        if spec.super in specs:
            props = dict(_make_props(specs[spec.super], stack + [spec.name]))
        else:
            props = {}
        for p in _own_props(specs, spec):
            props[p.ident] = p.name
        spec.props = props
        done.add(spec.name)
        return spec.props

    for s in specs.values():
        _make_props(s, [])
