import re

from . import base


def normalize(gen: base.Generator):
    _add_global_aliases(gen)
    _expand_aliases(gen)
    _resolve_aliases(gen)
    _eval_expressions(gen)
    # _synthesize_ext_configs_and_props(gen)
    _synthesize_ext_variant_types(gen)
    _synthesize_ext_type_properties(gen)
    _check_undefined(gen)
    _make_props(gen)


##


def _add_global_aliases(gen: base.Generator):
    """Add globals aliases.

    If we have `mod.GlobalName` and `mod.some.module.GlobalName`, and `mod.some.module`
    is in `GLOBAL_MODULES`, the former should an alias for the latter.
    """

    for typ in gen.typeDict.values():
        if typ.name in gen.aliases:
            continue
        m = re.match(r'^gws\.([A-Z].*)$', typ.name)
        if not m:
            continue
        for mod in base.v.GLOBAL_MODULES:
            name = mod + DOT + m.group(1)
            if name in gen.typeDict:
                base.log.debug(f'global alias {typ.name!r} => {name!r}')
                gen.aliases[typ.name] = name
                break


def _expand_aliases(gen: base.Generator):
    """Expand aliases.

    Given t1 -> alias of t2, t2 -> alias of t3, establish t1 -> t3.
    """

    def _exp(target, stack):
        if target in gen.typeDict:
            return target
        if target in stack:
            raise base.GeneratorError(f'circular alias {stack!r} => {target!r}')
        if target in gen.aliases:
            return _exp(gen.aliases[target], stack + [target])
        if target.startswith(base.v.APP_NAME):
            base.log.warning(f'unbound alias {target!r}')
        return target

    new_aliases = {}
    for src, target in gen.aliases.items():
        new_target = _exp(target, [])
        if new_target != target:
            base.log.debug(f'alias expanded: {src!r} => {target!r} => {new_target!r}')
        new_aliases[src] = new_target
    gen.aliases = new_aliases


_type_scalars = [
    'tArg',
    'tItem',
    'tKey',
    'tValue',
    'tTarget',
    'tOwner',
    'tReturn',
]

_type_lists = [
    'tArgs',
    'tItems',
    'tSupers',
]


def _resolve_aliases(gen: base.Generator):
    """Replace references to aliases with their target type uids."""

    new_type_dict = {}

    def _rename_uid(uid):
        if uid in gen.aliases:
            new = gen.aliases[uid]
        else:
            new = COMMA.join(gen.aliases.get(s) or s for s in uid.split(COMMA))
        if new != uid:
            base.log.debug(f'resolved alias {uid!r} => {new!r}')
        return new

    for typ in gen.typeDict.values():
        if typ.uid in new_type_dict:
            continue

        if typ.uid in gen.aliases:
            base.log.debug(f'skip resolving {typ.uid} {typ.c}')
            continue

        dct = vars(typ)

        for f in _type_scalars:
            if f in dct:
                dct[f] = _rename_uid(dct[f])
        for f in _type_lists:
            if f in dct:
                dct[f] = [_rename_uid(s) for s in dct[f]]
        if not typ.name:
            typ.uid = _rename_uid(typ.uid)

        new_type_dict[typ.uid] = typ

    gen.typeDict = new_type_dict


def _eval_expressions(gen: base.Generator):
    """Replace enum and constant values with literal values"""

    def _get_type(name):
        if name in gen.aliases:
            name = gen.aliases[name]
        return gen.typeDict.get(name)

    def _eval(base_type, val):
        c, value = val
        if c == base.c.LITERAL:
            return value
        if isinstance(value, list):
            return [_eval(base_type, v) for v in value]

        if isinstance(value, dict):
            return {k: _eval(base_type, v) for k, v in value.items()}

        # constant?
        typ = _get_type(value)
        if typ and typ.c == base.c.CONSTANT:
            return typ.constValue

        # enum?
        obj_name, _, item = value.rpartition('.')
        typ = _get_type(obj_name)
        if typ and typ.c == base.c.ENUM and item in typ.enumValues:
            return typ.enumValues[item]

        base.log.warning(f'invalid expression {value!r} in {base_type.name!r}')
        return None

    for typ in gen.typeDict.values():
        if typ.defaultExpression:
            typ.defaultValue = _eval(typ, typ.defaultExpression)
            typ.hasDefault = True
            base.log.debug(f'evaluated {typ.defaultExpression!r} => {typ.defaultValue!r}')


def _synthesize_ext_configs_and_props(gen: base.Generator):
    """Synthesize gws.ext.config... and gws.ext.props for ext objects that don't define them explicitly"""

    # don't need this for now

    existing_names = set(t.extName for t in gen.typeDict.values() if t.extName)

    for typ in list(gen.typeDict.values()):
        if not typ.extName or not typ.extName.startswith(base.v.EXT_OBJECT_PREFIX):
            continue
        for kind in ['config', 'props']:
            parts = typ.extName.split('.')
            parts[2] = kind
            ext_name = DOT.join(parts)
            if ext_name in existing_names:
                continue
            new_typ = gen.add_type(
                c=base.c.CLASS,
                doc=typ.doc,
                ident='_' + parts[-1],
                # e.g. gws.ext.object.modelField.integer becomes gws.ext.props.modelField._integer
                name=DOT.join(parts[:-1]) + '._' + parts[-1],
                pos=typ.pos,
                tSupers=[base.v.DEFAULT_EXT_SUPERS[kind]],
                extName=ext_name,
                _SYNTHESIZED=True,
            )
            base.log.debug(f'synthesized {new_typ.uid!r} from {typ.uid!r}')


def _synthesize_ext_variant_types(gen: base.Generator):
    """Synthesize by-category variant types for ext objects

    Example:

        When we have

            gws.ext.object.layer.qgis
            gws.ext.object.layer.wms
            gws.ext.object.layer.wfs

        This will create a Variant `gws.ext.object.layer` with the members `qgis`, `wms`, `wfs`
    """

    variants = {}

    for typ in gen.typeDict.values():
        if typ.c == base.c.EXT:
            target_typ = gen.get_type(typ.tTarget)
            if not target_typ:
                base.log.debug(f'not found {typ.tTarget!r} for {typ.extName!r}')
                continue
            target_typ.extName = typ.extName
            category, _, name = typ.extName.rpartition(DOT)
            variants.setdefault(category, {})[name] = target_typ.uid

    for name, members in variants.items():
        variant_typ = gen.add_type(
            c=base.c.VARIANT,
            tMembers=members,
            name=name,
            extName=name,
        )
        base.log.debug(f'created variant {variant_typ.uid!r} for {list(members.values())}')


def _synthesize_ext_type_properties(gen: base.Generator):
    """Synthesize ``type`` properties for ext.config and ext.props objects"""

    for typ in list(gen.typeDict.values()):
        if not typ.extName or not typ.extName.startswith((base.v.EXT_CONFIG_PREFIX, base.v.EXT_PROPS_PREFIX)):
            continue
        name = typ.extName.rpartition(DOT)[-1]
        literal_typ = gen.add_type(
            c=base.c.LITERAL,
            literalValues=[name],
            pos=typ.pos,
        )
        gen.add_type(
            c=base.c.PROPERTY,
            doc='object type',
            ident=base.v.VARIANT_TAG,
            name=typ.name + DOT + base.v.VARIANT_TAG,
            pos=typ.pos,
            defaultValue='default',
            hasDefault=True,
            tValue=literal_typ.uid,
            tOwner=typ.uid,
        )


def _make_props(gen: base.Generator):
    done = {}
    own_props_by_name = {}

    for typ in gen.typeDict.values():
        if typ.c == base.c.PROPERTY:
            obj_name, _, prop_name = typ.name.rpartition('.')
            own_props_by_name.setdefault(obj_name, {})[prop_name] = typ

    def _merge(typ, props, own_props):
        for name, p in own_props.items():
            if name in props:
                # cannot weaken a required prop to optional
                if p.hasDefault and not props[name].hasDefault:
                    p.defaultValue = None
                    p.hasDefault = False

            props[name] = p

    def _make(typ, stack):
        if typ.name in done:
            return done[typ.name]
        if typ.name in stack:
            raise base.GeneratorError(f'circular inheritance {stack!r}->{typ.name!r}')

        props = {}

        for sup in typ.tSupers:
            super_typ = gen.typeDict.get(sup)
            if super_typ:
                props.update(_make(super_typ, stack + [typ.name]))
            elif sup.startswith(base.v.APP_NAME) and 'vendor' not in sup:
                base.log.warning(f'unknown supertype {sup!r}')

        if typ.name in own_props_by_name:
            _merge(typ, props, own_props_by_name[typ.name])

        typ.tProperties = {k: v.name for k, v in props.items()}

        done[typ.name] = props
        return props

    for typ in gen.typeDict.values():
        if typ.c == base.c.CLASS:
            _make(typ, [])


def _check_undefined(gen: base.Generator):
    for typ in gen.typeDict.values():
        if typ.c != base.c.UNDEFINED:
            continue
        if not typ.name.startswith(base.v.APP_NAME):
            # foreign module
            continue
        if '.vendor.' in typ.name:
            # vendor module
            continue
        if '._' in typ.name:
            # private type
            continue
        base.log.warning(f'undefined type {typ.uid!r} in {typ.pos}')


DOT = '.'
COMMA = ','
