import re

from typing import cast

from . import base, util


def normalize(gen: base.Generator):
    _add_global_aliases(gen)
    _expand_aliases(gen)
    _resolve_aliases(gen)
    _check_variants(gen)
    _evaluate_defaults(gen)
    _synthesize_ext_configs_and_props(gen)
    _synthesize_ext_type_properties(gen)
    _synthesize_ext_variant_types(gen)
    _check_undefined(gen)
    _make_props(gen)


##


def _add_global_aliases(gen):
    """Add globals aliases.

     If we have `mod.GlobalName` and `mod.some.module.GlobalName`, and `mod.some.module` 
     is in `GLOBAL_MODULES`, the former should an alias for the latter.
     """

    for typ in gen.types.values():
        if typ.name in gen.aliases:
            continue
        m = re.match(r'^gws\.([A-Z].*)$', typ.name)
        if not m:
            continue
        for mod in base.GLOBAL_MODULES:
            name = mod + DOT + m.group(1)
            if name in gen.types:
                base.log.debug(f'global alias {typ.name!r} => {name!r}')
                gen.aliases[typ.name] = name
                break


def _expand_aliases(gen):
    def _exp(target, stack):
        if target in gen.types:
            return target
        if target in stack:
            raise base.Error(f'circular alias {stack!r} => {target!r}')
        if target in gen.aliases:
            return _exp(gen.aliases[target], stack + [target])
        if target.startswith(base.APP_NAME):
            base.log.warn(f'unbound alias {target!r}')
        return target

    new_aliases = {}
    for src, target in gen.aliases.items():
        new_aliases[src] = _exp(target, [])
    gen.aliases = new_aliases


_type_scalars = [
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


def _resolve_aliases(gen):
    new_types = {}

    def _rename_uid(uid):
        if uid in gen.aliases:
            new = gen.aliases[uid]
        else:
            new = COMMA.join(gen.aliases.get(s, s) for s in uid.split(COMMA))
        if new != uid:
            base.log.debug(f'resolved alias {uid!r} => {new!r}')
        return new

    for typ in gen.types.values():
        if typ.uid in new_types:
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

        new_types[typ.uid] = typ

    gen.types = new_types


def _evaluate_defaults(gen):
    """Replace enum and constant values with literal values"""

    def _get_type(name):
        if name in gen.aliases:
            name = gen.aliases[name]
        return gen.types.get(name)

    def _eval(val):
        c, value = val
        if c == base.C.LITERAL:
            return value
        if isinstance(value, list):
            return [_eval(v) for v in value]

        if isinstance(value, dict):
            return {k: _eval(v) for k, v in value.items()}

        # constant?
        typ = _get_type(value)
        if typ and typ.c == base.C.CONSTANT:
            return typ.value

        # enum?
        obj_name, _, item = value.rpartition('.')
        typ = _get_type(obj_name)
        if typ and typ.c == base.C.ENUM and item in typ.enumValues:
            return typ.enumValues[item]

        base.log.warn(f'invalid expression {value!r}')
        return None

    for typ in gen.types.values():
        d = getattr(typ, 'EVAL_DEFAULT', None)
        if d:
            typ.default = _eval(d)
            typ.hasDefault = True
            base.log.debug(f'evaluated {d!r} => {typ.default!r}')


def _check_variants(gen):
    """Create Variant objects from VariantStubs

    Example:

    Given

        Foo: VariantStub { items ['Type1', 'Type2'] }
        Type1 { type t.Literal['first'] }
        Type2 { type t.Literal['second'] }

    we create a mapping { "type value" => "type name" }, e.g;

        Foo: Variant {
            tMembers {
                first:  Type1
                second: Type2
            }
        }
    """

    for typ in gen.types.values():
        if typ.c == base.C.VARIANT and not typ.tMembers:
            members = {}
            for tItem in typ.tItems:
                try:
                    item_type = gen.types.get(tItem)
                    tag_property_type = gen.types.get(item_type.name + '.' + base.VARIANT_TAG)
                    tag_value_type = gen.types.get(tag_property_type.tValue)
                    if tag_value_type.c == base.C.LITERAL and len(tag_value_type.literalValues) == 1:
                        members[tag_value_type.literalValues[0]] = item_type.name
                    else:
                        raise ValueError()
                except Exception:
                    raise base.Error(f'invalid Variant: {typ.pos!r}')
            delattr(typ, 'tItems')
            typ.tMembers = members


def _synthesize_ext_configs_and_props(gen):
    """Synthesize gws.ext.config... and gws.ext.props for ext objects that don't define them explicitly"""

    upd = {}

    existing_names = set(t.extName for t in gen.types.values() if t.extName)

    for t in gen.types.values():
        if t.extName.startswith(base.EXT_OBJECT_PREFIX):
            ps = t.extName.split('.')
            for kind in ['config', 'props']:
                ps[2] = kind
                name = DOT.join(ps)
                if name not in existing_names:
                    nt = gen.new_type(
                        base.C.CLASS,
                        doc=t.doc,
                        ident=kind,
                        name=name,
                        pos=t.pos,
                        tSupers=[base.DEFAULT_EXT_SUPERS[kind]],
                        extName=name,
                    )
                    upd[nt.uid] = nt

    gen.types.update(upd)


def _synthesize_ext_type_properties(gen):
    """Synthesize type properties for ext.config and ext.props objects"""

    upd = {}

    for t in gen.types.values():
        if t.extName.startswith((base.EXT_CONFIG_PREFIX, base.EXT_PROPS_PREFIX)):
            name = t.extName.rpartition(DOT)[-1]
            literal = gen.new_type(base.C.LITERAL, literalValues=[name], pos=t.pos)
            upd[literal.uid] = literal

            nt = gen.new_type(
                base.C.PROPERTY,
                doc='object type',
                ident=base.VARIANT_TAG,
                name=t.name + DOT + base.VARIANT_TAG,
                pos=t.pos,
                default='default',
                hasDefault=True,
                tValue=literal.uid,
                tOwner=t.uid,
            )
            upd[nt.uid] = nt

    gen.types.update(upd)


def _synthesize_ext_variant_types(gen):
    """Synthesize by-category variant types for ext objects

    Example:

        When we have

            gws.ext.object.layer.qgis
            gws.ext.object.layer.wms
            gws.ext.object.layer.wfs

        This will create a Variant `gws.ext.object.layer` with the members `qgis`, `wms`, `wfs`
    """

    variants = {}

    for typ in gen.types.values():
        if typ.extName and not typ.extName.startswith(base.EXT_COMMAND_PREFIX):
            # "gws.ext.object.owsService.wms" belongs to "gws.ext.object.owsService"
            var_name, _, name = typ.extName.rpartition(DOT)
            variants.setdefault(var_name, {})[name] = typ.name

    upd = {}

    for name, members in variants.items():
        variant_typ = gen.new_type(base.C.VARIANT, tMembers=members)
        upd[variant_typ.uid] = variant_typ
        alias_typ = gen.new_type(base.C.TYPE, name=name, extName=name, tTarget=variant_typ.uid)
        upd[alias_typ.uid] = alias_typ
        base.log.debug(f'created variant {variant_typ.uid!r} for {list(members.values())}')

    gen.types.update(upd)


def _make_props(gen):
    done = {}
    own_props_by_name = {}

    for typ in gen.types.values():
        if typ.c == base.C.PROPERTY:
            obj_name, _, prop_name = typ.name.rpartition('.')
            own_props_by_name.setdefault(obj_name, {})[prop_name] = typ

    def _merge(typ, props, own_props):
        for name, p in own_props.items():
            if name in props:
                # cannot weaken a required prop to optional
                if p.hasDefault and not props[name].hasDefault:
                    p.default = None
                    p.hasDefault = False

            props[name] = p

    def _make(typ, stack):
        if typ.name in done:
            return done[typ.name]
        if typ.name in stack:
            raise base.Error(f'circular inheritance {stack!r}->{typ.name!r}')

        props = {}

        for sup in typ.tSupers:
            super_typ = gen.types.get(sup)
            if super_typ:
                props.update(_make(super_typ, stack + [typ.name]))
            elif sup.startswith(base.APP_NAME) and 'vendor' not in sup:
                base.log.warn(f'unknown supertype {sup!r}')

        if typ.name in own_props_by_name:
            _merge(typ, props, own_props_by_name[typ.name])

        typ.tProperties = {k: v.name for k, v in props.items()}

        done[typ.name] = props
        return props

    for typ in gen.types.values():
        if typ.c == base.C.CLASS:
            _make(typ, [])


def _check_undefined(gen):
    for typ in gen.types.values():
        if typ.c == base.C.UNDEFINED:
            base.log.warn(f'undefined type {typ.uid!r} in {typ.pos}')


DOT = '.'
COMMA = ','
