from typing import cast
from . import base


def extract(gen: base.Generator):
    """Extracts server specs from the types library.

    For the server, we need
        - all gws.ext.object.xxx, but not their properties (too many)
        - all gws.ext.config.xxx and gws.ext.properties.xxx
        - their properties, recursively
        - command methods (gws.ext.command.xxx)
        - their args and rets, recursively
    """

    queue = list(set(typ.uid for typ in gen.types.values() if typ.extName))
    out = {}
    _extract(gen, queue, out)
    return out


def _extract(gen, queue, out):
    while queue:

        typ = gen.types[queue.pop(0)]
        if typ.uid in out or typ.c == base.C.ATOM:
            continue

        mod = gen.types.get(typ.tModule)

        if typ.c == base.C.METHOD and typ.extName.startswith(base.EXT_COMMAND_PREFIX):
            out[typ.uid] = dict(
                c=base.C.COMMAND, extName=typ.extName, ident=typ.ident,
                tOwner=typ.tOwner, tArg=typ.tArgs[-1], modName=mod.name, modPath=mod.path)
            queue.append(typ.tOwner)
            queue.append(typ.tArgs[-1])
            continue

        if typ.c == base.C.CLASS and typ.extName.startswith(base.EXT_OBJECT_PREFIX):
            out[typ.uid] = dict(
                c=base.C.OBJECT, extName=typ.extName, ident=typ.ident,
                modName=mod.name, modPath=mod.path)
            continue

        if typ.c == base.C.CLASS and typ.extName.startswith(base.EXT_CONFIG_PREFIX):
            out[typ.uid] = dict(c=base.C.CONFIG, extName=typ.extName, tProperties=typ.tProperties)
            queue.extend(typ.tProperties.values())
            continue

        if typ.c == base.C.CLASS and typ.extName.startswith(base.EXT_PROPS_PREFIX):
            out[typ.uid] = dict(c=base.C.PROPS, extName=typ.extName, tProperties=typ.tProperties)
            queue.extend(typ.tProperties.values())
            continue

        if typ.c == base.C.DICT:
            out[typ.uid] = dict(c=typ.c, tKey=typ.tKey, tValue=typ.tValue)
            queue.append(typ.tKey)
            queue.append(typ.tValue)
            continue

        if typ.c in {base.C.LIST, base.C.SET}:
            out[typ.uid] = dict(c=typ.c, tItem=typ.tItem)
            queue.append(typ.tItem)
            continue

        if typ.c in {base.C.OPTIONAL, base.C.TYPE}:
            out[typ.uid] = dict(c=typ.c, tTarget=typ.tTarget)
            queue.append(typ.tTarget)
            continue

        if typ.c in {base.C.TUPLE, base.C.UNION}:
            out[typ.uid] = dict(c=typ.c, tItems=typ.tItems)
            queue.extend(typ.tItems)
            continue

        if typ.c == base.C.VARIANT:
            out[typ.uid] = dict(c=typ.c, tMembers=typ.tMembers)
            queue.extend(typ.tMembers.values())
            continue

        if typ.c == base.C.PROPERTY:
            out[typ.uid] = dict(c=typ.c, tValue=typ.tValue, tOwner=typ.tOwner, default=typ.default, hasDefault=typ.hasDefault)
            queue.append(typ.tValue)
            queue.append(typ.tOwner)
            continue

        if typ.c == base.C.CLASS:
            out[typ.uid] = dict(c=typ.c, tProperties=typ.tProperties)
            queue.extend(typ.tProperties.values())
            continue

        if typ.c == base.C.ENUM:
            out[typ.uid] = dict(c=typ.c, enumValues=typ.enumValues, enumDocs=typ.enumDocs)
            continue

        if typ.c == base.C.LITERAL:
            out[typ.uid] = dict(c=typ.c, values=typ.values)
            continue

        raise base.Error(f'unbound object {typ.c}: {typ.uid!r} in {typ.pos}')
