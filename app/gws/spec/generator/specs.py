from typing import Dict
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

    queue = []
    out: Dict[str, Dict] = dict()

    typ = gen.types['gws.base.application.core.Config']
    out[typ.uid] = dict(c=base.C.CONFIG, tProperties=typ.tProperties)
    queue.extend(typ.tProperties.values())

    typ = gen.types['gws.base.application.core.Object']
    mod = gen.types.get(typ.tModule)
    out[typ.uid] = dict(c=base.C.OBJECT, modName=mod.name, modPath=mod.path)  # type: ignore

    queue.extend(set(typ.uid for typ in gen.types.values() if typ.extName))

    _extract(gen, queue, out)

    for uid, spec in out.items():
        typ = gen.types[uid]
        spec.setdefault('c', typ.c)
        spec.setdefault('uid', typ.uid)
        if typ.ident:
            spec.setdefault('ident', typ.ident)
        if typ.extName:
            spec.setdefault('extName', typ.extName)

    return sorted(out.values(), key=lambda t: t['uid'])


def _extract(gen, queue, out):
    while queue:

        typ = gen.types[queue.pop(0)]
        if typ.uid in out or typ.c == base.C.ATOM:
            continue

        if typ.c == base.C.METHOD and typ.extName.startswith(base.EXT_COMMAND_PREFIX):
            mod = gen.types.get(typ.tModule)
            out[typ.uid] = dict(c=base.C.COMMAND, tOwner=typ.tOwner, tArg=typ.tArgs[-1], modName=mod.name, modPath=mod.path)
            queue.append(typ.tOwner)
            queue.append(typ.tArgs[-1])
            continue

        if typ.c == base.C.CLASS and typ.extName.startswith(base.EXT_OBJECT_PREFIX):
            mod = gen.types.get(typ.tModule)
            out[typ.uid] = dict(c=base.C.OBJECT, modName=mod.name, modPath=mod.path)
            continue

        if typ.c == base.C.CLASS and typ.extName.startswith(base.EXT_CONFIG_PREFIX):
            out[typ.uid] = dict(c=base.C.CONFIG, tProperties=typ.tProperties)
            queue.extend(typ.tProperties.values())
            continue

        if typ.c == base.C.CLASS and typ.extName.startswith(base.EXT_PROPS_PREFIX):
            out[typ.uid] = dict(c=base.C.PROPS, tProperties=typ.tProperties)
            queue.extend(typ.tProperties.values())
            continue

        if typ.c == base.C.CLASS:
            mod = gen.types.get(typ.tModule)
            out[typ.uid] = dict(c=base.C.CLASS, modName=mod.name, modPath=mod.path, tProperties=typ.tProperties)
            queue.extend(typ.tProperties.values())
            continue

        if typ.c == base.C.DICT:
            out[typ.uid] = dict(tKey=typ.tKey, tValue=typ.tValue)
            queue.append(typ.tKey)
            queue.append(typ.tValue)
            continue

        if typ.c in {base.C.LIST, base.C.SET}:
            out[typ.uid] = dict(tItem=typ.tItem)
            queue.append(typ.tItem)
            continue

        if typ.c in {base.C.OPTIONAL, base.C.TYPE}:
            out[typ.uid] = dict(tTarget=typ.tTarget)
            queue.append(typ.tTarget)
            continue

        if typ.c in {base.C.TUPLE, base.C.UNION}:
            out[typ.uid] = dict(tItems=typ.tItems)
            queue.extend(typ.tItems)
            continue

        if typ.c == base.C.VARIANT:
            out[typ.uid] = dict(tMembers=typ.tMembers)
            queue.extend(typ.tMembers.values())
            continue

        if typ.c == base.C.PROPERTY:
            out[typ.uid] = dict(tValue=typ.tValue, tOwner=typ.tOwner, default=typ.default, hasDefault=typ.hasDefault)
            queue.append(typ.tValue)
            queue.append(typ.tOwner)
            continue

        if typ.c == base.C.CLASS:
            out[typ.uid] = dict(tProperties=typ.tProperties)
            queue.extend(typ.tProperties.values())
            continue

        if typ.c == base.C.ENUM:
            out[typ.uid] = dict(enumValues=typ.enumValues, enumDocs=typ.enumDocs)
            continue

        if typ.c == base.C.LITERAL:
            out[typ.uid] = dict(literalValues=typ.literalValues)
            continue

        if typ.c == base.C.EXT:
            continue

        raise base.Error(f'unbound object {typ.c}: {typ.uid!r} in {typ.pos}')
