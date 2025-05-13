from . import base
from .base import Type


def extract(gen: base.Generator):
    """Extracts server specs from the types library.

    For the server, we need
        - all gws.ext.object.xxx, but not their properties (too many)
        - all gws.ext.config.xxx and gws.ext.properties.xxx
        - their properties, recursively
        - command methods (gws.ext.command.xxx)
        - their args and rets, recursively
    """

    out: dict[str, Type] = {}
    _extract(gen, out)
    gen.serverTypes = [out[uid] for uid in sorted(out)]


def _extract(gen: base.Generator, out: dict[str, Type]):
    queue = []

    def _add(typ: Type, **kwargs):
        mod = gen.get_type(typ.tModule)
        if mod:
            kwargs['modName'] = mod.name
            kwargs['modPath'] = mod.modPath
        vars(typ).update(kwargs)
        out[typ.uid] = typ

    typ = gen.require_type('gws.base.application.core.Config')
    _add(typ)
    queue.extend(typ.tProperties.values())

    typ = gen.require_type('gws.base.application.core.Object')
    _add(typ)

    queue.extend(set(typ.uid for typ in gen.typeDict.values() if typ.extName))

    while queue:
        typ = gen.require_type(queue.pop(0))
        if typ.uid in out or typ.c == base.c.ATOM:
            continue

        if typ.c == base.c.METHOD and typ.extName.startswith(base.v.EXT_COMMAND_PREFIX):
            _add(typ)
            queue.append(typ.tOwner)
            queue.append(typ.tArg)
            continue

        if typ.c == base.c.CLASS and typ.extName.startswith(base.v.EXT_OBJECT_PREFIX):
            _add(typ)
            continue

        if typ.c == base.c.CLASS:
            _add(typ)
            queue.extend(typ.tProperties.values())
            continue

        if typ.c == base.c.DICT:
            _add(typ)
            queue.append(typ.tKey)
            queue.append(typ.tValue)
            continue

        if typ.c in {base.c.LIST, base.c.SET}:
            _add(typ)
            queue.append(typ.tItem)
            continue

        if typ.c in {base.c.OPTIONAL, base.c.TYPE}:
            _add(typ)
            queue.append(typ.tTarget)
            continue

        if typ.c in {base.c.TUPLE, base.c.UNION}:
            _add(typ)
            queue.extend(typ.tItems)
            continue

        if typ.c == base.c.VARIANT:
            _add(typ)
            queue.extend(typ.tMembers.values())
            continue

        if typ.c == base.c.PROPERTY:
            _add(typ)
            queue.append(typ.tValue)
            queue.append(typ.tOwner)
            continue

        if typ.c == base.c.CLASS:
            _add(typ)
            queue.extend(typ.tProperties.values())
            continue

        if typ.c == base.c.ENUM:
            _add(typ)
            continue

        if typ.c == base.c.LITERAL:
            _add(typ)
            continue

        if typ.c == base.c.EXT:
            continue

        raise base.GeneratorError(f'unbound object {typ.c}: {typ.uid!r} in {typ.pos}')
