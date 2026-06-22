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

    p = _Extractor(gen)
    p.run()


class _Extractor:
    out: dict[str, Type] = {}
    queue: list[str] = []

    def __init__(self, gen: base.Generator):
        self.gen = gen

    def run(self):
        self.out = {}
        self.queue = []
        self.extract()
        self.gen.serverTypes = [self.out[uid] for uid in sorted(self.out)]

    def add(self, typ: Type, **kwargs):
        kwargs.setdefault('isConfig', False)

        mod = self.gen.get_type(typ.tModule)
        if mod:
            kwargs['modName'] = mod.name
            kwargs['modPath'] = mod.modPath

        vars(typ).update(kwargs)
        self.out[typ.uid] = typ

    def extract(self):
        # config-related types
        self.queue = ['gws.base.application.core.Config']
        self.extract_all(isConfig=True)

        # application objects, including methods and their args/rets
        self.queue = ['gws.base.application.core.Object']
        self.extract_all()

        # ext objects
        self.queue = list(set(typ.uid for typ in self.gen.typeDict.values() if typ.extName))
        self.extract_all()

    def extract_all(self, **kwargs):
        while self.queue:
            self.extract_one(**kwargs)

    def extract_one(self, **kwargs):
        typ = self.gen.require_type(self.queue.pop(0))

        if typ.uid in self.out or typ.c == base.c.ATOM:
            return

        if typ.c == base.c.METHOD and typ.extName.startswith(base.v.EXT_COMMAND_PREFIX):
            self.add(typ, **kwargs)
            self.queue.append(typ.tOwner)
            self.queue.append(typ.tArg)
            return

        if typ.c == base.c.CLASS and typ.extName.startswith(base.v.EXT_OBJECT_PREFIX):
            self.add(typ, **kwargs)
            return

        if typ.c == base.c.CLASS:
            self.add(typ, **kwargs)
            self.queue.extend(typ.tProperties.values())
            return

        if typ.c == base.c.DICT:
            self.add(typ, **kwargs)
            self.queue.append(typ.tKey)
            self.queue.append(typ.tValue)
            return

        if typ.c in {base.c.LIST, base.c.SET}:
            self.add(typ, **kwargs)
            self.queue.append(typ.tItem)
            return

        if typ.c in {base.c.OPTIONAL, base.c.TYPE}:
            self.add(typ, **kwargs)
            self.queue.append(typ.tTarget)
            return

        if typ.c in {base.c.TUPLE, base.c.UNION}:
            self.add(typ, **kwargs)
            self.queue.extend(typ.tItems)
            return

        if typ.c == base.c.VARIANT:
            self.add(typ, **kwargs)
            self.queue.extend(typ.tMembers.values())
            return

        if typ.c == base.c.PROPERTY:
            self.add(typ, **kwargs)
            self.queue.append(typ.tValue)
            self.queue.append(typ.tOwner)
            return

        if typ.c == base.c.CLASS:
            self.add(typ, **kwargs)
            self.queue.extend(typ.tProperties.values())
            return

        if typ.c == base.c.ENUM:
            self.add(typ, **kwargs)
            return

        if typ.c == base.c.LITERAL:
            self.add(typ, **kwargs)
            return

        if typ.c == base.c.EXT:
            return

        raise base.GeneratorError(f'unbound object {typ.c}: {typ.uid!r} in {typ.pos}')
