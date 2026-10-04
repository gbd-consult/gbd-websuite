"""Select the types the server needs at run time."""

from . import base
from .base import Type


def extract(gen: base.Generator):
    """Extract the server types into ``gen.serverTypes``.

    The server needs:

    - all ``gws.ext.object`` classes, but not their properties,
    - all ``gws.ext.config`` and ``gws.ext.props`` classes and their properties, recursively,
    - command methods (``gws.ext.command``), their owners and their arguments, recursively.

    Types reachable from the application ``Config`` are marked with ``isConfig``.
    The result is sorted by uid.

    Args:
        gen: Generator state.

    Raises:
        ``GeneratorError``: If a referenced type is unknown or cannot be extracted.
    """

    p = _Extractor(gen)
    p.run()


class _Extractor:
    """Walks the types from a queue of start types and collects them."""

    out: dict[str, Type] = {}
    """Extracted types, keyed by uid."""
    queue: list[str] = []
    """Uids of the types to process."""

    def __init__(self, gen: base.Generator):
        self.gen = gen

    def run(self):
        """Extract the server types and store them in the generator."""

        self.out = {}
        self.queue = []
        self.extract()
        self.gen.serverTypes = [self.out[uid] for uid in sorted(self.out)]

    def add(self, typ: Type, **kwargs):
        """Add a type to the output, setting its module name and path.

        Args:
            typ: Type to add.
            **kwargs: Additional attributes to set, e.g. ``isConfig``.
        """

        kwargs.setdefault('isConfig', False)

        mod = self.gen.get_type(typ.tModule)
        if mod:
            kwargs['modName'] = mod.name
            kwargs['modPath'] = mod.modPath

        vars(typ).update(kwargs)
        self.out[typ.uid] = typ

    def extract(self):
        """Extract the config types, the application object and all ext types."""

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
        """Process the queue until it is empty.

        Args:
            **kwargs: Attributes to set on each extracted type.
        """

        while self.queue:
            self.extract_one(**kwargs)

    def extract_one(self, **kwargs):
        """Process the next type in the queue and enqueue the types it refers to.

        Args:
            **kwargs: Attributes to set on the extracted type.

        Raises:
            ``GeneratorError``: If the type is unknown or of a kind that cannot be extracted.
        """

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
