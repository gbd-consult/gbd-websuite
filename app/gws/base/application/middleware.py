"""Middleware manager."""

import gws


class Object(gws.MiddlewareManager):
    """Middleware manager."""

    objectMap: dict[str, gws.Node]
    """Registered objects by name."""
    deps: dict[str, list[str]]
    """Dependency names by object name."""
    names: list[str]
    """Object names in dependency order."""

    def __init__(self):
        self.objectMap = {}
        self.deps = {}
        self.names = []
        self.sorted = False

    def register(self, obj, name, depends_on=None):
        self.objectMap[name] = obj
        self.deps[name] = depends_on
        self.sorted = False

    def objects(self):
        if not self.sorted:
            self._sort()
            self.sorted = True
        return [self.objectMap[name] for name in self.names]

    def _sort(self):
        """Sort the object names in dependency order."""
        self.names = []
        colors = {}
        for name in self.objectMap:
            self._sort_visit(name, colors, [])

    def _sort_visit(self, name, colors, stack):
        """Visit a name and its dependencies depth first, raise on cycles and unknown names."""
        stack = stack + [name]

        if colors.get(name) == 2:
            return
        if colors.get(name) == 1:
            raise gws.Error('middleware: cyclic dependency: ' + '->'.join(stack))

        if name not in self.objectMap:
            raise gws.Error('middleware: not found: ' + '->'.join(stack))

        colors[name] = 1

        depends_on = self.deps[name]
        if depends_on:
            for d in depends_on:
                self._sort_visit(d, colors, stack)

        colors[name] = 2
        self.names.append(name)
