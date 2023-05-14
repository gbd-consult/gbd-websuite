import gws


class Manager:
    objects: dict[str, gws.IMiddleware]
    deps: dict[str, list[str]]
    names: list[str]

    def __init__(self):
        self.objects = {}
        self.deps = {}
        self.names = []
        self.sorted = False

    def register(self, name, obj, depends_on=None):
        self.objects[name] = obj
        self.deps[name] = depends_on
        self.sorted = False

    def sorted_objects(self):
        if not self.sorted:
            self._sort()
            self.sorted = True
        return [(name, self.objects[name]) for name in self.names]

    def _sort(self):
        self.names = []
        colors = {}
        for name in self.objects:
            self._sort_visit(name, colors, [])

    def _sort_visit(self, name, colors, stack):
        stack += [name]

        if colors.get(name) == 2:
            return
        if colors.get(name) == 1:
            raise gws.Error('middleware: cyclic dependency: ' + '->'.join(stack))

        if name not in self.objects:
            raise gws.Error('middleware: not found: ' + '->'.join(stack))

        colors[name] = 1

        depends_on = self.deps[name]
        if depends_on:
            for d in depends_on:
                self._sort_visit(d, colors, stack)

        colors[name] = 2
        self.names.append(name)
