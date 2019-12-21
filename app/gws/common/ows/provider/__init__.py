import gws
import gws.types as t


class Object(gws.Object, t.OwsProviderObject):
    def operation(self, name: str) -> t.OwsOperation:
        for op in self.operations:
            if op.name == name:
                return op
