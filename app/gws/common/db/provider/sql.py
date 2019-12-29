import gws
import gws.types as t


#:stub SqlProviderObject
class Object(gws.Object):
    def select(self, args: t.SelectArgs, extra_connect_params: dict = None) -> t.List[t.Feature]:
        pass

    def edit_operation(self, operation: str, table: t.SqlTable, features: t.List[t.Feature]) -> t.List[t.Feature]:
        pass

    def describe(self, table: t.SqlTable) -> t.SqlTableDescription:
        pass
