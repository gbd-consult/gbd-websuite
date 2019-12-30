import gws
import gws.types as t


#:export IDbProvider
class Object(gws.Object, t.IDbProvider):
    pass


#:export ISqlProvider
class Sql(Object, t.ISqlProvider):
    def select(self, args: t.SelectArgs, extra_connect_params: dict = None) -> t.List[t.IFeature]:
        pass

    def edit_operation(self, operation: str, table: t.SqlTable, features: t.List[t.IFeature]) -> t.List[t.IFeature]:
        pass

    def describe(self, table: t.SqlTable) -> t.Dict[str, t.SqlTableColumn]:
        pass
