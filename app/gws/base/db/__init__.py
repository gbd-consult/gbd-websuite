import gws
import gws.types as t


class Config(gws.Config):
    """Database configuration"""

    providers: t.List[gws.ext.db.provider.Config]  #: database providers


class SqlTableConfig(gws.Config):
    """SQL database table"""

    name: str  #: table name
    geometryColumn: t.Optional[str]  #: geometry column name
    keyColumn: t.Optional[str]  #: primary key column name
    searchColumn: t.Optional[str]  #: column to be searched for
