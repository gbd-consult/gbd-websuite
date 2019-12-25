# type: ignore

### Database-related.

from .base import Optional, List, Dict, Crs
from ..data import Config, Data
from .auth import AuthUser
from .object import Object
from .feature import Shape, Feature
from .attribute import Attribute, AttributeType


class SqlTableConfig(Config):
    """SQL database table"""

    name: str  #: table name
    geometryColumn: Optional[str]  #: geometry column name
    keyColumn: Optional[str]  #: primary key column name
    searchColumn: Optional[str]  #: column to be searched for


class SqlTable(Data):
    name: str
    key_column: str = ''
    search_column: str = ''
    geometry_column: str = ''
    geometry_type: str = ''
    geometry_crs: Crs = ''


class SelectArgs(Data):
    keyword: Optional[str]
    limit: Optional[int]
    tolerance: Optional[float]
    shape: Optional['Shape']
    sort: Optional[str]
    table: SqlTable
    uids: Optional[List[str]]
    extraWhere: Optional[str]


class SqlTableColumn(Data):
    name: str
    type: 'AttributeType'
    native_type: str
    crs: Crs
    is_key: bool
    is_geometry: bool


#: alias:
SqlTableDescription = Dict[str, SqlTableColumn]


class DbProviderObject(Object):
    pass


class SqlProviderObject(DbProviderObject):
    error: type
    connect_params: dict

    def select(self, args: SelectArgs, extra_connect_params: dict = None) -> List['Feature']:
        pass

    def edit_operation(self, operation: str, table: SqlTable, features: List['Feature']) -> List['Feature']:
        pass

    def describe(self, table: SqlTable) -> SqlTableDescription:
        pass


class StorageEntry(Data):
    category: str
    name: str


class StorageObject(Object):
    def read(self, entry: StorageEntry, user: 'AuthUser') -> dict:
        return {}

    def write(self, entry: StorageEntry, user: 'AuthUser', data: dict) -> str:
        return ''

    def dir(self, user: 'AuthUser') -> List[StorageEntry]:
        return []
