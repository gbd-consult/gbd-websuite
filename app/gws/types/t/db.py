# type: ignore

### Database-related.

from .base import Optional, List
from .data import Config, Data
from .auth import AuthUser
from .object import Object
from .feature import Shape, Feature
from .attribute import Attribute


class SqlTableConfig(Config):
    """SQL database table"""

    geometryColumn: Optional[str]  #: geometry column name
    keyColumn: str = 'id'  #: primary key column name
    name: str  #: table name
    searchColumn: Optional[str]  #: column to be searched for


class SelectArgs(Data):
    keyword: Optional[str]
    limit: Optional[int]
    tolerance: Optional[float]
    shape: Optional['Shape']
    sort: Optional[str]
    table: SqlTableConfig
    ids: Optional[List[str]]
    extraWhere: Optional[str]


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


class SqlProviderObject(Object):
    error: type
    connect_params: dict

    def connect(self, extra_connect_params: dict = None):
        pass

    def select(self, args: SelectArgs, extra_connect_params: dict = None) -> List['Feature']:
        pass

    def insert(self, table: SqlTableConfig, recs: List[dict]) -> List[str]:
        pass

    def update(self, table: SqlTableConfig, recs: List[dict]) -> List[str]:
        pass

    def delete(self, table: SqlTableConfig, recs: List[dict]) -> List[str]:
        pass

    def describe(self, table: SqlTableConfig) -> List['Attribute']:
        pass
    