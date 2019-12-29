### Database-related.

from .base import Optional, List, Dict, Crs
from ..data import Config, Data
from .feature import Shape
from .attribute import AttributeType


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


class StorageEntry(Data):
    category: str
    name: str
