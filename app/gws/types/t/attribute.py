# type: ignore

### Attributes and data models.

from .base import List, Optional, Enum
from .data import Data, Config


class AttributeType(Enum):
    bool = 'bool'
    bytes = 'bytes'
    date = 'date'
    datetime = 'datetime'
    float = 'float'
    geometry = 'geometry'
    int = 'int'
    list = 'list'
    str = 'str'
    time = 'time'

    geoCurve = 'curve'
    geoGeomcollection = 'geomcollection'
    geoGeometry = 'geometry'
    geoLinestring = 'linestring'
    geoMulticurve = 'multicurve'
    geoMultilinestring = 'multilinestring'
    geoMultipoint = 'multipoint'
    geoMultipolygon = 'multipolygon'
    geoMultisurface = 'multisurface'
    geoPoint = 'point'
    geoPolygon = 'polygon'
    geoPolyhedralsurface = 'polyhedralsurface'
    geoSurface = 'surface'


class AttributeConfig(Config):
    """Attribute configuration"""

    title: str = ''  #: title
    name: str = ''  #: internal name
    value: str = ''  #: computed value
    source: str = ''  #: source attribute
    type: Optional[AttributeType]  #: type


class Attribute(Data):
    title: str = ''
    name: str = ''
    type: str = ''
    value: str = ''


class DataModel:
    attributes: List[Attribute]
