# type: ignore

### Attributes and data models.

from .base import List, Enum, FormatStr
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

    name: str = ''  #: internal name
    source: str = ''  #: source attribute
    title: str = ''  #: display title
    type: AttributeType = 'str'  #: type
    value: FormatStr = ''  #: computed value


class Attribute(Data):
    name: str
    title: str = ''
    type: str = 'str'
    value: str


class DataModel:
    attributes: List[Attribute]


class DataModelConfig(Config):
    """Data model configuration."""
    attributes: List[AttributeConfig]
