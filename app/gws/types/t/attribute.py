# type: ignore

### Attributes and data models.

from .base import Any, List, Optional, Enum, FormatStr
from ..data import Data, Config, Props
from .object import Object


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


class Attribute(Data):
    name: str
    title: str = ''
    type: AttributeType = 'str'
    value: Any = None


class DataModelRule(Data):
    """Attribute conversion rule"""

    name: str = ''  #: target attribute name
    value: Optional[str]  #: constant value
    source: str = ''  #: source attribute
    title: str = ''  #: target attribute display title
    type: AttributeType = 'str'  #: target attribute type
    format: FormatStr = ''  #: attribute formatter
    expression: str = ''  #: attribute formatter


class DataModelConfig(Config):
    """Data model."""
    rules: List[DataModelRule]


class DataModelProps(Props):
    rules: List[DataModelRule]


class DataModelObject(Object):
    rules: List[DataModelRule]

    def apply(self, atts: List['Attribute']) -> List['Attribute']:
        pass
