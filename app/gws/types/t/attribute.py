### Attributes and data models.

from .base import Any, Enum
from ..data import Data


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


class GeometryType(Enum):
    curve = 'curve'
    geomcollection = 'geomcollection'
    geometry = 'geometry'
    linestring = 'linestring'
    multicurve = 'multicurve'
    multilinestring = 'multilinestring'
    multipoint = 'multipoint'
    multipolygon = 'multipolygon'
    multisurface = 'multisurface'
    point = 'point'
    polygon = 'polygon'
    polyhedralsurface = 'polyhedralsurface'
    surface = 'surface'


class Attribute(Data):
    name: str
    title: str = ''
    type: AttributeType = 'str'
    value: Any = None
    editable: bool = True
