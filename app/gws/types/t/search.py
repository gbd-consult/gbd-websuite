### Search

from .base import List, Optional, Point, Extent, Crs
from .data import Data
from .object import Object
from .template import FormatObject
from .map import LayerObject, ProjectObject
from .feature import Feature, Shape


class SearchArguments(Data):
    axis: str
    bbox: Extent
    count: int
    crs: Crs
    feature_format: 'FormatObject'
    keyword: Optional[str]
    layers: List['LayerObject']
    limit: int
    params: dict
    point: Point
    project: 'ProjectObject'
    resolution: float
    shapes: List['Shape']
    tolerance: int


class SearchResult(Data):
    feature: 'Feature'
    layer: 'LayerObject'
    provider: 'SearchProviderObject'


class SearchProviderObject(Object):
    feature_format: 'FormatObject'
    geometry_required: bool
    keyword_required: bool
    title: str
    type: str

    def can_run(self, args: SearchArguments) -> bool:
        return False

    def run(self, layer: Optional['LayerObject'], args: SearchArguments) -> List['Feature']:
        return []

    def context_shape(self, args: SearchArguments):
        pass
