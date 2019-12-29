### Search

from .base import List, Optional, Point, Extent, Crs
from ..data import Data


class SearchArgs(Data):
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

