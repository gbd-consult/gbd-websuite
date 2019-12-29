### Shapes and features.

from .base import Optional, List
from ..data import Data, Props
from .style import StyleProps
from .attribute import Attribute

import shapely.geometry.base


class ShapeProps(Props):
    geometry: dict
    crs: str


class FeatureProps(Data):
    uid: str = ''
    attributes: List['Attribute'] = ''
    elements: dict = {}
    layerUid: str = ''
    shape: Optional['ShapeProps']
    style: Optional['StyleProps']
