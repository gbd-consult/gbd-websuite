## Map renderer

from .base import List, Enum, Point, Size, Extent
from ..data import Data
from .style import Style

import PIL.Image


class SvgFragment:
    points: List[Point]
    svg: str


class RenderView(Data):
    bbox: Extent
    center: Point
    dpi: int
    rotation: int
    scale: int
    size_mm: Size
    size_px: Size


class RenderInputItemType(Enum):
    image = 'image'
    features = 'features'
    fragment = 'fragment'
    svg_layer = 'svg_layer'
    bbox_layer = 'bbox_layer'


class RenderInputItem(Data):
    type: str = ''
    image: PIL.Image.Image = None
    features: List['Feature']
    layer: 'LayerObject' = None
    sub_layers: List[str] = []
    opacity: float = None
    print_as_vector: bool = None
    style: 'Style' = None
    fragment: 'SvgFragment' = None
    dpi: int = None


class RenderInput(Data):
    view: 'RenderView'
    background_color: int
    items: List[RenderInputItem]


class RenderOutputItemType(Enum):
    image = 'image'
    path = 'path'
    svg = 'svg'


class RenderOutputItem(Data):
    type: str
    image: PIL.Image.Image
    path: str = ''
    elements: List[str] = []


class RenderOutput(Data):
    view: 'RenderView'
    items: List[RenderOutputItem]
