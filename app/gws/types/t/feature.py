### Shapes and features.

from .base import Optional, Extent, Crs
from .data import Data, Props
from .style import StyleProps
from .template import FormatObject

import shapely.geometry.base

class ShapeProps(Props):
    geometry: dict
    crs: str



class Shape:
    crs: Crs
    geo: shapely.geometry.base.BaseGeometry
    props: dict
    type: str
    wkb: str
    wkb_hex: str
    wkt: str
    bounds: Extent

    def transform(self, to_crs):
        pass


class FeatureProps(Data):
    attributes: Optional[dict]
    category: str = ''
    description: str = ''
    label: str = ''
    shape: Optional['ShapeProps']
    style: Optional['StyleProps']
    teaser: str = ''
    title: str = ''
    uid: Optional[str]


class Feature:
    attributes: dict
    description: str
    category: str
    label: str
    props: 'FeatureProps'
    shape: 'Shape'
    shape_props: 'ShapeProps'
    style: 'StyleProps'
    teaser: str
    title: str
    uid: str

    def transform(self, to_crs):
        """Transform the feature to another CRS"""
        pass

    def to_svg(self, bbox, dpi, scale, rotation):
        """Render the feature as SVG"""
        pass

    def to_geojson(self):
        """Render the feature as GeoJSON"""
        pass

    def apply_format(self, fmt: 'FormatObject', context: dict = None):
        pass