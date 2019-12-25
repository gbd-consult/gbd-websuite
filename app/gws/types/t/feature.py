### Shapes and features.

from .base import Optional, List, Extent, Crs
from ..data import Data, Props
from .style import StyleProps
from .attribute import Attribute, DataModelObject
from .template import FormatObject
from .map import LayerObject

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
    uid: str = ''
    attributes: List['Attribute'] = ''
    elements: dict = {}
    layerUid: str = ''
    shape: Optional['ShapeProps']
    style: Optional['StyleProps']


class FeatureConvertor:
    feature_format: 'FormatObject'
    data_model: 'DataModelObject'


class Feature:
    attributes: List['Attribute']
    elements: dict
    convertor: FeatureConvertor
    layer: 'LayerObject'
    props: 'FeatureProps'
    shape: 'Shape'
    shape_props: 'ShapeProps'
    style: 'StyleProps'
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

    def set_default_style(self, style):
        pass

    def convert(self, target_crs: Crs, convertor: 'FeatureConvertor' = None) -> 'Feature':
        pass
