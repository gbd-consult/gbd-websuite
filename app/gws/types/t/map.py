### Maps and layers

from .base import List, Extent, Crs
from .auth import AuthUser
from .attribute import DataModelObject
from .object import Object
from .meta import MetaData
from .feature import Feature, FeatureProps
from .ows import OwsServiceObject
from .template import FormatObject


class LayerObject(Object):
    has_legend: bool
    has_cache: bool
    has_search: bool
    is_public: bool
    layers: List['LayerObject']

    map: 'MapObject'
    meta: 'MetaData'
    opacity: float

    title: str
    description: str

    crs: Crs
    extent: Extent
    resolutions: List[float]

    data_model: 'DataModelObject'
    feature_format: 'FormatObject'

    def mapproxy_config(self, mc):
        pass

    def render_bbox(self, bbox, width, height, **client_params):
        pass

    def render_xyz(self, x, y, z):
        pass

    def render_svg(self, bbox, dpi, scale, rotation, style):
        pass

    def render_legend(self):
        pass

    def get_features(self, bbox, limit) -> List['Feature']:
        return []

    def edit_access(self, user: 'AuthUser'):
        pass

    def edit_operation(self, operation: str, feature_props: List['FeatureProps']) -> List['Feature']:
        return []

    def ows_enabled(self, service: 'OwsServiceObject') -> bool:
        return False


class MapObject(Object):
    init_resolution: float
    layers: List['LayerObject']
    coordinatePrecision: int
    crs: Crs
    extent: Extent
    resolutions: List[float]
    coordinate_precision: int


class ProjectObject(Object):
    map: MapObject
    title: str
    locales: List[str]
    meta: 'MetaData'
