import re

import gws
import gws.types as t


class BaseConfig(t.Config):
    #: CRS for this source
    crs: t.Optional[t.crsref]
    #: extent
    extent: t.Optional[t.Extent]
    #: object type
    type: str


class LayerFilterConfig(t.Config):
    """Layer filter"""

    level: int = 0
    names: t.Optional[t.List[str]]
    pattern: t.regex = ''


class LayerFilter(t.Data):
    level: int
    names: t.List[str]
    pattern: str


class Base(gws.Object, t.SourceObject):
    def __init__(self):
        super().__init__()
        self.crs = ''
        self.extent = None
        self.layers = []
        self.service = None

    def configure(self):
        super().configure()
        self.uid = self.parent.uid + '.' + self.uid

    def service_metadata(self):
        if self.service:
            return self.service.meta

    def layer_metadata(self, layer_name):
        if self.service:
            for la in self.service.layers:
                if la.name == layer_name:
                    return la.meta

    def mapproxy_config(self, mc, options=None):
        pass

    def get_features(self, keyword, shape, sort, limit):
        return []

    def modify_features(self, operation, feature_params):
        pass


def filter_layers(layers: t.List[t.SourceLayer], slf: LayerFilter) -> t.List[t.SourceLayer]:
    return [
        sl
        for sl in layers
        if _layer_matches(sl, slf)
    ]


def _layer_matches(layer: t.SourceLayer, slf: LayerFilter):
    if not slf:
        return True
    if slf.names and layer.name not in slf.names:
        return False
    if slf.level and layer.a_level != slf.level:
        return False
    if slf.pattern and not re.search(slf.pattern, layer.a_path):
        return False
    return True
