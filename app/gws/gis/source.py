import re

import gws
import gws.types as t


class LayerFilterConfig(t.Config):
    """Layer filter"""

    level: int = 0  #: use layers at this level
    names: t.Optional[t.List[str]]  #: use these layer names
    pattern: t.regex = ''  #: match a pattern against the layer full path


class LayerFilter(t.Data):
    level: int
    names: t.List[str]
    pattern: str


def filter_layers(layers: t.List[t.SourceLayer], slf: LayerFilter) -> t.List[t.SourceLayer]:
    return [sl for sl in layers if _layer_matches(sl, slf)]


def filter_image_layers(layers: t.List[t.SourceLayer], slf: LayerFilter) -> t.List[t.SourceLayer]:
    return [sl for sl in filter_layers(layers, slf) if sl.is_image]


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
