import re

import gws
import gws.gis.proj
import gws.gis.extent

import gws.types as t


#:export
class SourceStyle(t.Data):
    is_default = False
    legend: t.Url = ''
    meta: t.MetaData = None


#:export
class SourceLayer(t.Data):
    data_source = {}

    supported_crs: t.List[t.Crs] = []
    extents: t.Dict[t.Crs, t.Extent] = {}

    is_expanded = False
    is_group = False
    is_image = False
    is_queryable = False
    is_visible = False

    layers: t.List['SourceLayer'] = []

    meta: t.MetaData = None
    name = ''
    title = ''

    opacity = 1
    scale_range: t.List[float] = []
    styles: t.List[SourceStyle] = []
    legend = ''
    resource_urls = {}

    a_path = ''
    a_uid = ''
    a_level = 0


class LayerFilterConfig(t.Config):
    """Layer filter"""

    level: int = 0  #: use layers at this level
    names: t.Optional[t.List[str]]  #: use these layer names (top-to-bottom order)
    pattern: t.Regex = ''  #: match a pattern against the layer full path


class LayerFilter(t.Data):
    level: int
    names: t.List[str]
    pattern: str


def filter_layers(
        layers: t.List[t.SourceLayer],
        slf: LayerFilter,
        image_only=False,
        queryable_only=False) -> t.List[t.SourceLayer]:
    if slf:
        s = gws.get(slf, 'level')
        if s:
            layers = [sl for sl in layers if sl.a_level == s]

        s = gws.get(slf, 'names')
        if s:
            # NB: if 'names' is given, maintain the given order, which is expected to be top-to-bottom
            # see note in ext/layers/wms
            layers2 = []
            for name in s:
                for sl in layers:
                    if sl.name == name:
                        layers2.append(sl)
                        break
            layers = layers2

        s = gws.get(slf, 'pattern')
        if s:
            layers = [sl for sl in layers if re.search(s, sl.a_path)]

    if image_only:
        layers = [sl for sl in layers if sl.is_image]
    if queryable_only:
        layers = [sl for sl in layers if sl.is_queryable]

    return layers


def image_layers(sl: t.SourceLayer) -> t.List[t.SourceLayer]:
    if sl.is_image:
        return [sl]
    if sl.layers:
        return [s for sub in sl.layers for s in image_layers(sub)]
    return []


def best_source_layer_extent(sl: t.SourceLayer, map_crs):
    for crs, ext in sl.extents.items():
        if crs == map_crs:
            return crs, ext
    for crs, ext in sl.extents.items():
        if gws.gis.proj.equal(crs, 'EPSG:4326'):
            return crs, ext
    for crs, ext in sl.extents.items():
        return crs, ext


def extent_from_layers(sls: t.List[t.SourceLayer], map_crs):
    exts = []
    for sl in sls:
        if sl.extents:
            crs, ext = best_source_layer_extent(sl, map_crs)
            ext = gws.gis.extent.valid(ext)
            if ext:
                exts.append(gws.gis.proj.transform_bbox(ext, crs, map_crs))
    if exts:
        return gws.gis.extent.merge(exts)
