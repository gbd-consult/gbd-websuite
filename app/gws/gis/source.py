import re

import gws
import gws.gis.proj
import gws.gis.extent
import gws.common.metadata

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
    supported_bounds: t.List[t.Bounds] = []

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
    excludePattern: t.Regex = ''  #: match a pattern against the layer full path


class LayerFilter(t.Data):
    level: int
    names: t.List[str]
    pattern: str


def filter_layers(layers: t.List[t.SourceLayer], slf: LayerFilter, image_only=False, queryable_only=False) -> t.List[t.SourceLayer]:
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

        s = gws.get(slf, 'excludePattern')
        if s:
            layers = [sl for sl in layers if not re.search(s, sl.a_path)]

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


def crs_from_layers(source_layers: t.List[t.SourceLayer]) -> t.List[t.Crs]:
    cs = set()

    for sl in source_layers:
        if not sl.supported_crs:
            continue
        if not cs:
            cs.update(sl.supported_crs)
        else:
            cs = cs.intersection(sl.supported_crs)

    return sorted(cs)


def bounds_from_layers(source_layers: t.List[t.SourceLayer], target_crs) -> t.Bounds:
    """Return merged bounds from a list of source layers in the target_crs."""

    exts = []

    for sl in source_layers:
        if not sl.supported_bounds:
            continue
        bb = _best_bounds(sl.supported_bounds, target_crs)
        if bb:
            e = gws.gis.extent.transform(bb.extent, bb.crs, target_crs)
            exts.append(e)

    if exts:
        # gws.p('BOUNDS', [sl.name for sl in source_layers], target_crs, exts, gws.gis.extent.merge(exts))
        return t.Bounds(
            crs=target_crs,
            extent=gws.gis.extent.merge(exts))


def _best_bounds(bs: t.List[t.Bounds], target_crs):
    for b in bs:
        if gws.gis.proj.equal(b.crs, target_crs):
            return b
    for b in bs:
        if gws.gis.proj.equal(b.crs, gws.EPSG_3857):
            return b
    for b in bs:
        return b
