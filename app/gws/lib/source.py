import re

import gws
import gws.types as t
import gws.lib.metadata
import gws.lib.extent
import gws.lib.proj



class Style(gws.Data):
    is_default: bool
    legend: gws.Url
    meta: gws.lib.metadata.Data
    name: str


class Layer(gws.Data):
    a_level: int
    a_path: str
    a_uid: str

    data_source: dict

    supported_crs: t.List[gws.Crs]
    supported_bounds: t.List[gws.Bounds]

    is_expanded: bool
    is_group: bool
    is_image: bool
    is_queryable: bool
    is_visible: bool

    layers: t.List['Layer']

    meta: gws.lib.metadata.Data
    name: str
    title: str

    opacity: int
    scale_range: t.List[float]
    styles: t.List[Style]
    legend: str
    resource_urls: dict


class Filter(gws.Data):
    """Source layer filter"""

    level: int = 0  #: match only layers at this level
    names: t.Optional[t.List[str]]  #: match these layer names (top-to-bottom order)
    pattern: gws.Regex = ''  #: match layers whose full path matches a pattern


def layer_matches(sl: Layer, slf: Filter) -> bool:
    """Check if a source layer matches the filter"""

    if not slf:
        return True

    s = gws.get(slf, 'level')
    if s and sl.a_level != s:
        return False

    s = gws.get(slf, 'names')
    if s is not None and sl.name not in s:
        return False

    s = gws.get(slf, 'pattern')
    if s and not re.search(s, sl.a_path):
        return False

    return True


def filter_layers(layers: t.List[Layer], slf: Filter, image_only=False, queryable_only=False) -> t.List[Layer]:
    """Filter source layers by the given layer filter."""

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


def image_layers(sl: Layer) -> t.List[Layer]:
    if sl.is_image:
        return [sl]
    if sl.layers:
        return [s for sub in sl.layers for s in image_layers(sub)]
    return []


def bounds_from_layers(source_layers: t.List[Layer], target_crs) -> gws.Bounds:
    """Return merged bounds from a list of source layers in the target_crs."""

    exts = []

    for sl in source_layers:
        if not sl.supported_bounds:
            continue
        bb = _best_bounds(sl.supported_bounds, target_crs)
        if bb:
            e = gws.lib.extent.transform(bb.extent, bb.crs, target_crs)
            exts.append(e)

    if exts:
        # gws.p('BOUNDS', [sl.name for sl in source_layers], target_crs, exts, gws.lib.extent.merge(exts))
        return gws.Bounds(
            crs=target_crs,
            extent=gws.lib.extent.merge(exts))


def crs_from_layers(source_layers: t.List[Layer]) -> t.List[gws.Crs]:
    """Return an intersection of crs supported by each source layer."""

    cs: t.Set[str] = set()

    for sl in source_layers:
        if not sl.supported_crs:
            continue
        if not cs:
            cs.update(sl.supported_crs)
        else:
            cs = cs.intersection(sl.supported_crs)

    return sorted(cs)


def _best_bounds(bs: t.List[gws.Bounds], target_crs):
    for b in bs:
        if gws.lib.proj.equal(b.crs, target_crs):
            return b
    for b in bs:
        if gws.lib.proj.equal(b.crs, gws.EPSG_3857):
            return b
    for b in bs:
        return b
