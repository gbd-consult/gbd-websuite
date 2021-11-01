import re
import gws
import gws.types as t
import gws.lib.metadata
import gws.lib.extent
import gws.lib.gis.util


##

class LayerFilterConfig(gws.Config):
    """Source layer filter"""

    level: int = 0  #: match only layers at this level
    names: t.Optional[t.List[str]]  #: match these layer names (top-to-bottom order)
    pattern: gws.Regex = ''  #: match layers whose full path matches a pattern
    onlyGroups: bool = False  #: if true, match only group layers
    onlyLeaves: bool = False  #: if true, match only leaf layers


class LayerFilter(gws.Data):
    level: int
    names: t.List[str]
    pattern: gws.Regex
    is_group: bool
    is_image: bool
    is_queryable: bool


def layer_filter_from_config(cfg, **kwargs) -> t.Optional[LayerFilter]:
    if not cfg:
        return None

    slf = LayerFilter(
        level=cfg.level or 0,
        names=cfg.names or [],
        pattern=cfg.pattern,
        **kwargs
    )
    if cfg.onlyGroups:
        slf.is_group = True
    if cfg.onlyLeaves:
        slf.is_group = False
    return slf


def layer_matches(sl: gws.SourceLayer, slf: t.Optional[LayerFilter]) -> bool:
    """Check if a source layer matches the filter"""

    if not slf:
        return True

    if slf.level and sl.a_level != slf.level:
        return False

    if slf.names and sl.name not in slf.names:
        return False

    if slf.pattern and not re.search(slf.pattern, sl.a_path):
        return False

    if slf.is_group is not None and sl.is_group != slf.is_group:
        return False

    if slf.is_image is not None and sl.is_image != slf.is_image:
        return False

    if slf.is_queryable is not None and sl.is_queryable != slf.is_queryable:
        return False

    return True


def filter_layers(layers: t.List[gws.SourceLayer], slf: LayerFilter) -> t.List[gws.SourceLayer]:
    """Filter source layers by the given layer filter."""

    if not slf:
        return layers

    found = []

    def walk(sl):
        # if a layer matches, add it and don't go any further
        # otherwise, inspect the sublayers (@TODO optimize if slf.level is given)
        if layer_matches(sl, slf):
            found.append(sl)
            return
        for sl2 in sl.layers:
            walk(sl2)

    for sl in layers:
        walk(sl)

    # NB: if 'names' is given, maintain the given order, which is expected to be top-to-bottom
    # see note in ext/layers/wms

    if slf.names:
        found.sort(key=lambda sl: slf.names.index(sl.name) if sl.name in slf.names else -1)

    return found


def combined_bounds(layers: t.List[gws.SourceLayer], target_crs: gws.ICrs) -> gws.Bounds:
    """Return merged bounds from a list of source layers in the target_crs."""

    exts = []

    for sl in layers:
        if not sl.supported_bounds:
            continue
        b = gws.lib.gis.util.best_bounds(target_crs, sl.supported_bounds)
        if b:
            ext = gws.lib.extent.transform(b.extent, b.crs, target_crs)
            exts.append(ext)

    if exts:
        return gws.Bounds(crs=target_crs, extent=gws.lib.extent.merge(exts))


def supported_crs_list(layers: t.List[gws.SourceLayer]) -> t.List[gws.ICrs]:
    """Return an intersection of crs supported by each source layer."""

    cs: set = set()

    for sl in layers:
        if not sl.supported_bounds:
            continue
        if not cs:
            cs.update([b.crs for b in sl.supported_bounds])
        else:
            cs = cs.intersection([b.crs for b in sl.supported_bounds])

    return list(cs)
