import re

import gws
import gws.gis.extent
import gws.types as t


##

class LayerFilterConfig(gws.Config):
    """Source layer filter"""

    level: int = 0  #: match only layers at this level
    names: t.Optional[t.List[str]]  #: match these layer names (top-to-bottom order)
    titles: t.Optional[t.List[str]]  #: match these layer titles
    pattern: gws.Regex = ''  #: match layers whose full path matches a pattern
    onlyGroups: bool = False  #: if true, match only group layers
    onlyLeaves: bool = False  #: if true, match only leaf layers


class LayerFilter(gws.Data):
    level: int
    names: t.List[str]
    titles: t.List[str]
    pattern: gws.Regex
    isGroup: bool
    isImage: bool
    isQueryable: bool


def layer_filter_from_config(cfg, **kwargs) -> t.Optional[LayerFilter]:
    if not cfg:
        return None

    slf = LayerFilter(
        level=cfg.level or 0,
        names=cfg.names or [],
        titles=cfg.titles or [],
        pattern=cfg.pattern,
        **kwargs
    )
    if cfg.onlyGroups:
        slf.isGroup = True
    if cfg.onlyLeaves:
        slf.isGroup = False
    return slf


def layer_matches(sl: gws.SourceLayer, slf: t.Optional[LayerFilter]) -> bool:
    """Check if a source layer matches the filter"""

    if not slf:
        return True

    if slf.level and sl.aLevel != slf.level:
        return False

    if slf.names and sl.name not in slf.names:
        return False

    if slf.titles and sl.title not in slf.titles:
        return False

    if slf.pattern and not re.search(slf.pattern, sl.aPath):
        return False

    if slf.isGroup is not None and sl.isGroup != slf.isGroup:
        return False

    if slf.isImage is not None and sl.isImage != slf.isImage:
        return False

    if slf.isQueryable is not None and sl.isQueryable != slf.isQueryable:
        return False

    return True


def check_layers(layers) -> t.List[gws.SourceLayer]:
    def walk(sl, parent_path, level):
        if not sl:
            return
        sl.aUid = gws.to_uid(sl.name or sl.metadata.get('title'))
        sl.aPath = parent_path + '/' + sl.aUid
        sl.aLevel = level
        sl.layers = gws.compact(walk(c, sl.aPath, level + 1) for c in (sl.layers or []))
        return sl

    return gws.compact(walk(sl, '', 1) for sl in layers)


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
        bb = gws.gis.crs.best_bounds(target_crs, sl.supported_bounds)
        ext = gws.gis.extent.transform(bb.extent, bb.crs, target_crs)
        exts.append(ext)

    if exts:
        return gws.Bounds(crs=target_crs, extent=gws.gis.extent.merge(exts))


def combined_crs_list(layers: t.List[gws.SourceLayer]) -> t.List[gws.ICrs]:
    """Return an intersection of crs supported by each source layer."""

    cs: set = set()

    for sl in layers:
        if not sl.supportedCrs:
            continue
        if not cs:
            cs.update(sl.supportedCrs)
        else:
            cs = cs.intersection(sl.supportedCrs)

    return list(cs)
