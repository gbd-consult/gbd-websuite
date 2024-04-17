from typing import Optional

import re

import gws
import gws.gis.extent


##

class LayerFilter(gws.Data):
    """Source layer filter"""

    level: int = 0
    """match only layers at this level"""
    names: Optional[list[str]]
    """match these layer names (top-to-bottom order)"""
    titles: Optional[list[str]]
    """match these layer titles"""
    pattern: gws.Regex = ''
    """match layers whose full path matches a pattern"""
    isGroup: Optional[bool]
    """if true, match only group layers"""
    isImage: Optional[bool]
    """if true, match only images layers"""
    isQueryable: Optional[bool]
    """if true, match only queryable layers"""
    isVisible: Optional[bool]
    """if true, match only visible layers"""


def layer_matches(sl: gws.SourceLayer, f: LayerFilter) -> bool:
    """Check if a source layer matches the filter"""

    if not f:
        return True

    if f.level and sl.aLevel != f.level:
        return False

    if f.names and sl.name not in f.names:
        return False

    if f.titles and sl.title not in f.titles:
        return False

    if f.pattern and not re.search(f.pattern, sl.aPath):
        return False

    if f.isGroup is not None and sl.isGroup != f.isGroup:
        return False

    if f.isImage is not None and sl.isImage != f.isImage:
        return False

    if f.isQueryable is not None and sl.isQueryable != f.isQueryable:
        return False

    if f.isVisible is not None and sl.isVisible != f.isVisible:
        return False

    return True


def check_layers(layers: list[gws.SourceLayer], revert: bool = False) -> list[gws.SourceLayer]:
    """Insert our properties in the source layer tree.

    Also remove empty layers.

    Args:
        layers: List of source layers
        revert: Revert the order of layers and sub-layers.
    """

    def walk(sl, parent_path, level):
        if not sl:
            return
        sl.aUid = gws.u.to_uid(sl.name or sl.metadata.get('title'))
        sl.aPath = parent_path + '/' + sl.aUid
        sl.aLevel = level
        sl.layers = gws.u.compact(walk(c, sl.aPath, level + 1) for c in (sl.layers or []))
        if revert:
            sl.layers = list(reversed(sl.layers))
        return sl

    ls = gws.u.compact(walk(sl, '', 1) for sl in layers)
    if revert:
        ls = list(reversed(ls))
    return ls


def filter_layers(
        layers: list[gws.SourceLayer],
        slf: LayerFilter = None,
        is_group: bool = None,
        is_image: bool = None,
        is_queryable: bool = None,
        is_visible: bool = None,
) -> list[gws.SourceLayer]:
    """Filter source layers by the given layer filter."""

    extra = {}
    if is_group is not None:
        extra['isGroup'] = is_group
    if is_image is not None:
        extra['isImage'] = is_image
    if is_queryable is not None:
        extra['isQueryable'] = is_queryable
    if is_visible is not None:
        extra['isVisible'] = is_visible

    if slf:
        if extra:
            slf = LayerFilter(slf, extra)
    elif extra:
        slf = LayerFilter(extra)
    else:
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


def combined_crs_list(layers: list[gws.SourceLayer]) -> list[gws.Crs]:
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
