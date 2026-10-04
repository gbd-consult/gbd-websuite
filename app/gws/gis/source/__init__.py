"""Source layers.

Source layers (``gws.SourceLayer``) describe the layers of an external
source, such as a WMS, WFS or WMTS service or a QGIS project, as read from
its capabilities. They form a tree. GWS layers that wrap a source select
the source layers they render or query with a ``LayerFilter``, e.g. in the
``sourceLayers`` option.

This package provides:

- ``check_layers``: prepares a source layer tree after parsing the
  capabilities. It assigns each layer a uid (``aUid``), a path of uids from
  the root (``aPath``) and a depth (``aLevel``, 1 for the top level), removes
  missing entries, optionally reverses the layer order, and derives a group's
  WGS extent from its children when it has none.
- ``LayerFilter``, ``layer_matches`` and ``filter_layers``: select source
  layers by depth, name, title, path pattern and flags. Filtering walks the
  tree and stops at the first matching layer of each branch.
- ``combined_crs_list``, ``combined_wgs_extent`` and ``combined_bounds``:
  combine the CRS lists and extents of several source layers.

The ``SourceLayer`` and ``SourceStyle`` types are defined in ``types.pyinc``.

Example::

    sls = gws.gis.source.check_layers(parsed_layers)
    roads = gws.gis.source.filter_layers(sls, gws.gis.source.LayerFilter(pattern='roads'), is_image=True)
    bounds = gws.gis.source.combined_bounds(roads, gws.lib.crs.WEBMERCATOR)

Configuration example, select source layers by name::

    sourceLayers.names ["streets" "buildings"]
"""

from typing import Optional, Iterable

import re

import gws
import gws.lib.bounds
import gws.lib.extent
import gws.lib.crs


##

class LayerFilter(gws.Data):
    """Selects source layers."""

    level: int = 0
    """Match only layers at this depth in the source tree, 1 being the top."""
    names: Optional[list[str]]
    """Match these layer names (top-to-bottom order)."""
    titles: Optional[list[str]]
    """Match these layer titles."""
    pattern: gws.Regex = ''
    """Match layers whose path matches a regular expression."""
    isGroup: Optional[bool]
    """Match only group layers."""
    isImage: Optional[bool]
    """Match only image layers."""
    isQueryable: Optional[bool]
    """Match only queryable layers."""
    isVisible: Optional[bool]
    """Match only visible layers."""


def layer_matches(sl: gws.SourceLayer, f: Optional[LayerFilter]) -> bool:
    """Check if a source layer matches a filter.

    All given conditions of the filter must match.

    Args:
        sl: Source layer.
        f: Layer filter. If ``None``, every layer matches.

    Returns:
        ``True`` if the layer matches.
    """

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


def check_layers(layers: Iterable[gws.SourceLayer], revert: bool = False) -> list[gws.SourceLayer]:
    """Prepare a source layer tree.

    Sets ``aUid``, ``aPath`` and ``aLevel`` on each layer, removes empty entries and sets
    the WGS extent of layers without one to the union of their children's extents.

    Args:
        layers: Top-level source layers.
        revert: Reverse the order of layers and sub-layers.

    Returns:
        The prepared top-level layers.
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

        if not sl.wgsExtent and sl.layers:
            exts = gws.u.compact(c.wgsExtent for c in sl.layers)
            if exts:
                sl.wgsExtent = gws.lib.extent.union(*exts)

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
    """Select source layers from a tree.

    The flag arguments are added to the filter. Each branch of the tree is searched until
    a matching layer is found; the sub-layers of a matching layer are not searched.
    If the filter has ``names``, the result is sorted in the order of ``names``.

    Args:
        layers: Top-level source layers.
        slf: Layer filter.
        is_group: Match only group layers, or only non-group layers if ``False``.
        is_image: Match only image layers, or only non-image layers if ``False``.
        is_queryable: Match only queryable layers, or only non-queryable layers if ``False``.
        is_visible: Match only visible layers, or only invisible layers if ``False``.

    Returns:
        Matching layers. If there are no filter conditions, ``layers`` is returned as is.
    """

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
    """Compute the CRS supported by all source layers.

    Layers without supported CRS are ignored.

    Args:
        layers: Source layers.

    Returns:
        The intersection of the supported CRS lists, in no particular order.
    """

    cs: set = set()

    for sl in layers:
        if not sl.supportedCrs:
            continue
        if not cs:
            cs.update(sl.supportedCrs)
        else:
            cs = cs.intersection(sl.supportedCrs)

    return list(cs)


def combined_wgs_extent(layers: list[gws.SourceLayer]) -> Optional[gws.Extent]:
    """Compute the union of the WGS extents of source layers.

    Args:
        layers: Source layers.

    Returns:
        The union extent, or ``None`` if no layer has an extent.
    """

    bs = gws.u.compact(sl.wgsExtent for sl in layers)
    if bs:
        return gws.lib.extent.union(*bs)


def combined_bounds(layers: list[gws.SourceLayer], crs: gws.Crs) -> Optional[gws.Bounds]:
    """Compute the union of the WGS extents of source layers, transformed to a CRS.

    Args:
        layers: Source layers.
        crs: Target CRS.

    Returns:
        Bounds in the target CRS, or ``None`` if no layer has an extent.
    """

    ext = combined_wgs_extent(layers)
    if ext:
        b = gws.Bounds(extent=ext, crs=gws.lib.crs.WGS84)
        return gws.lib.bounds.transform(b, crs)
