"""Zoom levels of maps and layers.

A map has a list of resolutions (map units per pixel), its zoom levels.
Layers use a subset of the map resolutions. This package computes these
lists from the ``zoom`` configuration of maps and layers, from source layer
scale hints, and converts between scales and resolutions.

Zoom levels can be configured in three ways:

- as levels of the standard tile grid of the map CRS
  (``gws.lib.grid.for_crs``), bounded by ``minLevel`` / ``maxLevel`` or by
  ``minScale`` / ``maxScale``. Without explicit bounds, the map gets levels
  0 to ``DEFAULT_MAX_LEVEL``.
- as an explicit list of ``scales`` (scale denominators).
- with the deprecated ``resolutions``, ``initResolution``,
  ``minResolution`` and ``maxResolution`` options, which still work and
  produce a configuration warning (``warn_deprecated_options``).

Levels are indices into the resolution list, 0 being the coarsest. Scale
bounds snap to the nearest available resolution. For a layer, the levels
are map-wide indices, and its resolutions are always taken from the map
resolutions. The initial map resolution is given by ``initLevel`` or
``initScale``, by default it is the middle of the list.

Scales and resolutions are converted with the OGC standard pixel size. For
geographic CRS, resolutions are in degrees per pixel, converted with
``gws.lib.crs.METERS_PER_DEGREE``. Scales must be between ``MIN_SCALE``
and ``MAX_SCALE``, levels between 0 and ``MAX_LEVEL``; otherwise a
``gws.ConfigurationError`` is raised.

Example::

    map.zoom { minLevel 6 maxLevel 19 initScale 50000 }

    map.layers+ {
        type "wms"
        provider.url "https://example.com/wms"
        zoom { maxScale 100000 }
    }

Python usage example::

    resolutions = gws.gis.zoom.resolutions_from_config(cfg, crs=gws.lib.crs.WEBMERCATOR)
    init = gws.gis.zoom.init_resolution(cfg, resolutions, crs=gws.lib.crs.WEBMERCATOR)
    scale = gws.gis.zoom.res_to_scale(init, gws.lib.crs.WEBMERCATOR)
"""

from typing import Optional

import math

import gws
import gws.lib.crs
import gws.lib.grid
import gws.lib.uom as units

DEFAULT_MAX_LEVEL = 20
"""Default finest level for grid-based resolutions."""

MAX_LEVEL = 30
"""Highest allowed zoom level."""

MIN_SCALE = 1
"""Smallest allowed scale denominator."""

MAX_SCALE = 500_000_000
"""Largest allowed scale denominator."""


class Config(gws.Config):
    """Zoom levels of a map or layer, given as scales or levels."""

    scales: Optional[list[float]]
    """Scale denominators of the zoom levels."""
    initScale: Optional[float]
    """Scale to open the map at, snapped to the nearest zoom level."""
    minScale: Optional[float]
    """Smallest scale denominator, snapped to the nearest zoom level."""
    maxScale: Optional[float]
    """Largest scale denominator, snapped to the nearest zoom level."""
    initLevel: Optional[int]
    """Zoom level to open the map at. (added in 8.5)"""
    minLevel: Optional[int]
    """Coarsest zoom level. (added in 8.5)"""
    maxLevel: Optional[int]
    """Finest zoom level. (added in 8.5)"""
    resolutions: Optional[list[float]]
    """Allowed resolutions. (deprecated in 8.5)"""
    initResolution: Optional[float]
    """Initial resolution. (deprecated in 8.5)"""
    minResolution: Optional[float]
    """Min. resolution. (deprecated in 8.5)"""
    maxResolution: Optional[float]
    """Max. resolution. (deprecated in 8.5)"""


_DEPRECATED_OPTIONS = {
    'resolutions': '"zoom.scales"',
    'initResolution': '"zoom.initScale" or "zoom.initLevel"',
    'minResolution': '"zoom.minScale" or "zoom.minLevel"',
    'maxResolution': '"zoom.maxScale" or "zoom.maxLevel"',
}


def warn_deprecated_options(cfg, root: gws.Root):
    """Register a configuration warning for each deprecated zoom option in use.

    Args:
        cfg: A zoom config.
        root: Configuration root.
    """

    for k, v in _DEPRECATED_OPTIONS.items():
        if gws.u.get(cfg, k) is not None:
            root.config_warning(f'"zoom.{k}" is deprecated, use {v}')


def resolutions_from_config(cfg, crs: gws.Crs = None) -> list[float]:
    """Compute map resolutions from a zoom config.

    An explicit ``scales`` (or deprecated ``resolutions``) list is taken as is;
    otherwise the resolutions of the standard grid for the CRS are used. Level bounds
    are indices into the list (0 = coarsest); scale bounds snap to the nearest entry.
    For the grid default, the bounds drive generation, so a ``minScale``
    beyond the default range extends the list.

    Args:
        cfg: A zoom config.
        crs: CRS of the map, Web Mercator if not given.

    Returns:
        A list of resolutions, sorted ascending.

    Raises:
        gws.ConfigurationError: If a value is out of bounds or the result is empty.
    """

    dsc = _explicit_resolutions(cfg, crs)
    if dsc:
        dsc = sorted(set(dsc), reverse=True)
        lo, hi = _index_bounds(cfg, dsc, crs)
        dsc = dsc[lo:hi + 1]
    else:
        dsc = _ladder_resolutions(cfg, crs)

    if not dsc:
        raise gws.ConfigurationError(f'empty resolutions {cfg!r}')
    return sorted(dsc)


def resolutions_for_layer(cfg, parent_resolutions: list[float], crs: gws.Crs = None) -> list[float]:
    """Compute layer resolutions from a zoom config.

    The result is always a subset of the parent (map) resolutions:
    ``scales`` entries snap to the nearest parent resolution; deprecated
    ``resolutions`` entries must match parent resolutions; level and scale
    bounds select from the parent list (levels are map-wide indices).

    Args:
        cfg: A zoom config.
        parent_resolutions: Parent (map) resolutions.
        crs: Map CRS, for scale conversions.

    Returns:
        A list of resolutions, sorted ascending.

    Raises:
        gws.ConfigurationError: If a value is invalid or the bounds select no resolutions.
    """

    pdsc = sorted(parent_resolutions, reverse=True)

    ls = gws.u.get(cfg, 'scales')
    if ls:
        idx = sorted(set(_nearest_index(pdsc, scale_to_res(s, crs)) for s in ls))
        dsc = [pdsc[i] for i in idx]
    else:
        ls = gws.u.get(cfg, 'resolutions')
        if ls:
            dsc = [pdsc[_exact_index(pdsc, r)] for r in ls]
            dsc = sorted(set(dsc), reverse=True)
        else:
            dsc = pdsc

    lo, hi = _index_bounds(cfg, pdsc)
    dsc = [r for r in dsc if pdsc[lo] >= r >= pdsc[hi]]

    return sorted(dsc)


def resolutions_from_source_layers(source_layers: list[gws.SourceLayer], parent_resolutions: list[float], crs: gws.Crs = None) -> list[float]:
    """Compute layer resolutions from source layer scale hints.

    The union of the scale ranges of all source layers acts as scale bounds over
    the parent resolutions, see ``resolutions_from_scale_range``. If any source layer
    has no scale range, the parent resolutions are returned.

    Args:
        source_layers: Source layers.
        parent_resolutions: Parent (map) resolutions.
        crs: Map CRS, for scale conversions.

    Returns:
        A list of resolutions, sorted ascending.
    """

    smin = []
    smax = []

    for sl in source_layers:
        sr = sl.scaleRange
        if not sr:
            return parent_resolutions
        smin.append(sr[0])
        smax.append(sr[1])

    if not smin:
        return parent_resolutions

    return resolutions_from_scale_range(min(smin), max(smax), parent_resolutions, crs)


def resolutions_from_scale_range(smin: float, smax: float, parent_resolutions: list[float], crs: gws.Crs = None) -> list[float]:
    """Compute layer resolutions from a scale range.

    The range bounds snap to the nearest parent resolutions and select the range between.

    Args:
        smin: Min. scale denominator.
        smax: Max. scale denominator.
        parent_resolutions: Parent (map) resolutions.
        crs: Map CRS, for scale conversions.

    Returns:
        A list of resolutions, sorted ascending. Empty if the range lies outside the parent resolutions.
    """

    rmin = scale_to_res(smin, crs)
    rmax = scale_to_res(smax, crs)

    pdsc = sorted(parent_resolutions, reverse=True)
    if rmin > pdsc[0] or rmax < pdsc[-1]:
        return []

    lo = _nearest_index(pdsc, rmax)
    hi = _nearest_index(pdsc, rmin)
    return sorted(pdsc[lo:hi + 1])


def init_resolution(cfg, resolutions: list, crs: gws.Crs = None) -> float:
    """Compute the initial resolution of a map.

    ``initLevel`` (an index, 0 = coarsest) wins over ``initScale``, which
    snaps to the nearest resolution; the default is the middle of the list.

    Args:
        cfg: A zoom config.
        resolutions: Map resolutions.
        crs: Map CRS, for scale conversions.

    Returns:
        One of the given resolutions.

    Raises:
        gws.ConfigurationError: If a value is out of bounds.
    """

    dsc = sorted(resolutions, reverse=True)

    lvl = _checked_level(cfg, 'initLevel')
    if lvl is not None:
        return dsc[min(max(lvl, 0), len(dsc) - 1)]

    init = _res_or_scale(cfg, 'initResolution', 'initScale', crs)
    if not init:
        return dsc[len(dsc) >> 1]
    return min(dsc, key=lambda r: abs(init - r))


def _ladder_resolutions(cfg, crs: gws.Crs = None) -> list[float]:
    mg = gws.lib.grid.for_crs(crs or gws.lib.crs.WEBMERCATOR)

    lo = _checked_level(cfg, 'minLevel') or 0
    rmax = _res_or_scale(cfg, 'maxResolution', 'maxScale', crs)
    if rmax:
        lo = max(lo, _nearest_level(mg, rmax))

    hi = _checked_level(cfg, 'maxLevel')
    rmin = _res_or_scale(cfg, 'minResolution', 'minScale', crs)
    if rmin:
        z = _nearest_level(mg, rmin)
        hi = z if hi is None else min(hi, z)
    if hi is None:
        hi = DEFAULT_MAX_LEVEL
    hi = min(hi, MAX_LEVEL)

    if lo > hi:
        raise gws.ConfigurationError(f'empty resolutions {cfg!r}')

    return [gws.lib.grid.resolution_for_level(mg, z) for z in range(lo, hi + 1)]


def _index_bounds(cfg, dsc: list[float], crs: gws.Crs = None) -> tuple[int, int]:
    n = len(dsc)

    lo = _checked_level(cfg, 'minLevel') or 0
    lo = min(max(lo, 0), n - 1)
    rmax = _res_or_scale(cfg, 'maxResolution', 'maxScale', crs)
    if rmax:
        lo = max(lo, _nearest_index(dsc, rmax))

    hi = _checked_level(cfg, 'maxLevel')
    hi = n - 1 if hi is None else min(max(hi, 0), n - 1)
    rmin = _res_or_scale(cfg, 'minResolution', 'minScale', crs)
    if rmin:
        hi = min(hi, _nearest_index(dsc, rmin))

    if lo > hi:
        raise gws.ConfigurationError(f'empty resolutions {cfg!r}')
    return lo, hi


def _nearest_index(dsc: list[float], res: float) -> int:
    return min(range(len(dsc)), key=lambda i: abs(dsc[i] - res))


def _nearest_level(mg: gws.MapGrid, res: float) -> int:
    z = gws.lib.grid.level_for_resolution(mg, res)
    if z > 0:
        r1 = gws.lib.grid.resolution_for_level(mg, z - 1)
        r2 = gws.lib.grid.resolution_for_level(mg, z)
        if abs(r1 - res) < abs(r2 - res):
            return z - 1
    return z


def _exact_index(dsc: list[float], res: float) -> int:
    for i, r in enumerate(dsc):
        if math.isclose(r, res, rel_tol=1e-6):
            return i
    raise gws.ConfigurationError(f'resolution {res!r} is not a map resolution')


def _explicit_resolutions(cfg, crs: gws.Crs = None):
    ls = gws.u.get(cfg, 'scales')
    if ls:
        return [_checked_res(scale_to_res(x, crs), x, crs) for x in ls]
    ls = gws.u.get(cfg, 'resolutions')
    if ls:
        return [_checked_res(x, x, crs) for x in ls]


def _res_or_scale(cfg, r, s, crs: gws.Crs = None):
    x = gws.u.get(cfg, r)
    if x:
        return _checked_res(x, x, crs)
    x = gws.u.get(cfg, s)
    if x:
        return _checked_res(scale_to_res(x, crs), x, crs)


def _checked_level(cfg, key):
    v = gws.u.get(cfg, key)
    if v is None:
        return None
    if not (0 <= v <= MAX_LEVEL):
        raise gws.ConfigurationError(f'invalid {key}: {v!r}')
    return v


def _checked_res(res, value, crs: gws.Crs = None):
    if not (scale_to_res(MIN_SCALE, crs) <= res <= scale_to_res(MAX_SCALE, crs)):
        raise gws.ConfigurationError(f'scale/resolution out of bounds: {value!r}')
    return res


def scale_to_res(scale: float, crs: gws.Crs = None) -> float:
    """Convert a scale denominator to a resolution.

    Args:
        scale: Scale denominator.
        crs: CRS. For a geographic CRS the result is in degrees per pixel, otherwise in meters per pixel.

    Returns:
        Resolution in map units per pixel.
    """

    return units.scale_to_res(scale) / _meters_per_unit(crs)


def res_to_scale(res: float, crs: gws.Crs = None) -> int:
    """Convert a resolution to a scale denominator.

    Args:
        res: Resolution in map units per pixel.
        crs: CRS. For a geographic CRS the resolution is in degrees per pixel, otherwise in meters per pixel.

    Returns:
        Scale denominator.
    """

    return units.res_to_scale(res * _meters_per_unit(crs))


def _meters_per_unit(crs: gws.Crs = None) -> float:
    if crs and crs.isGeographic:
        return gws.lib.crs.METERS_PER_DEGREE
    return 1.0
