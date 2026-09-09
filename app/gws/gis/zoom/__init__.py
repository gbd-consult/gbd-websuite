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
    """Zoom levels and resolutions"""

    scales: Optional[list[float]]
    """Allowed scales."""
    initScale: Optional[float]
    """Initial scale, snapped to the nearest resolution."""
    minScale: Optional[float]
    """Minimal scale, snapped to the nearest resolution."""
    maxScale: Optional[float]
    """Maximal scale, snapped to the nearest resolution."""
    initLevel: Optional[int]
    """Initial zoom level. (added in 8.5)"""
    minLevel: Optional[int]
    """Coarsest zoom level. (added in 8.5)"""
    maxLevel: Optional[int]
    """Finest zoom level. (added in 8.5)"""
    resolutions: Optional[list[float]]
    """Allowed resolutions. (deprecated in 8.5)"""
    initResolution: Optional[float]
    """Initial resolution. (deprecated in 8.5)"""
    minResolution: Optional[float]
    """Minimal resolution. (deprecated in 8.5)"""
    maxResolution: Optional[float]
    """Maximal resolution. (deprecated in 8.5)"""


def resolutions_from_config(cfg, crs: gws.Crs = None) -> list[float]:
    """Computes map resolutions from a config.

    An explicit ``scales`` (or deprecated ``resolutions``) list is taken as is;
    otherwise the grid ladder for the CRS is used. Level bounds are indices
    into the list (0 = coarsest); scale bounds snap to the nearest entry.
    For the ladder default, the bounds drive generation, so a ``minScale``
    beyond the default range extends the ladder.

    Args:
        cfg: A config.
        crs: CRS for the default resolutions.

    Returns:
        A list of resolutions, sorted ascending.
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
    """Computes layer resolutions from a config.

    The result is always a subset of the parent (map) resolutions:
    ``scales`` entries snap to the nearest parent resolution; deprecated
    ``resolutions`` entries must match parent resolutions; level and scale
    bounds select from the parent list (levels are map-wide indices).

    Args:
        cfg: A config.
        parent_resolutions: Parent (map) resolutions.
        crs: Map CRS, for scale conversions.

    Returns:
        A list of resolutions, sorted ascending.
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
    """Computes layer resolutions from source layer scale hints.

    The hints act as scale bounds over the parent resolutions: they snap to
    the nearest parent entries and select the range between.

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
    """Computes layer resolutions from a scale range.

    The range bounds snap to the nearest parent resolutions and select the range between.

    Args:
        smin: Min scale denominator.
        smax: Max scale denominator.
        parent_resolutions: Parent (map) resolutions.
        crs: Map CRS, for scale conversions.

    Returns:
        A list of resolutions, sorted ascending.
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
    """Returns the initial resolution.

    ``initLevel`` (an index, 0 = coarsest) wins over ``initScale``, which
    snaps to the nearest resolution; the default is the middle of the list.

    Args:
        cfg: A config.
        resolutions: List of resolutions.
        crs: Map CRS, for scale conversions.
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
    """Scale denominator to resolution in map units (degrees per pixel for geographic CRS)."""

    return units.scale_to_res(scale) / _meters_per_unit(crs)


def res_to_scale(res: float, crs: gws.Crs = None) -> int:
    """Resolution in map units to scale denominator (degrees per pixel for geographic CRS)."""

    return units.res_to_scale(res * _meters_per_unit(crs))


def _meters_per_unit(crs: gws.Crs = None) -> float:
    if crs and crs.isGeographic:
        return gws.lib.crs.METERS_PER_DEGREE
    return 1.0
