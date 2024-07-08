from typing import Optional

import gws
import gws.gis.extent
import gws.lib.uom as units

OSM_SCALES = [
    500_000_000,
    250_000_000,
    150_000_000,
    70_000_000,
    35_000_000,
    15_000_000,
    10_000_000,
    4_000_000,
    2_000_000,
    1_000_000,
    500_000,
    250_000,
    150_000,
    70_000,
    35_000,
    15_000,
    8_000,
    4_000,
    2_000,
    1_000,
    500,
]
"""Scales corresponding to OSM zoom levels. (https://wiki.openstreetmap.org/wiki/Zoom_levels)"""

OSM_RESOLUTIONS = list(reversed([units.scale_to_res(s) for s in OSM_SCALES]))
"""Resolutions corresponding to OSM zoom levels."""


class Config(gws.Config):
    """Zoom levels and resolutions"""

    resolutions: Optional[list[float]]
    """allowed resolutions"""
    initResolution: Optional[float]
    """initial resolution"""

    scales: Optional[list[float]]
    """allowed scales"""
    initScale: Optional[float]
    """initial scale"""

    minResolution: Optional[float]
    """minimal resolution"""
    maxResolution: Optional[float]
    """maximal resolution"""

    minScale: Optional[float]
    """minimal scale"""
    maxScale: Optional[float]
    """maximal scale"""


def resolutions_from_config(cfg, parent_resolutions: list[float] = None) -> list[float]:
    """Loads resolution from a config.

    Args:
        cfg: A config.
        parent_resolutions: List of parent resolutions.

    Returns:
        A list of resolutions.
    """

    # see also https://mapproxy.org/docs/1.11.0/configuration.html#res and below

    # @TODO deal with scales separately

    rmin = _res_or_scale(cfg, 'minResolution', 'minScale')
    rmax = _res_or_scale(cfg, 'maxResolution', 'maxScale')

    res = _explicit_resolutions(cfg) or parent_resolutions or list(OSM_RESOLUTIONS)
    if rmax and rmax < max(res):
        res = [r for r in res if r < rmax]
        res.append(rmax)
    if rmin and rmin > min(res):
        res = [r for r in res if r > rmin]
        res.append(rmin)

    return sorted(set(res))


def resolutions_from_source_layers(source_layers: list[gws.SourceLayer], parent_resolutions: list[float]) -> list[
    float]:
    """Loads resolution from a source layers.

    Args:
        source_layers: Source layers.
        parent_resolutions: List of parent resolutions.

    Returns:
        A list of resolutions.
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

    rmin = units.scale_to_res(min(smin))
    rmax = units.scale_to_res(max(smax))

    pmin = min(parent_resolutions)
    pmax = max(parent_resolutions)

    if rmin > pmax or rmax < pmin:
        return []

    lt = [r for r in parent_resolutions if r <= rmin]
    gt = [r for r in parent_resolutions if r >= rmax]

    rmin = max(lt) if lt else pmin
    rmax = min(gt) if gt else pmax

    return [r for r in parent_resolutions if rmin <= r <= rmax]


def resolutions_from_bounds(b: gws.Bounds, tile_size: int) -> list[float]:
    """Loads resolutions from bounds.

    Args:
        b: Bounds object.
        tile_size: The tile size.

    Returns:
        A list of resolutions.
    """

    siz = gws.gis.extent.size(b.extent)
    res = []
    for z in range(20):
        res.append(siz[0] / (tile_size * (1 << z)))
    return res


def init_resolution(cfg, resolutions: list) -> float:
    """Returns the initial resolution.

    Args:
        cfg: A config.
        resolutions: List of Resolutions.
        """
    init = _res_or_scale(cfg, 'initResolution', 'initScale')
    if not init:
        return resolutions[len(resolutions) >> 1]
    return min(resolutions, key=lambda r: abs(init - r))


def _explicit_resolutions(cfg):
    ls = gws.u.get(cfg, 'resolutions')
    if ls:
        return ls

    ls = gws.u.get(cfg, 'scales')
    if ls:
        return [units.scale_to_res(x) for x in ls]


def _res_or_scale(cfg, r, s):
    x = gws.u.get(cfg, r)
    if x:
        return x
    x = gws.u.get(cfg, s)
    if x:
        return units.scale_to_res(x)
