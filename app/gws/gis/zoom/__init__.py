from typing import Optional

import gws
import gws.gis.extent
import gws.lib.uom as units

# https://wiki.openstreetmap.org/wiki/Zoom_levels

OSM_SCALES = [
    150000000,
    70000000,
    35000000,
    15000000,
    10000000,
    4000000,
    2000000,
    1000000,
    500000,
    250000,
    150000,
    70000,
    35000,
    15000,
    8000,
    4000,
    2000,
    1000,
    500,
]

OSM_RESOLUTIONS = list(reversed([units.scale_to_res(s) for s in OSM_SCALES]))


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
    # see also https://mapproxy.org/docs/1.11.0/configuration.html#res and below

    # @TODO deal with scales separately

    rmin = _res_or_scale(cfg, 'minResolution', 'minScale')
    rmax = _res_or_scale(cfg, 'maxResolution', 'maxScale')

    res = _explicit_resolutions(cfg) or parent_resolutions
    if not res:
        res = list(OSM_RESOLUTIONS)
        if rmax and rmax > max(res):
            res.append(rmax)
        if rmin and rmin < min(res):
            res.append(rmin)

    if rmin:
        res = [r for r in res if r >= rmin]
    if rmax:
        res = [r for r in res if r <= rmax]

    return sorted(res)


def resolutions_from_source_layers(source_layers: list[gws.SourceLayer], parent_resolutions: list[float]) -> list[float]:
    smin = []
    smax = []

    for sl in source_layers:
        sr = sl.scaleRange
        if sr:
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
    siz = gws.gis.extent.size(b.extent)
    res = []
    for z in range(20):
        res.append(siz[0] / (tile_size * (1 << z)))
    return res


def init_resolution(cfg, resolutions):
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
