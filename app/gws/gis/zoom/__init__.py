import gws
import gws.lib.units as units
import gws.types as t

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

    resolutions: t.Optional[t.List[float]]  #: allowed resolutions
    initResolution: t.Optional[float]  #: initial resolution

    scales: t.Optional[t.List[float]]  #: allowed scales
    initScale: t.Optional[float]  #: initial scale

    minResolution: t.Optional[float]  #: minimal resolution
    maxResolution: t.Optional[float]  #: maximal resolution

    minScale: t.Optional[float]  #: minimal scale
    maxScale: t.Optional[float]  #: maximal scale


def resolutions_from_config(cfg, parent_resolultions: t.List[float] = None) -> t.List[float]:
    # see also https://mapproxy.org/docs/1.11.0/configuration.html#res and below

    # @TODO deal with scales separately

    res = _explicit_resolutions(cfg) or parent_resolultions
    if not res:
        return []

    a = _res_or_scale(cfg, 'minResolution', 'minScale')
    z = _res_or_scale(cfg, 'maxResolution', 'maxScale')

    if a:
        res = [x for x in res if x >= a]
    if z:
        res = [x for x in res if x <= z]

    return sorted(res)


def resolutions_from_source_layers(source_layers: t.List[gws.SourceLayer], parent_resolultions: t.List[float]) -> t.List[float]:
    smin = []
    smax = []

    for sl in source_layers:
        sr = sl.scaleRange
        if sr:
            smin.append(sr[0])
            smax.append(sr[1])

    if not smin:
        return parent_resolultions

    rmin = units.scale_to_res(min(smin))
    rmax = units.scale_to_res(max(smax))

    pmin = min(parent_resolultions)
    pmax = max(parent_resolultions)

    if rmin > pmax or rmax < pmin:
        return []

    lt = [r for r in parent_resolultions if r <= rmin]
    gt = [r for r in parent_resolultions if r >= rmax]

    rmin = max(lt) if lt else pmin
    rmax = min(gt) if gt else pmax

    return [r for r in parent_resolultions if rmin <= r <= rmax]


def init_resolution(cfg, resolutions):
    init = _res_or_scale(cfg, 'initResolution', 'initScale')
    if not init:
        return resolutions[len(resolutions) >> 1]
    return min(resolutions, key=lambda r: abs(init - r))


def _explicit_resolutions(cfg):
    ls = gws.get(cfg, 'resolutions')
    if ls:
        return ls

    ls = gws.get(cfg, 'scales')
    if ls:
        return [units.scale_to_res(x) for x in ls]


def _res_or_scale(cfg, r, s):
    x = gws.get(cfg, r)
    if x:
        return x
    x = gws.get(cfg, s)
    if x:
        return units.scale_to_res(x)
