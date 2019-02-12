import gws
import gws.tools.misc as misc
import gws.types as t


class Config(t.Config):
    """Zoom levels and resolutions"""

    resolutions: t.Optional[t.List[float]]  #: allowed resolutions
    initResolution: t.Optional[float]  #: initial resolution

    scales: t.Optional[t.List[float]]  #: allowed scales
    initScale: t.Optional[float]  #: initial scale

    minResolution: t.Optional[float]  #: minimal resolution
    maxResolution: t.Optional[float]  #: maximal resolution

    minScale: t.Optional[float]  #: minimal scale
    maxScale: t.Optional[float]  #: maximal scale


def effective_resolutions(cfg, parent_resolultions=None):
    # see also https://mapproxy.org/docs/1.11.0/configuration.html#res and below

    res = _explicit_resolutions(cfg) or parent_resolultions
    if not res:
        return []

    a = _res_or_scale(cfg, 'minResolution', 'minScale')
    z = _res_or_scale(cfg, 'maxResolution', 'maxScale')

    if a:
        res = [x for x in res if x >= a]
    if z:
        res = [x for x in res if x <= z]

    return sorted(res, reverse=True)


def init_resolution(cfg, resolutions):
    init = _res_or_scale(cfg, 'initResolution', 'initScale')
    if not init:
        return resolutions[0]
    return min(resolutions, key=lambda r: abs(init - r))


def _explicit_resolutions(cfg):
    ls = gws.get(cfg, 'resolutions')
    if ls:
        return ls

    ls = gws.get(cfg, 'scales')
    if ls:
        return [misc.scale2res(x) for x in ls]


def _res_or_scale(cfg, r, s):
    x = gws.get(cfg, r)
    if x:
        return x
    x = gws.get(cfg, s)
    if x:
        return misc.scale2res(x)
