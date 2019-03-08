import gws.types as t
import gws.config
import gws.gis.proj
import gws.tools.misc as misc

from . import wms, wfs, wmts


def best_axis(crs, inverted_axis_crs_list, service_name, service_version):
    # inverted_axis_crs_list - list of EPSG crs'es which are known to have an inverted axis for this service
    # crs - crs we're going to use with the service

    proj = gws.gis.proj.as_proj(crs)
    if inverted_axis_crs_list and proj.epsg in inverted_axis_crs_list:
        return 'yx'

    # @TODO some logic to guess the axis, based on crs, service_name and service_version
    # see https://docs.geoserver.org/latest/en/user/services/wfs/basics.html#wfs-basics-axis
    return 'xy'


def best_crs(target_crs, supported_crs):
    for crs in supported_crs:
        if gws.gis.proj.equal(crs, target_crs):
            return target_crs

    for crs in supported_crs:
        p = gws.gis.proj.as_proj(crs)
        if p and not p.is_latlong:
            gws.log.info(f'best_crs: using {p.epsg!r} for {target_crs!r}')
            return p.epsg

    raise ValueError(f'no match for {target_crs!r} in {supported_crs!r}')


def crs_and_shape(request_crs, supported_crs, shape):
    crs = gws.gis.proj.find(request_crs, supported_crs)
    if crs:
        gws.log.debug(f'CRS: found {crs!r} for {request_crs!r}')
        return crs, shape

    crs = supported_crs[0]
    gws.log.debug(f'CRS: use {crs!r} for {request_crs!r}')
    return crs, shape.transform(crs)


def shared_service(type, obj, cfg):
    type = type.upper()
    klass = None

    if type == 'WMS':
        klass = wms.Service
    if type == 'WFS':
        klass = wfs.Service
    if type == 'WMTS':
        klass = wmts.Service

    if not klass:
        raise ValueError(f'unknown service type {type!r}')

    url = cfg.get('url')
    uid = url
    params = cfg.get('params')
    if params:
        uid += '_' + misc.sha256(' '.join(f'{k}={v}' for k, v in sorted(params.items())))

    return obj.create_shared_object(klass, uid, t.Config({
        'uid': uid,
        'url': url,
        'params': params,
        'capsCacheMaxAge': cfg.get('capsCacheMaxAge')
    }))
