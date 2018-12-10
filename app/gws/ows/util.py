import gws.types as t
import gws.config
import gws.gis.proj
import gws.tools.misc as misc

from . import wms, wfs, wmts


# p = self.var('invertAxis')
# if p is None:
#     # https://docs.geoserver.org/latest/en/user/services/wfs/basics.html#wfs-basics-axis
#     p = self.service.default_crs.startswith('urn') and self.service.version >= '1.1.0'
# self.service.has_inverted_axis = p


def axis_for(axis, service_name, service_version, crs):
    if axis:
        return axis
    return 'xy'


def crs_for_object(obj, supported_crs):
    crs = _crs_for_object(obj, supported_crs)
    gws.log.info(f'using {crs!r} for {obj.uid!r}')
    return crs


def _crs_for_object(obj, supported_crs):
    if not supported_crs:
        raise gws.config.LoadError(f'no supported_crs')

    # must have this

    crs = obj.var('crs')
    if crs:
        cf = gws.gis.proj.find(crs, supported_crs)
        if cf:
            return cf
        raise gws.config.LoadError(f'CRS {crs!r} not found in {supported_crs!r}')

    # nice to have this

    crs = obj.var('crs', parent=True)
    if crs:
        cf = gws.gis.proj.find(crs, supported_crs)
        if cf:
            return cf

    return supported_crs[0]


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
