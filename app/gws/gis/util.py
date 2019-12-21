import gws.gis.proj
import gws.tools.misc
import gws.types as t


def compute_bbox(x, y, crs, resolution, pixel_width, pixel_height):
    """Given a point (in crs units), compute a bbox around it."""

    # @TODO

    # is_latlong = gws.gis.proj.is_latlong(crs)
    #
    # if is_latlong:
    #     x, y = gws.gis.proj.transform_xy(x, y, crs, 'EPSG:3857')

    bbox = [
        x - (pixel_width * resolution) / 2,
        y - (pixel_height * resolution) / 2,
        x + (pixel_width * resolution) / 2,
        y + (pixel_height * resolution) / 2,
    ]

    # if is_latlong:
    #     bbox = gws.gis.proj.transform_bbox(bbox, 'EPSG:3857', crs)

    return bbox


def shared_ows_provider(klass, obj, cfg):
    url = cfg.get('url')
    uid = url
    params = cfg.get('params')
    if params:
        uid += '_' + gws.tools.misc.sha256(' '.join(f'{k}={v}' for k, v in sorted(params.items())))

    return obj.create_shared_object(klass, uid, t.Config({
        'uid': uid,
        'url': url,
        'params': params,
        'capsCacheMaxAge': cfg.get('capsCacheMaxAge')
    }))


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
    # find the best matching crs for the target crs and the list of supported crs.
    # if target_crs is in the list, we're fine, otherwise try to find a projected crs

    crs = gws.gis.proj.find(target_crs, supported_crs)
    if crs:
        return crs

    for crs in supported_crs:
        p = gws.gis.proj.as_proj(crs)
        if p and not p.is_latlong:
            gws.log.debug(f'best_crs: using {p.epsg!r} for {target_crs!r}')
            return p.epsg

    raise ValueError(f'no match for {target_crs!r} in {supported_crs!r}')


def best_crs_and_shape(request_crs, supported_crs, shape):
    crs = best_crs(request_crs, supported_crs)
    return crs, shape.transform(crs)
