import gws
import gws.types as t
import gws.lib.proj


def make_bbox(x, y, crs, resolution, pixel_width, pixel_height):
    """Given a point (in crs units), make a widthXheight pixel bbox around it."""

    # @TODO

    # is_geographic = gws.lib.proj.is_geographic(crs)
    #
    # if is_geographic:
    #     x, y = gws.lib.proj.transform_xy(x, y, crs, 'EPSG:3857')

    bbox = [
        x - (pixel_width * resolution) / 2,
        y - (pixel_height * resolution) / 2,
        x + (pixel_width * resolution) / 2,
        y + (pixel_height * resolution) / 2,
    ]

    # if is_geographic:
    #     bbox = gws.lib.proj.transform_extent(bbox, 'EPSG:3857', crs)

    return bbox


def invert_bbox(bbox):
    return [bbox[1], bbox[0], bbox[3], bbox[2]]


def best_axis(crs, inverted_axis_crs_list, service_name, service_version):
    # inverted_axis_crs_list - list of EPSG crs'es which are known to have an inverted axis for this service
    # crs - crs we're going to use with the service

    proj = gws.lib.proj.as_proj(crs)
    if inverted_axis_crs_list and proj.epsg in inverted_axis_crs_list:
        return 'yx'

    # @TODO some logic to guess the axis, based on crs, service_name and service_version
    # see https://docs.geoserver.org/latest/en/user/services/wfs/basics.html#wfs-basics-axis
    return 'xy'


def best_crs(target_crs, supported_crs):
    # find the best matching crs for the target crs and the list of supported crs.

    # if target_crs is in the list, we're fine

    crs = gws.lib.proj.find(target_crs, supported_crs)
    if crs:
        return crs

    # @TODO find a projection with less errors

    # if webmercator is supported, use it

    crs = gws.lib.proj.find(gws.EPSG_3857, supported_crs)
    if crs:
        gws.log.debug(f'best_crs: using {crs!r} for {target_crs!r}')
        return crs

    # return first non-geographic CRS

    for crs in supported_crs:
        p = gws.lib.proj.as_proj(crs)
        if p and not p.is_geographic:
            gws.log.debug(f'best_crs: using {p.epsg!r} for {target_crs!r}')
            return p.epsg

    raise ValueError(f'no match for {target_crs!r} in {supported_crs!r}')


def best_crs_and_shape(request_crs, supported_crs, shape):
    crs = best_crs(request_crs, supported_crs)
    return crs, shape.transformed_to(crs)
