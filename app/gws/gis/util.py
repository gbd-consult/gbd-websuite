import gws.gis.proj


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
