import gws
import gws.config
import gws.lib.net
import gws.types as t


class ServiceException(Exception):
    pass


_base_url = None


def _call(service, params):
    global _base_url

    if not _base_url:
        _base_url = gws.config.root().app.mpx_url

    try:
        resp = gws.lib.net.http_request(_base_url + '/' + service, params=params)
        if resp.content_type.startswith('image'):
            return resp.content
        raise ServiceException(resp.text)
    except Exception:
        gws.log.exception()
        return


def wms_request(layer_uid, bounds: gws.Bounds, width, height, forward=None):
    args = {
        'bbox': bounds.extent,
        'width': width,
        'height': height,
        'crs': bounds.crs.epsg,
        'service': 'WMS',
        'request': 'GetMap',
        'version': '1.3.0',
        'format': 'image/png',
        'transparent': 'true',
        'styles': '',
        'layers': layer_uid
    }
    if forward:
        args.update(forward)
    return _call('wms', args)


def wmts_request(source_uid, x, y, z, tile_matrix, tile_size):
    args = {
        'tilecol': x,
        'tilerow': y,
        'tilematrix': z,
        'service': 'WMTS',
        'request': 'GetTile',
        'version': '1.0.0',
        'format': 'image/png',
        'tilematrixset': tile_matrix,
        'style': 'default',
        'layer': source_uid
    }
    return _call('ows', args)
