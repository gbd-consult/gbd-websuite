import time
import os

import gws
import gws.config
import gws.lib.net



class ServiceException(Exception):
    pass


def _call(service, params):
    url = getattr(gws.config.root().app, 'mpxUrl') + '/' + service
    resp = gws.lib.net.http_request(url, params=params)
    if resp.content_type.startswith('image'):
        return resp.content
    raise ServiceException(resp.text)


_retry_count = 3
_retry_pause = 5
_request_number = 0


def _call_with_retry(service, params):
    global _request_number

    _request_number += 1
    rc = 0

    while True:
        try:
            return _call(service, params)
        except Exception as exc:
            gws.log.exception()
            err = repr(exc)

        if rc >= _retry_count:
            gws.log.error(f'MAPPROXY_ERROR: {_request_number}/{rc} FAILED error={err} {params!r}')
            return

        gws.log.error(f'MAPPROXY_ERROR: {_request_number}/{rc} retry error={err}')
        time.sleep(_retry_pause)
        rc += 1


def wms_request(layer_uid, bounds: gws.Bounds, width, height, forward=None):
    mpx_no_transparency = os.getenv('GWS_MPX_NO_TRANSPARENCY', '0') == '1'
    params = {
        'bbox': bounds.extent,
        'width': width,
        'height': height,
        'crs': bounds.crs.epsg,
        'service': 'WMS',
        'request': 'GetMap',
        'version': '1.3.0',
        'format': 'image/png',
        'transparent': 'false' if mpx_no_transparency else 'true',
        'styles': '',
        'layers': layer_uid
    }
    if forward:
        params.update(forward)
    return _call_with_retry('wms', params)


def wmts_request(source_uid, x, y, z, tile_matrix, tile_size):
    params = {
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
    return _call_with_retry('ows', params)
