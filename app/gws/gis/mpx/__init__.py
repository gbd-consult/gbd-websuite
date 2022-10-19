import time

import gws
import gws.config
import gws.tools.net

import gws.types as t


class ServiceException(Exception):
    pass


def _call(service, params):
    url = 'http://%s:%s/%s' % (
        gws.config.root().var('server.mapproxy.host'),
        gws.config.root().var('server.mapproxy.port'),
        service
    )

    resp = gws.tools.net.http_request(url, params=params, timeout=0)
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
        err = None
        try:
            return _call(service, params)
        except Exception as exc:
            err = repr(exc)

        if rc >= _retry_count:
            gws.log.error(f'MAPPROXY_ERROR: {_request_number}/{rc} FAILED error={err} {params!r}')
            return

        gws.log.error(f'MAPPROXY_ERROR: {_request_number}/{rc} retry error={err}')
        time.sleep(_retry_pause)
        rc += 1


def wms_request(layer_uid, bounds: t.Bounds, width, height, forward=None):
    args = {
        'bbox': bounds.extent,
        'width': width,
        'height': height,
        'crs': bounds.crs,
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
    return _call_with_retry('wms', args)


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
    return _call_with_retry('ows', args)
