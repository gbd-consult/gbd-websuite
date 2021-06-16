import io

from PIL import Image

import gws
import gws.config
import gws.lib.net

import gws.types as t

class ServiceException(Exception):
    pass


def _call(service, params):
    url = 'http://%s:%s/%s' % (
        gws.config.root().var('server.mapproxy.host'),
        gws.config.root().var('server.mapproxy.port'),
        service
    )

    try:
        resp = gws.lib.net.http_request(url, params=params)
        if resp.content_type.startswith('image'):
            return resp.content
        text = resp.text
        if 'Exception' in text:
            raise ServiceException(text)
    except gws.lib.net.Error as e:
        gws.log.error('mapproxy http error', e)
        return
    except ServiceException as e:
        gws.log.error('mapproxy service exception', e)
        return


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
