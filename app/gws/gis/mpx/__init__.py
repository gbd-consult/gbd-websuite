import io

from PIL import Image

import gws
import gws.config
import gws.tools.net

_error_color = (255, 0, 140, 0)


def _error_image(w, h):
    img = Image.new('RGBA', (w, h), _error_color)
    with io.BytesIO() as out:
        img.save(out, format='png')
        return out.getvalue()


class ServiceException(Exception):
    pass


def _call(service, params):
    url = 'http://%s:%s/%s' % (
        gws.config.root().var('server.mapproxy.host'),
        gws.config.root().var('server.mapproxy.port'),
        service
    )

    try:
        resp = gws.tools.net.http_request(url, params=params)
        if resp.content_type.startswith('image'):
            return resp.content
        text = resp.text
        if 'Exception' in text:
            raise ServiceException(text)
    except gws.tools.net.Error as e:
        gws.log.error('mapproxy http error', e)
        return
    except ServiceException as e:
        gws.log.error('mapproxy service exception', e)
        return


def wms_request(layer_uid, bbox, width, height, crs, forward=None):
    args = {
        'bbox': bbox,
        'width': width,
        'height': height,
        'crs': crs,
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
    return _call('wms', args) or _error_image(width, height)


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
    return _call('ows', args) or _error_image(tile_size, tile_size)
