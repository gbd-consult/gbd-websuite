"""Tests for image formats of the WMS service."""

import gws
import gws.test.util as u
import gws.lib.image
import gws.lib.xmlx


@u.fixture(scope='module')
def root():
    cfg = """
        permissions.all "allow all"

        actions [
            { type "ows" }
        ]

        owsServices+ {
            type "wms"
            uid "WMS_1"
            imageFormats+ { name "PNG8" }
            imageFormats+ { name "PNG24" }
            imageFormats+ { name "JPEG" mimeTypes [ "Image/JPEG;Quality=low" "foo/bar" ] options { quality 10 } }
        }
        projects+ {
            uid "PROJECT_1"
            map.extent [0 0 100 200]
            map.srs "EPSG:3857"
            map.layers+ {
                uid "LAYER_1"
                title "LAYER_1"
                type "raster"
                provider.paths ["/gws-app/gws/plugin/ows_server/wms/_test/300x400.png"]
            }
        }
    """
    yield u.gws_root(cfg)


def _get(root, request, **params):
    return u.http.get(root, '/_/owsService', query_string={
        'request': request,
        'serviceUid': 'WMS_1',
        'projectUid': 'PROJECT_1',
        'version': '1.3.0',
        **params,
    })


def _get_map(root, fmt):
    return _get(root, 'GetMap', layers='layer_1', crs='EPSG:3857', bbox='0,0,100,200', width=30, height=60, format=fmt)


def _formats(root):
    el = gws.lib.xmlx.from_string(_get(root, 'GetCapabilities').get_data(), gws.XmlOptions(removeNamespaces=True))
    return el.textlist('Capability/Request/GetMap/Format')


def test_caps_formats_normalized(root):
    fs = _formats(root)
    assert 'image/png; mode=8bit' in fs
    assert 'image/png; mode=32bit' in fs
    assert 'image/jpeg; quality=low' in fs
    assert 'foo/bar' not in fs


def test_get_map_png8(root):
    res = _get_map(root, 'image/png8')
    assert res.status_code == 200
    assert gws.lib.image.from_bytes(res.get_data()).img.mode == 'P'


def test_get_map_png24_any_spelling(root):
    res = _get_map(root, 'IMAGE/PNG;MODE=32bit')
    assert res.status_code == 200
    assert gws.lib.image.from_bytes(res.get_data()).img.mode == 'RGBA'


def test_get_map_plain_png_is_first_format(root):
    res = _get_map(root, 'image/png')
    assert gws.lib.image.from_bytes(res.get_data()).img.mode == 'P'


def test_get_map_jpeg_with_params(root):
    res = _get_map(root, 'image/jpeg; quality=low')
    assert res.status_code == 200
    assert gws.lib.image.from_bytes(res.get_data()).img.format == 'JPEG'


def test_get_map_unknown_format(root):
    res = _get_map(root, 'foo/bar')
    assert b'InvalidFormat' in res.get_data()
