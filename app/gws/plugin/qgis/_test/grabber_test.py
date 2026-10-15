"""Tests for the QGIS grabber, without QGIS Server."""

import gws
import gws.test.util as u
import gws.base.grabber.core as core
import gws.lib.crs
import gws.lib.extent
import gws.lib.image
import gws.plugin.qgis.grabber as grabber

CRS_3857 = gws.lib.crs.get(3857)


def _opts(image_format):
    cache = gws.MapCache(name='grabber_' + gws.u.random_string(8), maxAge=0, maxLevel=20, requestBuffer=0, requestTiles=1)
    return core.Options(
        crs=CRS_3857,
        cache=cache,
        extent=gws.lib.extent.transform_from_wgs(CRS_3857.wgsMaxExtent, CRS_3857),
        imageFormat=image_format,
    )


def _grabber(image_format, calls=None):
    prov = gws.Data(get_map=lambda bounds, w, h, params, mime_type: calls.append((params, mime_type)) or b'png_1')
    return grabber.Object(_opts(image_format), provider=prov, params={'LAYERS': ['layer_1']})


def test_source_mime_types():
    assert _grabber(gws.lib.image.PNG8).sourceMimeType == 'image/png; mode=8bit'
    assert _grabber(gws.lib.image.PNG24).sourceMimeType == 'image/png'
    assert _grabber(gws.lib.image.JPEG).sourceMimeType == 'image/jpeg'
    assert _grabber(gws.lib.image.WEBP).sourceMimeType == 'image/webp'


def test_source_mime_type_unsupported():
    ifo = gws.ImageFormat(name='name_1', mimeTypes=[gws.lib.mime.GIF], options={})
    assert _grabber(ifo).sourceMimeType == ''


def test_fetch_box_sends_source_mime_type_and_params():
    calls = []
    gr = _grabber(gws.lib.image.PNG8, calls)
    gr.fetch_box_as_bytes(gws.Bounds(crs=CRS_3857, extent=(0, 0, 100, 100)), 10, 10, {'FILTER': 'filter_1'})
    assert calls == [({'LAYERS': ['layer_1'], 'FILTER': 'filter_1'}, 'image/png; mode=8bit')]
