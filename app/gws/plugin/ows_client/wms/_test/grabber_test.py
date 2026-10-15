"""Tests for the WMS grabber, without a service."""

import gws
import gws.test.util as u
import gws.base.grabber.core as core
import gws.lib.crs
import gws.lib.extent
import gws.lib.image
import gws.plugin.ows_client.wms.grabber as grabber

CRS_3857 = gws.lib.crs.get(3857)


def _opts(image_format):
    cache = gws.MapCache(name='grabber_' + gws.u.random_string(8), maxAge=0, maxLevel=20, requestBuffer=0, requestTiles=1)
    return core.Options(
        crs=CRS_3857,
        cache=cache,
        extent=gws.lib.extent.transform_from_wgs(CRS_3857.wgsMaxExtent, CRS_3857),
        imageFormat=image_format,
    )


def _grabber(image_format, mime_types, calls=None):
    op = gws.OwsOperation(verb=gws.OwsVerb.GetMap, mimeTypes=mime_types)
    prov = gws.Data(
        maxRequestPixels=4096,
        get_operation=lambda verb: op if verb == gws.OwsVerb.GetMap else None,
        get_map=lambda bounds, w, h, source_layers, mime: calls.append(mime) or b'png_1',
    )
    return grabber.Object(_opts(image_format), provider=prov, sourceLayers=[gws.SourceLayer(name='layer_1')], sourceCrs=CRS_3857)


def test_source_mime_type_specific():
    gr = _grabber(gws.lib.image.PNG8, [gws.lib.mime.PNG, gws.lib.mime.PNG8])
    assert gr.sourceMimeType == gws.lib.mime.PNG8


def test_source_mime_type_plain():
    gr = _grabber(gws.lib.image.PNG24, [gws.lib.mime.PNG, gws.lib.mime.PNG8])
    assert gr.sourceMimeType == gws.lib.mime.PNG


def test_source_mime_type_no_match():
    gr = _grabber(gws.lib.image.JPEG, [gws.lib.mime.PNG])
    assert gr.sourceMimeType == ''


def test_fetch_box_sends_source_mime_type():
    calls = []
    gr = _grabber(gws.lib.image.PNG8, [gws.lib.mime.PNG, gws.lib.mime.PNG8], calls)
    assert gr.fetch_box_as_bytes(gws.Bounds(crs=CRS_3857, extent=(0, 0, 100, 100)), 10, 10) == b'png_1'
    assert calls == [gws.lib.mime.PNG8]
