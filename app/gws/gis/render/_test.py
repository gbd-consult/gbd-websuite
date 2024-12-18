"""Tests for the render module."""

import gws
import gws.test.util as u
import gws.gis.render as render
import gws.lib.crs
import gws.lib.image


def test_map_view_from_center():
    size = (400.0, 400.0, gws.Uom.px)
    center = (100.0, 500.0)
    crs = gws.lib.crs.WGS84
    dpi = 1000
    rotation = 0
    assert render.map_view_from_center(size, center, crs, dpi, rotation).dpi == 1000
    assert render.map_view_from_center(size, center, crs, dpi, rotation).pxSize == (400.0, 400.0)
    assert render.map_view_from_center(size, center, crs, dpi, rotation).rotation == 0
    assert render.map_view_from_center(size, center, crs, dpi, rotation).mmSize == (10.16, 10.16)
    assert render.map_view_from_center(size, center, crs, dpi, rotation).center == (100.0, 500.0)
    assert render.map_view_from_center(size, center, crs, dpi, rotation).scale == 0
    assert render.map_view_from_center(size, center, crs, dpi, rotation).bounds.crs == crs
    assert render.map_view_from_center(size, center, crs, dpi, rotation).bounds.extent == (100.0, 500.0, 100.0, 500.0)


def test_map_view_from_bbox():
    size = (400.0, 400.0, gws.Uom.px)
    bbox = (100.0, 100.0, 500.0, 500.0)
    crs = gws.lib.crs.WGS84
    dpi = 1000
    rotation = 0
    assert render.map_view_from_bbox(size, bbox, crs, dpi, rotation).dpi == 1000
    assert render.map_view_from_bbox(size, bbox, crs, dpi, rotation).rotation == 0
    assert render.map_view_from_bbox(size, bbox, crs, dpi, rotation).pxSize == (400.0, 400.0)
    assert render.map_view_from_bbox(size, bbox, crs, dpi, rotation).mmSize == (10.16, 10.16)
    assert render.map_view_from_bbox(size, bbox, crs, dpi, rotation).center == (300.0, 300.0)
    assert render.map_view_from_bbox(size, bbox, crs, dpi, rotation).scale == 3571
    assert render.map_view_from_bbox(size, bbox, crs, dpi, rotation).bounds.crs == crs
    assert render.map_view_from_bbox(size, bbox, crs, dpi, rotation).bounds.extent == (100.0, 100.0, 500.0, 500.0)


# is it a mapping from the map to px?
def test_map_view_transformer():
    size = (400.0, 400.0, gws.Uom.px)
    bbox = (100.0, 100.0, 500.0, 500.0)
    crs = gws.lib.crs.WGS84
    dpi = 1000
    rotation = 0
    mv = render.map_view_from_bbox(size, bbox, crs, dpi, rotation)
    f = render.map_view_transformer(mv)
    assert f(1, 2) == (-1091, 5490)


def test_map_view_transformer_rotated():
    size = (400.0, 400.0, gws.Uom.px)
    bbox = (100.0, 100.0, 500.0, 500.0)
    crs = gws.lib.crs.WGS84
    dpi = 1000
    rotation = 45
    mv = render.map_view_from_bbox(size, bbox, crs, dpi, rotation)
    f = render.map_view_transformer(mv)
    assert f(1, 2) == (-2449, 2197)


def test_render_map_mm_bbox():
    layer = gws.Layer
    layer.opacity = 0.5

    p1 = gws.MapRenderInputPlane(layer=layer)
    p2 = gws.MapRenderInputPlane(layer=layer)
    p3 = gws.MapRenderInputPlane(layer=layer)

    mri = gws.MapRenderInput(
        backgroundColor=0,
        bbox=(100, 100, 300, 300),
        crs=gws.lib.crs.WGS84,
        dpi=1000,
        mapSize=(200, 200, gws.Uom.mm),
        # notify = print('callable'),
        planes=[p1, p2, p3],
        rotation=0,
        scale=100)

    assert render.render_map(mri).__str__() == (
        "{'planes': [], 'view': {'dpi': 96, 'rotation': 0, 'mmSize': (200, 200), "
        "'pxSize': (755.9055118110236, 755.9055118110236), 'bounds': {'crs': "
        "<crs:4326>, 'extent': (100, 100, 300, 300)}, 'center': (200.0, 200.0), "
        "'scale': 944}}")


def test_render_map_px_center():
    layer = gws.Layer
    layer.opacity = 0.5

    p1 = gws.MapRenderInputPlane(layer=layer)
    p2 = gws.MapRenderInputPlane(layer=layer)
    p3 = gws.MapRenderInputPlane(layer=layer)

    mri = gws.MapRenderInput(
        backgroundColor=0,
        center=(150, 150),
        crs=gws.lib.crs.WGS84,
        dpi=1000,
        mapSize=(200, 200, gws.Uom.px),
        # notify = print('callable'),
        planes=[p1, p2, p3],
        rotation=0,
        scale=100)

    assert render.render_map(mri).__str__() == (
        "{'planes': [], 'view': {'dpi': 96, 'rotation': 0, 'pxSize': (200, 200), "
        "'mmSize': (52.916666666666664, 52.916666666666664), 'center': (150, 150), "
        "'scale': 100, 'bounds': {'crs': <crs:4326>, 'extent': (147.35416666666666, "
        '147.35416666666666, 152.64583333333334, 152.64583333333334)}}}')


def test_output_to_html_element():
    p1 = gws.MapRenderOutputPlane()
    p2 = gws.MapRenderOutputPlane()
    p3 = gws.MapRenderOutputPlane()

    mv = gws.MapView(mmSize=(200, 200))
    mro = gws.MapRenderOutput(view=mv, planes=[p1, p2, p3])

    assert render.output_to_html_element(mro).to_string() == u.fxml('<div '
                                                                    'style="position:relative;overflow:hidden;left:0;top:0;width:200mm;height:200mm">'
                                                                    '<img/></div>')


def test_output_to_html_string():
    p1 = gws.MapRenderOutputPlane()
    p2 = gws.MapRenderOutputPlane()
    p3 = gws.MapRenderOutputPlane()

    mv = gws.MapView(mmSize=(200, 200))
    mro = gws.MapRenderOutput(view=mv, planes=[p1, p2, p3])

    assert render.output_to_html_string(mro) == u.fxml('<div '
                                                       'style="position:relative;overflow:hidden;left:0;top:0;width:200mm;height:200mm">'
                                                       '<img/></div>')
