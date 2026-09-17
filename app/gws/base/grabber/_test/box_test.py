"""Tests for the box source grabber."""

import numpy as np

import gws
import gws.base.grabber.box as box
import gws.base.grabber.core as core
import gws.lib.crs
import gws.lib.extent
import gws.lib.grid
import gws.lib.image
import gws.lib.mime
import gws.test.util as u


@u.fixture(autouse=True)
def system_dirs():
    gws.u.ensure_dir(gws.c.LOCKS_DIR)
    gws.u.ensure_dir(gws.c.EPHEMERAL_DIR)
    gws.u.ensure_dir(gws.c.MAP_CACHE_DIR)


RED = (255, 0, 0, 255)


def _max_extent(srid):
    crs = gws.lib.crs.get(srid)
    return gws.lib.extent.transform_from_wgs(crs.wgsMaxExtent, crs)


class FakeBox(box.Object):
    def __init__(self, opts, source_srid):
        super().__init__(opts)
        self.sourceCrs = gws.lib.crs.get(source_srid)
        self.fetches = []

    def fetch_box_as_image(self, bounds, width, height, params=None):
        self.fetches.append((bounds, width, height, params))
        return self.paint(bounds, width, height)

    def paint(self, bounds, width, height):
        return gws.lib.image.from_size((width, height), color=RED)


class PositionBox(FakeBox):
    """Paints R = tile column, G = tile row, relative to the request buffer."""

    def paint(self, bounds, width, height):
        arr = np.zeros((height, width, 4), np.uint8)
        ys, xs = np.mgrid[0:height, 0:width]
        arr[..., 0] = ((xs - self.requestBuffer) // 256) % 256
        arr[..., 1] = ((ys - self.requestBuffer) // 256) % 256
        arr[..., 3] = 255
        return gws.lib.image.from_array(arr)


class CounterBox(FakeBox):
    """Paints R = fetch number."""

    def paint(self, bounds, width, height):
        return gws.lib.image.from_size((width, height), color=(len(self.fetches), 0, 0, 255))


def _opts(srid=3857, max_age=0, extent=None, request_tiles=4, request_buffer=64):
    cache = gws.MapCache(
        name='grabber_' + gws.u.random_string(8),
        maxAge=max_age,
        maxLevel=20,
        requestBuffer=request_buffer,
        requestTiles=request_tiles,
    )
    return core.Options(
        crs=gws.lib.crs.get(srid),
        cache=cache,
        extent=extent or _max_extent(srid),
        imageFormat=gws.ImageFormat(name='png8', mimeTypes=[gws.lib.mime.PNG], options={'mode': 'P'}),
    )


def _grabber(cls=FakeBox, srid=3857, source_srid=None, **kwargs):
    return cls(_opts(srid, **kwargs), source_srid or srid)


GRID_3857 = gws.lib.grid.for_crs(gws.lib.crs.get(3857))


def _range_extent(rng):
    return gws.lib.grid.extent_for_range(GRID_3857, rng)


##


def test_request_shaping_comes_from_cache_config():
    gr = _grabber(request_tiles=3, request_buffer=10)
    assert gr.requestTiles == 3
    assert gr.requestBuffer == 10


def test_block_request_covers_block_plus_buffer():
    gr = _grabber()
    gr.compose_tile_block_as_image_dict((2125, 1365, 12))
    res = gws.lib.grid.resolution_for_level(GRID_3857, 12)
    bounds, w, h, params = gr.fetches[0]
    assert bounds.crs == gr.sourceCrs
    assert bounds.extent == gws.lib.extent.buffer(_range_extent((2124, 1364, 2127, 1367, 12)), 64 * res)
    assert (w, h) == (4 * 256 + 128, 4 * 256 + 128)
    assert params is None


def test_block_is_clipped_to_layer_range():
    gr = _grabber(extent=_range_extent((2125, 1365, 2126, 1367, 12)))
    images = gr.compose_tile_block_as_image_dict((2125, 1365, 12))
    res = gws.lib.grid.resolution_for_level(GRID_3857, 12)
    bounds, w, h, _ = gr.fetches[0]
    assert bounds.extent == gws.lib.extent.buffer(_range_extent((2125, 1365, 2126, 1367, 12)), 64 * res)
    assert (w, h) == (2 * 256 + 128, 3 * 256 + 128)
    assert sorted(images) == [(x, y, 12) for x in (2125, 2126) for y in (1365, 1366, 1367)]


def test_block_tiles_are_cut_from_one_image():
    gr = _grabber(PositionBox)
    images = gr.compose_tile_block_as_image_dict((2124, 1364, 12))
    assert len(images) == 16
    assert len(gr.fetches) == 1
    for (x, y, z), img in images.items():
        assert img.size() == (256, 256)
        arr = img.to_array()
        assert (arr[..., 0] == x - 2124).all()
        assert (arr[..., 1] == y - 1364).all()
        assert (arr[..., 3] == 255).all()


def test_dynamic_block_is_meta_tiled_too():
    gr = _grabber()
    params = {'param_1': 'value_1'}
    images = gr.compose_tile_block_as_image_dict((2125, 1365, 12), params)
    bounds, w, h, p = gr.fetches[0]
    assert len(images) == 16
    assert (w, h) == (4 * 256 + 128, 4 * 256 + 128)
    assert p is params


def test_meta_tiling_shares_one_fetch_per_block():
    gr = _grabber()
    d = gr.get_tiles_as_image_dict((2122, 1362, 2125, 1365, 12))
    assert len(d) == 16
    assert len(gr.fetches) == 4


##


def test_same_crs_box_is_one_fetch():
    gr = _grabber()
    extent = (1e6, 6e6, 1e6 + 3000, 6e6 + 2000)
    img = gr.compose_box_as_image(extent, 300, 200)
    assert img.size() == (300, 200)
    assert (img.to_array() == RED).all()
    bounds, w, h, _ = gr.fetches[0]
    assert bounds.crs == gr.sourceCrs
    assert bounds.extent == extent
    assert (w, h) == (300, 200)


def test_large_box_is_chunked_with_seam_buffer():
    gr = _grabber(CounterBox)
    gr.maxRequestPixels = 512
    extent = (1e6, 6e6, 1e6 + 8000, 6e6 + 3000)
    img = gr.compose_box_as_image(extent, 800, 300)
    assert img.size() == (800, 300)
    assert len(gr.fetches) == 3

    xres = 10
    yres = 10
    b0, w0, h0, _ = gr.fetches[0]
    assert (w0, h0) == (384 + 128, 300 + 128)
    assert b0.extent == (1e6 - 64 * xres, 6e6 + 3000 - (300 + 64) * yres, 1e6 + (384 + 64) * xres, 6e6 + 3000 + 64 * yres)

    b2, w2, h2, _ = gr.fetches[2]
    assert (w2, h2) == (32 + 128, 300 + 128)
    assert b2.extent[0] == 1e6 + (768 - 64) * xres
    assert b2.extent[2] == 1e6 + 8000 + 64 * xres

    for _, w, h, _ in gr.fetches:
        assert w <= 512 and h <= 512

    arr = img.to_array()
    assert (arr[:, 0:384, 0] == 1).all()
    assert (arr[:, 384:768, 0] == 2).all()
    assert (arr[:, 768:800, 0] == 3).all()
    assert (arr[..., 3] == 255).all()


def test_cross_crs_box_is_fetched_in_source_crs_and_warped():
    gr = _grabber(srid=25832, source_srid=3857)
    extent = (500000 - 1280, 5700000 - 1280, 500000 + 1280, 5700000 + 1280)
    img = gr.compose_box_as_image(extent, 256, 256)
    assert img.size() == (256, 256)
    assert tuple(img.to_array()[128, 128]) == RED

    assert len(gr.fetches) == 1
    bounds, w, h, _ = gr.fetches[0]
    assert bounds.crs.srid == 3857
    src = gws.lib.extent.transform(extent, gr.targetCrs, gr.sourceCrs)
    assert bounds.extent[0] < src[0] and bounds.extent[1] < src[1]
    assert bounds.extent[2] > src[2] and bounds.extent[3] > src[3]
    assert 256 < w < 300 and 256 < h < 300


def test_cross_crs_box_outside_source_area_is_transparent():
    gr = _grabber(srid=3857, source_srid=25832)
    extent = gws.lib.extent.transform_from_wgs((134, -26, 136, -24), gr.targetCrs)
    img = gr.compose_box_as_image(extent, 100, 100)
    assert img.size() == (100, 100)
    assert (img.to_array()[..., 3] == 0).all()
    assert gr.fetches == []


def test_transform_resolution():
    crs = gws.lib.crs.get(3857)
    assert abs(crs.transform_resolution((1e6, 6e6, 1e6 + 3000, 6e6 + 2000), 10, crs) - 10) < 1e-6

    crs = gws.lib.crs.get(25832)
    r = crs.transform_resolution((500000 - 1280, 5700000 - 1280, 500000 + 1280, 5700000 + 1280), 10, gws.lib.crs.get(3857))
    assert 15 < r < 17


##


def test_fetch_box_not_implemented_raises():
    class NoFetchBox(box.Object):
        def __init__(self, opts):
            super().__init__(opts)
            self.sourceCrs = self.targetCrs

    gr = NoFetchBox(_opts())
    with u.raises(NotImplementedError):
        gr.get_box_as_image((1e6, 6e6, 1e6 + 3000, 6e6 + 2000), 30, 20)
    with u.raises(NotImplementedError):
        gr.fetch_box_as_bytes(gws.Bounds(crs=gr.sourceCrs, extent=(0, 0, 10, 10)), 20, 10)
