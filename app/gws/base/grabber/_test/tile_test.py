"""Tests for the tile source grabber."""

import numpy as np

import gws
import gws.base.grabber.core as core
import gws.base.grabber.tile as tile
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


def _color(col, row, z):
    return (col % 256, row % 256, z, 255)


def _tms(crs, max_level):
    return gws.lib.grid.matrix_set_for_grid(gws.lib.grid.for_crs(crs), max_level)


def _windowed_tms(crs, max_level):
    """Matrices covering only the north-west quarter of the default grid."""

    sg = gws.lib.grid.for_crs(crs)
    x0, y0, x1, y1 = sg.extent
    xm = (x0 + x1) / 2
    ym = (y0 + y1) / 2
    ms = []
    for z in range(1, max_level + 1):
        n = 1 << (z - 1)
        ms.append(gws.TileMatrix(
            identifier=str(z),
            resolution=gws.lib.grid.resolution_for_level(sg, z),
            x=x0,
            y=y1,
            width=n,
            height=n,
            tileWidth=sg.tileSize,
            tileHeight=sg.tileSize,
            extent=(x0, ym, xm, y1),
        ))
    return gws.TileMatrixSet(identifier='', crs=crs, matrices=ms)


class FakeTile(tile.Object):
    def __init__(self, opts, source_srid, max_level=20, windowed=False):
        super().__init__(opts)
        self.sourceCrs = gws.lib.crs.get(source_srid)
        fn = _windowed_tms if windowed else _tms
        self.sourceTms = fn(self.sourceCrs, max_level)
        self.fetches = []

    def fetch_tile_as_bytes(self, tm, col, row):
        self.fetches.append((int(tm.identifier), col, row))
        ts = int(tm.tileWidth)
        return gws.lib.image.from_size((ts, ts), color=_color(col, row, int(tm.identifier))).to_bytes(gws.lib.mime.PNG)


def _max_extent(srid):
    crs = gws.lib.crs.get(srid)
    return gws.lib.extent.transform_from_wgs(crs.wgsMaxExtent, crs)


def _opts(srid=3857, max_age=0):
    cache = gws.MapCache(
        name='grabber_' + gws.u.random_string(8),
        maxAge=max_age,
        maxLevel=20,
        requestBuffer=64,
        requestTiles=4,
    )
    return core.Options(
        crs=gws.lib.crs.get(srid),
        cache=cache,
        extent=_max_extent(srid),
        imageFormat=gws.ImageFormat(name='png8', mimeTypes=[gws.lib.mime.PNG], options={'mode': 'P'}),
    )


def _grabber(srid=3857, source_srid=None, **kwargs):
    return FakeTile(_opts(srid), source_srid or srid, **kwargs)


GRID_3857 = gws.lib.grid.for_crs(gws.lib.crs.get(3857))
M = _tms(gws.lib.crs.get(3857), 20).matrices


def _tile_extent(mt):
    return gws.lib.grid.extent_for_tile(GRID_3857, mt)


def _rgba(img):
    return np.array(img.img.convert('RGBA'))


##


def test_no_meta_tiling():
    gr = _grabber()
    assert gr.requestTiles == 1
    gr.get_tiles_as_image_dict((2124, 1364, 2125, 1365, 12))
    assert sorted(gr.fetches) == [(12, 2124, 1364), (12, 2124, 1365), (12, 2125, 1364), (12, 2125, 1365)]


def test_aligned_tile_is_one_source_tile():
    gr = _grabber()
    img = gr.get_tile_as_image((2124, 1364, 12))
    assert gr.fetches == [(12, 2124, 1364)]
    assert img.size() == (256, 256)
    assert (_rgba(img) == _color(2124, 1364, 12)).all()


def test_box_spanning_tiles_is_mosaicked():
    gr = _grabber()
    extent = gws.lib.grid.extent_for_range(GRID_3857, (2124, 1364, 2125, 1365, 12))
    img = gr.compose_box_as_image(extent, 512, 512)
    assert sorted(gr.fetches) == [(12, 2124, 1364), (12, 2124, 1365), (12, 2125, 1364), (12, 2125, 1365)]
    arr = img.to_array()
    assert tuple(arr[10, 10]) == _color(2124, 1364, 12)
    assert tuple(arr[10, 300]) == _color(2125, 1364, 12)
    assert tuple(arr[300, 10]) == _color(2124, 1365, 12)
    assert tuple(arr[300, 300]) == _color(2125, 1365, 12)


def test_box_is_downsampled_from_finer_matrix():
    gr = _grabber()
    extent = _tile_extent((2124, 1364, 12))
    img = gr.compose_box_as_image(extent, 200, 200)
    assert gr.fetches == [(12, 2124, 1364)]
    assert img.size() == (200, 200)


def test_box_beyond_source_max_level_uses_finest_matrix():
    gr = _grabber(max_level=10)
    img = gr.get_tile_as_image((2124, 1364, 12))
    assert gr.fetches == [(10, 531, 341)]
    assert img.size() == (256, 256)
    assert (_rgba(img) == _color(531, 341, 10)).all()


def test_windowed_matrix_serves_tiles_inside_the_window():
    gr = _grabber(windowed=True)
    assert abs(gr.sourceTms.matrices[11].resolution - M[12].resolution) < 1e-9
    img = gr.get_tile_as_image((100, 200, 12))
    assert gr.fetches == [(12, 100, 200)]
    assert (_rgba(img) == _color(100, 200, 12)).all()


def test_tile_outside_source_matrix_is_transparent():
    gr = _grabber(windowed=True)
    img = gr.get_tile_as_image((2124, 1364, 12))
    assert gr.fetches == []
    assert img.size() == (256, 256)
    assert (img.to_array()[..., 3] == 0).all()


def test_cross_crs_tile_is_warped_from_buffered_source_tiles():
    gr = _grabber(srid=25832, source_srid=3857)
    x0, y0, x1, y1, z = gws.lib.grid.range_for_extent(gr.grid, (500000, 5700000, 500001, 5700001), 12)
    img = gr.get_tile_as_image((x0, y0, z))

    assert img.size() == (256, 256)
    assert _rgba(img)[128, 128, 3] == 255
    assert gr.fetches
    assert all(f[0] == 12 for f in gr.fetches)

    src = gws.lib.extent.transform(gws.lib.grid.extent_for_tile(gr.grid, (x0, y0, z)), gr.targetCrs, gr.sourceCrs)
    c0, r0, c1, r1, _ = gws.lib.grid.matrix_range_for_extent(M[12], src)
    assert {(c, r) for _, c, r in gr.fetches} >= {(c, r) for c in range(c0, c1 + 1) for r in range(r0, r1 + 1)}


def test_cross_crs_box_outside_source_is_transparent():
    gr = _grabber(srid=3857, source_srid=25832)
    extent = gws.lib.extent.transform_from_wgs((134, -26, 136, -24), gr.targetCrs)
    img = gr.compose_box_as_image(extent, 100, 100)
    assert img.size() == (100, 100)
    assert (img.to_array()[..., 3] == 0).all()
    assert gr.fetches == []


##


def test_source_tiles_are_cached_between_requests():
    gr = _grabber()
    gr.get_tile_as_image((2124, 1364, 12))
    gr.get_tile_as_image((2124, 1364, 12))
    assert gr.fetches == [(12, 2124, 1364)]

    gr2 = FakeTile(_opts(), 3857)
    gr2.get_tile_as_image((2124, 1364, 12))
    assert gr2.fetches == [(12, 2124, 1364)]


def test_fetch_tile_not_implemented_raises():
    class NoFetchTile(tile.Object):
        def __init__(self, opts):
            super().__init__(opts)
            self.sourceCrs = self.targetCrs
            self.sourceTms = _tms(self.sourceCrs, 20)

    gr = NoFetchTile(_opts())
    with u.raises(NotImplementedError):
        gr.get_tile_as_image((2124, 1364, 12))
