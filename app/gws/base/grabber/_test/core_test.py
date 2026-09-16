"""Tests for the grabber base."""

import os
import time
import unittest.mock as mock

import gws
import gws.base.grabber.box
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
PNG8 = gws.ImageFormat(name='png8', mimeTypes=[gws.lib.mime.PNG], options={'mode': 'P'})


class FakeGrabber(gws.base.grabber.box.Object):
    def __init__(self, opts):
        super().__init__(opts)
        self.sourceCrs = self.targetCrs
        self.fetches = []
        self.error = None

    def fetch_box_as_image(self, bounds, width, height, params=None):
        self.fetches.append((bounds.extent, width, height, params))
        if self.error:
            raise self.error
        return gws.lib.image.from_size((width, height), color=RED)


def _grabber(max_age=0, max_level=14, name=None, srid=3857, extent=None, image_format=None):
    cache = gws.MapCache(
        name=name or 'grabber_' + gws.u.random_string(8),
        maxAge=max_age,
        maxLevel=max_level,
        requestBuffer=64,
        requestTiles=4,
    )
    opts = core.Options(
        crs=gws.lib.crs.get(srid),
        cache=cache,
        extent=extent or _max_extent(srid),
        imageFormat=image_format or PNG8,
        provider=None,
    )
    return FakeGrabber(opts)


def _block_extent(z=12, bx=2124, by=1364, n=4):
    return gws.lib.grid.extent_for_range(gws.lib.grid.for_crs(gws.lib.crs.get(3857)), (bx, by, bx + n - 1, by + n - 1, z))


def _key(params):
    return gws.u.sha256(params)[:12] if params else ''


def _rgba(b):
    return gws.lib.image.from_bytes(b).img.convert('RGBA')


def _alpha(img):
    return img.to_array()[..., 3]


def _block_tiles(z=12, bx=2124, by=1364):
    return [(bx + i, by + j, z) for i in range(4) for j in range(4)]


def test_uncached_block_is_fetched_once():
    gr = _grabber(max_age=0)
    for mt in _block_tiles():
        gr.get_tile_as_bytes(mt)
    assert len(gr.fetches) == 1
    assert gr.fetches[0][1:3] == (4 * 256 + 128, 4 * 256 + 128)


def test_uncached_tiles_go_to_ephemeral_store_only():
    gr = _grabber(max_age=0)
    mt = _block_tiles()[0]
    gr.get_tile_as_bytes(mt)
    assert gr.store_for(15).has(mt, 60)
    assert not gr.store.has(mt, 3600)


def test_cached_tiles_go_to_persistent_store():
    gr = _grabber(max_age=3600, max_level=14)
    mt = _block_tiles(z=12)[0]
    gr.get_tile_as_bytes(mt)
    assert gr.store.has(mt, 3600)
    assert not gr.store_for(15).has(mt, 60)


def test_levels_above_max_level_go_to_ephemeral_store():
    gr = _grabber(max_age=3600, max_level=14)
    mt = _block_tiles(z=15, bx=4248, by=2728)[0]
    gr.get_tile_as_bytes(mt)
    assert gr.store_for(15).has(mt, 60)
    assert not gr.store.has(mt, 3600)


def test_dynamic_requests_are_meta_tiled_into_a_params_keyed_ephemeral_store():
    gr = _grabber(max_age=3600)
    params = {'param_1': 'value_1'}
    tiles = _block_tiles()[:3]
    for mt in tiles:
        gr.get_tile_as_bytes(mt, params)
    assert len(gr.fetches) == 1
    assert gr.fetches[0][1:4] == (4 * 256 + 128, 4 * 256 + 128, params)

    st = gr.store_for(12, _key(params))
    assert st.baseDir != gr.store_for(12).baseDir
    assert st.maxAge == core.EPHEMERAL_MAX_AGE
    for mt in _block_tiles():
        assert st.has(mt, 60)
        assert not gr.store.has(mt, 3600)
        assert not gr.store_for(15).has(mt, 60)


def test_dynamic_requests_with_different_params_use_different_stores():
    gr = _grabber(max_age=0)
    mt = _block_tiles()[0]
    gr.get_tile_as_bytes(mt, {'param_1': 'value_1'})
    gr.get_tile_as_bytes(mt, {'param_1': 'value_1'})
    assert len(gr.fetches) == 1
    gr.get_tile_as_bytes(mt, {'param_1': 'value_2'})
    assert len(gr.fetches) == 2
    gr.get_tile_as_bytes(mt)
    assert len(gr.fetches) == 3
    assert gr.fetches[2][3] is None


def test_dynamic_lock_identity_includes_params():
    gr = _grabber()
    params = {'param_1': 'value_1'}
    with mock.patch.object(gws.u, 'server_lock', wraps=gws.u.server_lock) as lock:
        gr.get_tile_as_bytes((2127, 1367, 12), params)
    lock.assert_called_once_with(f'grabber_{gr.cache.name}_{_key(params)}_12_2124_1364', core.BLOCK_LOCK_TIMEOUT)
    assert _key(params) != _key({'param_1': 'value_2'})
    assert _key(None) == _key({}) == ''


def test_expired_ephemeral_tile_is_fetched_again():
    gr = _grabber(max_age=0)
    mt = _block_tiles()[0]
    gr.get_tile_as_bytes(mt)
    old = time.time() - core.EPHEMERAL_MAX_AGE - 10
    os.utime(gr.store_for(15).path(mt), (old, old))
    gr.get_tile_as_bytes(mt)
    assert len(gr.fetches) == 2


def test_block_is_composed_under_lock_and_reread_after():
    gr = _grabber(max_age=0)
    mt = _block_tiles()[0]
    with mock.patch.object(gr, 'compose_tile_block_as_image_dict', wraps=gr.compose_tile_block_as_image_dict) as compose:
        gr.get_tile_as_bytes(mt)
        with gws.u.server_lock('grabber_' + gr.cache.name + '__12_2124_1364', 0):
            pass
        assert compose.call_count == 1


def test_busy_block_lock_returns_empty_tile():
    gr = _grabber(max_age=0)
    mt = _block_tiles()[0]
    with mock.patch.object(core, 'BLOCK_LOCK_TIMEOUT', 0):
        with gws.u.server_lock('grabber_' + gr.cache.name + '__12_2124_1364', 0):
            b = gr.get_tile_as_bytes(mt)
    assert b == gr.empty_tile()
    assert gr.fetches == []
    assert not gr.store_for(15).has(mt, 60)


def test_store_write_survives_missing_dir(tmp_path):
    st = gws.gis.cache.store.Object(str(tmp_path / 'store_1'), 60, 'png')
    with mock.patch('os.replace', side_effect=FileNotFoundError('gone')):
        st.write((1, 2, 3), b'blob')
    assert st.read((1, 2, 3)) is None


##


def test_defaults():
    gr = _grabber()
    assert gr.targetCrs.srid == 3857
    assert gr.grid.crs.srid == 3857
    assert gr.imageFormat.name == 'png8'
    assert gr.mime == gws.lib.mime.PNG
    assert gr.levels() == list(range(core.MAX_LEVEL + 1))
    assert gr.extent == _max_extent(3857)
    assert gr.tile_range_for_level(0) == (0, 0, 0, 0, 0)
    assert gr.store.baseDir == f'{gws.c.MAP_CACHE_DIR}/{gr.cache.name}'
    assert gr.store_for(15).baseDir == f'{gws.c.EPHEMERAL_DIR}/tiles_{gr.cache.name}_'
    assert gr.store.extension == 'png'


def test_geographic_grid():
    gr = _grabber(srid=4326)
    assert gr.grid.extent == (-180, -90, 180, 90)
    assert gr.tile_range_for_level(0) == (0, 0, 1, 0, 0)
    assert gws.lib.grid.extent_for_tile(gr.grid, (1, 0, 0)) == (0, -90, 180, 90)


def test_configured_extent_limits_tile_ranges():
    gr = _grabber(extent=_block_extent())
    assert gr.tile_range_for_level(12) == (2124, 1364, 2127, 1367, 12)
    assert gr.tile_range_for_level(13) == (4248, 2728, 4255, 2735, 13)
    assert gr.tile_range_for_level(0) == (0, 0, 0, 0, 0)


def test_extent_outside_grid_raises():
    with u.raises(gws.ConfigurationError):
        _grabber(extent=(30e6, 30e6, 31e6, 31e6))


def test_jpeg_format():
    fmt = gws.ImageFormat(name='jpeg', mimeTypes=[gws.lib.mime.JPEG], options={})
    gr = _grabber(image_format=fmt)
    assert gr.mime == gws.lib.mime.JPEG
    assert gr.store.extension == 'jpeg'
    b = gr.get_tile_as_bytes(_block_tiles()[0])
    assert b[:2] == b'\xff\xd8'


##


def test_tile_outside_range_is_transparent_without_fetch():
    gr = _grabber(max_age=3600, extent=_block_extent())
    mt = (2000, 1364, 12)
    b = gr.get_tile_as_bytes(mt)
    assert b == gr.empty_tile()
    assert gr.fetches == []
    assert not gr.store.has(mt, 3600)
    assert not gr.store_for(15).has(mt, 60)


def test_tile_beyond_max_level_is_transparent_without_fetch():
    gr = _grabber()
    b = gr.get_tile_as_bytes((0, 0, core.MAX_LEVEL + 1))
    assert b == gr.empty_tile()
    assert gr.fetches == []


def test_empty_tile_is_transparent():
    gr = _grabber()
    img = _rgba(gr.empty_tile())
    assert img.size == (256, 256)
    assert img.getextrema()[3] == (0, 0)
    assert (_alpha(gr.get_tile_as_image((0, 0, core.MAX_LEVEL + 1))) == 0).all()


def test_get_tile_as_image_matches_bytes():
    gr = _grabber()
    mt = _block_tiles()[0]
    img = gr.get_tile_as_image(mt).img.convert('RGBA')
    assert img.size == (256, 256)
    assert img.getpixel((0, 0)) == RED
    assert _rgba(gr.get_tile_as_bytes(mt)).tobytes() == img.tobytes()
    assert len(gr.fetches) == 1


def test_get_tiles_is_sparse():
    gr = _grabber(extent=_block_extent())
    d = gr.get_tiles_as_bytes_dict((2122, 1362, 2125, 1365, 12))
    assert sorted(d) == [(2124, 1364, 12), (2124, 1365, 12), (2125, 1364, 12), (2125, 1365, 12)]
    assert gr.get_tiles_as_image_dict((0, 0, 1, 1, core.MAX_LEVEL + 1)) == {}
    assert gr.get_tiles_as_image_dict((0, 0, 3, 3, 12)) == {}


def test_cache_hit_reads_store():
    gr = _grabber(max_age=3600)
    mt = _block_tiles()[0]
    gr.get_tile_as_bytes(mt)
    b = gr.get_tile_as_bytes(mt)
    assert len(gr.fetches) == 1
    assert b == gr.store.read(mt)


def test_same_cache_name_shares_store():
    gr1 = _grabber(max_age=3600)
    gr2 = _grabber(max_age=3600, name=gr1.cache.name)
    mt = _block_tiles()[0]
    gr1.get_tile_as_bytes(mt)
    assert gr2.get_tile_as_bytes(mt) == gr1.store.read(mt)
    assert gr2.fetches == []


def test_source_failure_propagates_and_stores_nothing():
    gr = _grabber(max_age=3600)
    gr.error = gws.ExternalServiceError('source_error_1')
    mt = _block_tiles()[0]
    with u.raises(gws.ExternalServiceError):
        gr.get_tile_as_bytes(mt)
    assert not gr.store.has(mt, 3600)
    assert not gr.store_for(15).has(mt, 60)


def test_lock_identity_is_block_snapped():
    gr = _grabber()
    with mock.patch.object(gws.u, 'server_lock', wraps=gws.u.server_lock) as lock:
        gr.get_tile_as_bytes((2127, 1367, 12))
    lock.assert_called_once_with(f'grabber_{gr.cache.name}__12_2124_1364', core.BLOCK_LOCK_TIMEOUT)


def test_ephemeral_block_write_triggers_cleanup():
    gr = _grabber(max_age=0)
    gr.store_for(12)
    with mock.patch.object(gws.u, 'ephemeral_cleanup') as cleanup:
        gr.get_tile_as_bytes(_block_tiles()[0])
    assert cleanup.call_count == 1

    gr = _grabber(max_age=3600)
    with mock.patch.object(gws.u, 'ephemeral_cleanup') as cleanup:
        gr.get_tile_as_bytes(_block_tiles()[0])
    assert cleanup.call_count == 0


def test_store_for():
    gr = _grabber(max_age=3600, max_level=14)
    assert gr.store_for(12) is gr.store
    assert gr.store_for(15) is gr.store_for(16)
    assert gr.store_for(12, _key({'param_1': 'value_1'})).baseDir.endswith(_key({'param_1': 'value_1'}))
    assert gr.is_storing(14)
    assert not gr.is_storing(15)
    assert not _grabber(max_age=0).is_storing(0)


##


def test_uncached_box_composes_at_exact_resolution():
    gr = _grabber(max_age=0)
    extent = (1e6, 6e6, 1e6 + 3000, 6e6 + 2000)
    img = gr.get_box_as_image(extent, 300, 200)
    assert img.size() == (300, 200)
    assert gr.fetches == [(extent, 300, 200, None)]


def test_box_as_bytes_is_encoded_in_image_format():
    gr = _grabber(max_age=0)
    b = gr.get_box_as_bytes((1e6, 6e6, 1e6 + 3000, 6e6 + 2000), 300, 200)
    img = gws.lib.image.from_bytes(b)
    assert b[:8] == b'\x89PNG\r\n\x1a\n'
    assert img.size() == (300, 200)
    assert len(gr.fetches) == 1


def test_cached_box_is_mosaicked_from_stored_tiles():
    gr = _grabber(max_age=3600, max_level=20)
    extent = _block_extent(n=2)
    img = gr.get_box_as_image(extent, 512, 512)
    assert img.size() == (512, 512)
    assert (img.to_array()[..., 3] == 255).all()
    assert len(gr.fetches) == 1
    for mt in _block_tiles():
        assert gr.store.has(mt, 3600)

    gr.get_box_as_bytes(extent, 512, 512)
    assert len(gr.fetches) == 1


def test_cached_box_at_non_aligned_extent_is_warped_from_tiles():
    gr = _grabber(max_age=3600, max_level=20)
    x0, y0, x1, y1 = _block_extent(n=2)
    dx = (x1 - x0) / 10
    img = gr.get_box_as_image((x0 + dx, y0 + dx, x1 - dx, y1 - dx), 300, 300)
    assert img.size() == (300, 300)
    assert tuple(img.to_array()[150, 150]) == RED
    assert all(f[1:3] == (4 * 256 + 128, 4 * 256 + 128) for f in gr.fetches)


def test_cached_box_beyond_max_level_composes_directly():
    gr = _grabber(max_age=3600, max_level=10)
    extent = _block_extent(n=2)
    gr.get_box_as_image(extent, 512, 512)
    assert gr.fetches == [(extent, 512, 512, None)]
    assert gr.store.count() == 0


def test_dynamic_box_bypasses_store():
    gr = _grabber(max_age=3600, max_level=20)
    extent = _block_extent(n=2)
    params = {'param_1': 'value_1'}
    gr.get_box_as_image(extent, 512, 512, params)
    assert gr.fetches == [(extent, 512, 512, params)]
    assert gr.store.count() == 0
    assert gr.store_for(15).count() == 0


def test_cached_box_overlapping_no_data_is_transparent():
    gr = _grabber(max_age=3600, max_level=20, extent=_block_extent())
    extent = _block_extent(bx=2000, by=1000, n=2)
    img = gr.get_box_as_image(extent, 512, 512)
    assert img.size() == (512, 512)
    assert (_alpha(img) == 0).all()
    assert gr.fetches == []


def test_box_size_is_rounded():
    gr = _grabber(max_age=0)
    img = gr.get_box_as_image((1e6, 6e6, 1e6 + 3000, 6e6 + 2000), 300.4, 199.6)
    assert img.size() == (300, 200)


def test_compose_box_not_implemented_in_base():
    gr = core.Object(core.Options(
        crs=gws.lib.crs.get(3857),
        cache=gws.MapCache(name='grabber_' + gws.u.random_string(8), maxAge=0, maxLevel=0, requestBuffer=0, requestTiles=1),
        extent=_max_extent(3857),
        imageFormat=PNG8,
        provider=None,
    ))
    with u.raises(NotImplementedError):
        gr.get_tile_as_bytes(_block_tiles()[0])
    with u.raises(NotImplementedError):
        gr.get_box_as_bytes((1e6, 6e6, 1e6 + 3000, 6e6 + 2000), 30, 20)


def test_block_composition_not_implemented_in_base_for_meta_tiling():
    gr = core.Object(core.Options(
        crs=gws.lib.crs.get(3857),
        cache=gws.MapCache(name='grabber_' + gws.u.random_string(8), maxAge=0, maxLevel=0, requestBuffer=0, requestTiles=1),
        extent=_max_extent(3857),
        imageFormat=PNG8,
        provider=None,
    ))
    gr.requestTiles = 4
    gr.compose_box_as_image = lambda extent, w, h, params=None: gws.lib.image.from_size((w, h))
    with u.raises(NotImplementedError):
        gr.get_tile_as_bytes(_block_tiles()[0])
